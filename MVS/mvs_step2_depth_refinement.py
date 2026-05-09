"""
mvs_step2_depth_refinement.py
------------------------------
Step 2 — Depth Map Refinement

Clean up each raw depth map before fusion.

Swappable strategies
  CPU:
    - BilateralRefiner          — bilateral filter (edge-preserving smooth)
    - ConfidenceThresholdRefiner— mask out low-confidence pixels
    - HoleFillRefiner           — inpaint small invalid regions
    - ChainRefiner              — apply multiple refiners in sequence
  GPU:
    - BilateralRefinerGPU       — bilateral filter on GPU (PyTorch / CUDA)
    - JointBilateralRefinerGPU  — guided by colour image (better edge preservation)
"""

from __future__ import annotations
import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional

from mvs_types import ACCELERATOR, DepthMap, MVSCamera, MVSReconstruction


# ============================================================
#  INTERFACE
# ============================================================

class DepthRefinerBase(ABC):
    @abstractmethod
    def refine(self, depth_map: DepthMap, camera: MVSCamera) -> DepthMap:
        """Return a refined copy of the depth map."""


# ============================================================
#  CPU IMPLEMENTATIONS
# ============================================================

class BilateralRefiner(DepthRefinerBase):
    """
    Edge-preserving smoothing via joint bilateral filter.
    Uses the depth map itself as guide signal.
    Invalid pixels (depth == 0) are excluded from averaging.
    """

    def __init__(self, d: int = 9, sigma_color: float = 75.0, sigma_space: float = 75.0,
                 n_iters: int = 1):
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.n_iters = n_iters

    def refine(self, depth_map: DepthMap, camera: MVSCamera) -> DepthMap:
        depth = depth_map.depth.copy()
        valid = depth > 0

        # Normalise to [0,255] for cv2.bilateralFilter
        d_max = depth[valid].max() if valid.any() else 1.0
        d_norm = (depth / d_max * 255).astype(np.float32)

        for _ in range(self.n_iters):
            d_filtered = cv2.bilateralFilter(d_norm, self.d,
                                             self.sigma_color, self.sigma_space)
            # Restore invalid pixels
            d_filtered[~valid] = 0.0
            d_norm = d_filtered

        depth_out = (d_norm / 255.0 * d_max).astype(np.float32)
        depth_out[~valid] = 0.0

        conf = depth_map.confidence.copy() if depth_map.confidence is not None else None
        return DepthMap(image_id=depth_map.image_id, depth=depth_out, confidence=conf)


class ConfidenceThresholdRefiner(DepthRefinerBase):
    """
    Remove depth estimates below a confidence threshold.
    Simple but effective first-pass filter.
    """

    def __init__(self, min_confidence: float = 0.3):
        self.min_conf = min_confidence

    def refine(self, depth_map: DepthMap, camera: MVSCamera) -> DepthMap:
        depth = depth_map.depth.copy()
        conf  = depth_map.confidence

        if conf is not None:
            low_conf = conf < self.min_conf
            depth[low_conf] = 0.0
            conf_out = conf.copy()
            conf_out[low_conf] = 0.0
        else:
            conf_out = None

        return DepthMap(image_id=depth_map.image_id, depth=depth, confidence=conf_out)


class HoleFillRefiner(DepthRefinerBase):
    """
    Fill small holes (invalid regions) in the depth map using:
      1. Morphological closing to fill tiny gaps
      2. Inpainting for larger holes (up to `max_hole_size` pixels)
    """

    def __init__(self, max_hole_size: int = 50, morph_size: int = 3):
        self.max_hole_size = max_hole_size
        self.morph_size = morph_size

    def refine(self, depth_map: DepthMap, camera: MVSCamera) -> DepthMap:
        depth = depth_map.depth.copy()
        valid = (depth > 0).astype(np.uint8)
        newly_filled = np.zeros_like(valid, dtype=bool)

        # Morphological closing to fill tiny single-pixel gaps
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_size, self.morph_size)
        )
        closed_valid = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, kernel)

        # Find small holes (connected components of invalid pixels)
        inv_mask = (closed_valid == 0).astype(np.uint8)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv_mask)

        fill_mask = np.zeros_like(inv_mask)
        for label in range(1, n_labels):
            if stats[label, cv2.CC_STAT_AREA] <= self.max_hole_size:
                fill_mask[labels == label] = 1

        if fill_mask.any():
            # Inpaint: normalise depth, fill, restore scale
            d_max = depth[valid == 1].max() if (valid == 1).any() else 1.0
            d_u8  = np.clip(depth / d_max * 255, 0, 255).astype(np.uint8)
            d_filled = cv2.inpaint(d_u8, fill_mask, inpaintRadius=3,
                                   flags=cv2.INPAINT_TELEA)
            # Write filled values only into hole pixels
            newly_filled = (fill_mask == 1)
            depth[newly_filled] = d_filled[newly_filled].astype(np.float32) / 255.0 * d_max

        conf = depth_map.confidence.copy() if depth_map.confidence is not None else None
        if conf is not None:
            conf[depth == 0] = 0.0
            if newly_filled.any():
                conf[newly_filled] = np.maximum(conf[newly_filled], 0.25)
        return DepthMap(image_id=depth_map.image_id, depth=depth, confidence=conf)


class MedianRefiner(DepthRefinerBase):
    """
    Median filter — removes salt-and-pepper noise without blurring edges.
    Only filters valid pixels; zeros remain zero.
    """

    def __init__(self, kernel_size: int = 5):
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.ksize = kernel_size

    def refine(self, depth_map: DepthMap, camera: MVSCamera) -> DepthMap:
        depth = depth_map.depth.copy()
        valid = depth > 0

        d_max = depth[valid].max() if valid.any() else 1.0
        d_u8  = np.clip(depth / d_max * 255, 0, 255).astype(np.uint8)
        d_med = cv2.medianBlur(d_u8, self.ksize)

        depth_out = d_med.astype(np.float32) / 255.0 * d_max
        depth_out[~valid] = 0.0

        conf = depth_map.confidence.copy() if depth_map.confidence is not None else None
        return DepthMap(image_id=depth_map.image_id, depth=depth_out, confidence=conf)


class GuidedFilterRefiner(DepthRefinerBase):
    """
    Guided filter (He et al. 2013) using the colour image as the guide.
    Transfers edge structure from the colour image to the depth map.
    """

    def __init__(self, radius: int = 16, eps: float = 0.01):
        self.radius = radius
        self.eps = eps

    def _guided_filter(self, guide: np.ndarray, src: np.ndarray,
                        r: int, eps: float) -> np.ndarray:
        """Guided filter (single-channel guide and src, float32 [0,1])."""
        mean_I   = cv2.boxFilter(guide, cv2.CV_64F, (r, r))
        mean_p   = cv2.boxFilter(src,   cv2.CV_64F, (r, r))
        mean_Ip  = cv2.boxFilter(guide * src, cv2.CV_64F, (r, r))
        cov_Ip   = mean_Ip - mean_I * mean_p
        mean_II  = cv2.boxFilter(guide * guide, cv2.CV_64F, (r, r))
        var_I    = mean_II - mean_I * mean_I
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
        return (mean_a * guide + mean_b).astype(np.float32)

    def refine(self, depth_map: DepthMap, camera: MVSCamera) -> DepthMap:
        depth = depth_map.depth.copy()
        valid = depth > 0

        # Load colour guide
        try:
            guide_bgr = cv2.imread(camera.image_path)
            if guide_bgr is not None:
                guide_bgr = cv2.resize(guide_bgr, (depth.shape[1], depth.shape[0]))
                guide = cv2.cvtColor(guide_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
            else:
                guide = depth.astype(np.float64)
        except Exception:
            guide = depth.astype(np.float64)

        d_max = float(depth[valid].max()) if valid.any() else 1.0
        d_norm = (depth / d_max).astype(np.float64)

        filtered = self._guided_filter(guide, d_norm, self.radius, self.eps)
        filtered = np.clip(filtered, 0, 1).astype(np.float32) * d_max
        filtered[~valid] = 0.0

        conf = depth_map.confidence.copy() if depth_map.confidence is not None else None
        return DepthMap(image_id=depth_map.image_id, depth=filtered, confidence=conf)


class ChainRefiner(DepthRefinerBase):
    """Apply multiple refiners in sequence."""

    def __init__(self, refiners: List[DepthRefinerBase]):
        self.refiners = refiners

    def refine(self, depth_map: DepthMap, camera: MVSCamera) -> DepthMap:
        dm = depth_map
        for r in self.refiners:
            dm = r.refine(dm, camera)
        return dm


# ============================================================
#  GPU IMPLEMENTATIONS
# ============================================================

class BilateralRefinerGPU(DepthRefinerBase):
    """
    Bilateral filter on GPU using PyTorch.
    Separable approximation: applies 1-D spatial × range kernels.
    Falls back to CPU bilateral if PyTorch unavailable.
    """

    def __init__(self, kernel_size: int = 9, sigma_color: float = 0.1,
                 sigma_space: float = 5.0):
        self.ksize = kernel_size
        self.sc = sigma_color
        self.ss = sigma_space

    def refine(self, depth_map: DepthMap, camera: MVSCamera) -> DepthMap:
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            return BilateralRefiner(self.ksize, self.sc * 255, self.ss).refine(depth_map, camera)

        device = torch.device(ACCELERATOR.torch_device)
        depth = depth_map.depth.copy()
        valid = depth > 0

        d_max = float(depth[valid].max()) if valid.any() else 1.0
        d_t = torch.from_numpy(depth / d_max).float().to(device).unsqueeze(0).unsqueeze(0)

        H, W = depth.shape
        ks = self.ksize
        half = ks // 2

        # Build spatial kernel weights
        coords = torch.arange(-half, half + 1, device=device, dtype=torch.float32)
        spatial_w = torch.exp(-coords**2 / (2 * self.ss**2))

        # Horizontal pass
        padded = F.pad(d_t, (half, half, 0, 0), mode='replicate')
        output_h = torch.zeros_like(d_t)
        weight_h  = torch.zeros_like(d_t)
        for i, cx in enumerate(range(-half, half + 1)):
            shifted = padded[:, :, :, i:i + W]
            range_w = torch.exp(-((shifted - d_t) ** 2) / (2 * self.sc ** 2))
            w = spatial_w[i] * range_w
            output_h += w * shifted
            weight_h  += w
        output_h = output_h / weight_h.clamp(1e-8)

        # Vertical pass
        padded_v = F.pad(output_h, (0, 0, half, half), mode='replicate')
        output   = torch.zeros_like(output_h)
        weight_v  = torch.zeros_like(output_h)
        for i, cy in enumerate(range(-half, half + 1)):
            shifted = padded_v[:, :, i:i + H, :]
            range_w = torch.exp(-((shifted - output_h) ** 2) / (2 * self.sc ** 2))
            w = spatial_w[i] * range_w
            output += w * shifted
            weight_v += w
        output = output / weight_v.clamp(1e-8)

        depth_out = (output.squeeze().cpu().numpy() * d_max).astype(np.float32)
        depth_out[~valid] = 0.0

        conf = depth_map.confidence.copy() if depth_map.confidence is not None else None
        return DepthMap(image_id=depth_map.image_id, depth=depth_out, confidence=conf)


class JointBilateralRefinerGPU(DepthRefinerBase):
    """
    Joint / cross bilateral filter: spatial weights from the colour image,
    range weights from the depth map.
    Preserves colour edges in the depth (avoids depth bleeding).
    Falls back to GuidedFilterRefiner if PyTorch unavailable.
    """

    def __init__(self, kernel_size: int = 11, sigma_color: float = 0.1,
                 sigma_space: float = 7.0):
        self.ksize = kernel_size
        self.sc = sigma_color
        self.ss = sigma_space

    def refine(self, depth_map: DepthMap, camera: MVSCamera) -> DepthMap:
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            return GuidedFilterRefiner().refine(depth_map, camera)

        device = torch.device(ACCELERATOR.torch_device)
        depth = depth_map.depth.copy()
        valid = depth > 0

        guide_bgr = cv2.imread(camera.image_path)
        if guide_bgr is None:
            return BilateralRefinerGPU(self.ksize, self.sc, self.ss).refine(depth_map, camera)
        guide_bgr = cv2.resize(guide_bgr, (depth.shape[1], depth.shape[0]))
        guide_gray = cv2.cvtColor(guide_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        H, W = depth.shape
        d_max = float(depth[valid].max()) if valid.any() else 1.0
        d_t = torch.from_numpy(depth / d_max).float().to(device)
        g_t = torch.from_numpy(guide_gray).float().to(device)

        ks = self.ksize; half = ks // 2
        coords = torch.arange(-half, half+1, device=device, dtype=torch.float32)
        sp_w = torch.exp(-coords**2 / (2 * self.ss**2))

        # Unfold for vectorised patch computation
        d_4d = d_t.unsqueeze(0).unsqueeze(0)
        g_4d = g_t.unsqueeze(0).unsqueeze(0)
        d_pad = F.pad(d_4d, (half,half,half,half), mode='replicate')
        g_pad = F.pad(g_4d, (half,half,half,half), mode='replicate')

        output = torch.zeros(H, W, device=device)
        weight = torch.zeros(H, W, device=device)

        for dy in range(ks):
            for dx in range(ks):
                d_shift = d_pad[0, 0, dy:dy+H, dx:dx+W]
                g_shift = g_pad[0, 0, dy:dy+H, dx:dx+W]
                sw = sp_w[dy] * sp_w[dx]
                rw = torch.exp(-((d_shift - d_t)**2) / (2 * self.sc**2))
                w  = sw * rw
                output += w * d_shift
                weight += w

        depth_out = (output / weight.clamp(1e-8)).cpu().numpy().astype(np.float32) * d_max
        depth_out[~valid] = 0.0

        conf = depth_map.confidence.copy() if depth_map.confidence is not None else None
        return DepthMap(image_id=depth_map.image_id, depth=depth_out, confidence=conf)


# ============================================================
#  PIPELINE STEP FUNCTION
# ============================================================

def refine_depth_maps(
    reconstruction: MVSReconstruction,
    refiner: Optional[DepthRefinerBase] = None,
    image_ids: Optional[List[int]] = None,
) -> dict:
    """
    Step 2 — Refine all raw depth maps.

    Parameters
    ----------
    reconstruction : MVSReconstruction (depth_maps modified in-place)
    refiner        : DepthRefinerBase (default: ChainRefiner with conf + bilateral + holefill)
    image_ids      : subset to process (default: all with a depth map)

    Returns
    -------
    dict image_id → refined DepthMap
    """
    if refiner is None:
        if ACCELERATOR.has_cuda or ACCELERATOR.has_mps:
            refiner = ChainRefiner([
                ConfidenceThresholdRefiner(0.2),
                JointBilateralRefinerGPU(kernel_size=9),
                HoleFillRefiner(max_hole_size=100),
            ])
        else:
            refiner = ChainRefiner([
                ConfidenceThresholdRefiner(0.2),
                BilateralRefiner(d=9, sigma_color=50, sigma_space=50),
                HoleFillRefiner(max_hole_size=100),
            ])

    ids = image_ids if image_ids is not None else list(reconstruction.depth_maps.keys())
    print(f"[Step 2] Depth refinement ({refiner.__class__.__name__}) — {len(ids)} maps")

    refined = {}
    for iid in ids:
        if iid not in reconstruction.cameras:
            print(f"  [img {iid:03d}] skipped: missing camera in reconstruction.cameras")
            continue
        dm  = reconstruction.depth_maps[iid]
        cam = reconstruction.cameras[iid]
        dm_r = refiner.refine(dm, cam)
        reconstruction.depth_maps[iid] = dm_r
        refined[iid] = dm_r
        print(f"  [img {iid:03d}] valid: {dm.valid_count()} → {dm_r.valid_count()}")

    return refined
