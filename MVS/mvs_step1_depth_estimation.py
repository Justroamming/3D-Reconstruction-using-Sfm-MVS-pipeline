"""
mvs_step1_depth_estimation.py
------------------------------
Step 1 — Depth Map Estimation

Compute a per-pixel depth map for every registered image.

Swappable strategies
  CPU:
    - PatchMatchCPU        — classical PatchMatch stereo (NumPy/SciPy)
    - SGMDepthEstimator    — Semi-Global Matching via OpenCV StereoSGBM
    - PlaneSweeepCPU       — plane sweep stereo (multi-view cost volume)
  GPU:
    - PatchMatchGPU        — PatchMatch on GPU via PyTorch (CUDA / MPS / CPU-fallback)
    - PlaneSweepGPU        — plane-sweep cost volume on GPU via PyTorch
  DL placeholder:
    - MVSNetEstimator      — plug-in for MVSNet / IterMVS inference
"""

from __future__ import annotations
import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from mvs_types import ACCELERATOR, DepthMap, MVSCamera, MVSReconstruction


# ============================================================
#  SHARED HELPERS
# ============================================================

def _load_gray(path: str, width: int = 0, height: int = 0) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load {path}")
    if width and height:
        img = cv2.resize(img, (width, height))
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


def _load_color(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load {path}")
    return img                    # BGR uint8


def _select_src_views(
    ref_cam: MVSCamera,
    all_cameras: dict,
    n_views: int = 4,
) -> List[MVSCamera]:
    """
    Pick the `n_views` neighbours most likely to share overlap with `ref_cam`.
    Ranks by angle between optical axes (smaller = more frontal overlap).
    """
    ref_z = ref_cam.R[2]          # optical axis direction (world space)
    scores = []
    for cid, cam in all_cameras.items():
        if cid == ref_cam.image_id:
            continue
        src_z = cam.R[2]
        cos_a = float(np.clip(ref_z @ src_z, -1, 1))
        dist  = float(np.linalg.norm(ref_cam.centre - cam.centre))
        scores.append((cid, cos_a, dist))

    # Prefer large angle (frontal) AND moderate distance
    scores.sort(key=lambda x: -x[1])
    return [all_cameras[s[0]] for s in scores[:n_views]]


def _depth_range_from_sparse(
    ref_cam: MVSCamera,
    sparse_pts: Optional[np.ndarray],
    margin: float = 0.1,
) -> Tuple[float, float]:
    """Estimate min/max depth from the sparse point cloud."""
    if sparse_pts is None or len(sparse_pts) == 0:
        return 0.1, 100.0

    pts_c = (ref_cam.R @ sparse_pts.T).T + ref_cam.t
    depths = pts_c[:, 2]
    depths = depths[depths > 0]
    if len(depths) == 0:
        return 0.1, 100.0

    d_min = max(0.01, float(np.percentile(depths, 5))  * (1 - margin))
    d_max = float(np.percentile(depths, 95)) * (1 + margin)
    return d_min, d_max


def _ncc_patch(
    img_ref: np.ndarray,
    img_src: np.ndarray,
    y: int, x: int,
    y_s: float, x_s: float,
    half: int = 4,
) -> float:
    """Normalised Cross-Correlation of two patches (CPU)."""
    H, W = img_ref.shape
    Hs, Ws = img_src.shape

    y1, y2 = y - half, y + half + 1
    x1, x2 = x - half, x + half + 1
    if y1 < 0 or y2 > H or x1 < 0 or x2 > W:
        return -1.0

    ys1 = int(round(y_s)) - half; ys2 = ys1 + 2 * half + 1
    xs1 = int(round(x_s)) - half; xs2 = xs1 + 2 * half + 1
    if ys1 < 0 or ys2 > Hs or xs1 < 0 or xs2 > Ws:
        return -1.0

    p_ref = img_ref[y1:y2, x1:x2].ravel().astype(np.float32)
    p_src = img_src[ys1:ys2, xs1:xs2].ravel().astype(np.float32)

    p_ref -= p_ref.mean(); p_src -= p_src.mean()
    n_ref = np.linalg.norm(p_ref); n_src = np.linalg.norm(p_src)
    if n_ref < 1e-6 or n_src < 1e-6:
        return -1.0
    return float(np.dot(p_ref, p_src) / (n_ref * n_src))


# ============================================================
#  INTERFACE
# ============================================================

class DepthEstimatorBase(ABC):
    @abstractmethod
    def estimate(
        self,
        ref_cam: MVSCamera,
        src_cams: List[MVSCamera],
        reconstruction: MVSReconstruction,
    ) -> DepthMap:
        """
        Estimate a depth map for `ref_cam` using `src_cams` as reference views.

        Returns
        -------
        DepthMap for ref_cam
        """


# ============================================================
#  CPU IMPLEMENTATIONS
# ============================================================

class PatchMatchCPU(DepthEstimatorBase):
    """
    Classical PatchMatch stereo depth estimation (CPU).

    For each pixel in the reference image:
      1. Initialise a random depth hypothesis
      2. Propagate good depths from neighbours
      3. Evaluate with NCC patch matching across source views
      4. Refine with random perturbation

    Pure NumPy — slow but no GPU required.
    """

    def __init__(
        self,
        n_iters: int = 3,
        patch_half: int = 4,
        n_depth_samples: int = 64,
        n_src_views: int = 4,
        image_scale: float = 1.0,
    ):
        self.n_iters = n_iters
        self.patch_half = patch_half
        self.n_depth_samples = n_depth_samples
        self.n_src_views = n_src_views
        self.scale = image_scale

    def estimate(self, ref_cam, src_cams, reconstruction) -> DepthMap:
        d_min, d_max = _depth_range_from_sparse(
            ref_cam, reconstruction.sparse_points
        )

        img_ref = _load_gray(ref_cam.image_path)
        H, W = img_ref.shape

        if self.scale != 1.0:
            W2 = int(W * self.scale); H2 = int(H * self.scale)
            img_ref = cv2.resize(img_ref, (W2, H2))
            H, W = H2, W2

        src_imgs = [_load_gray(c.image_path, W, H) for c in src_cams]

        # Scale intrinsics if needed
        K_ref = ref_cam.K.copy()
        if self.scale != 1.0:
            K_ref[:2] *= self.scale

        # --- Initialise random depths ---
        depth = np.random.uniform(d_min, d_max, (H, W)).astype(np.float32)
        cost  = np.full((H, W), -1.0, dtype=np.float32)

        def eval_depth(y_r, x_r, d):
            """Evaluate cost for pixel (y_r, x_r) at depth d."""
            # Back-project to 3-D
            pt3d = d * (np.linalg.inv(K_ref) @ np.array([x_r, y_r, 1.0]))
            pt_w = ref_cam.R.T @ (pt3d - ref_cam.t)
            best = -1.0
            for cam_s, img_s in zip(src_cams, src_imgs):
                px = cam_s.project(pt_w.reshape(1, 3))[0]
                if np.isnan(px).any():
                    continue
                score = _ncc_patch(img_ref, img_s, y_r, x_r,
                                   px[1], px[0], self.patch_half)
                best = max(best, score)
            return best

        # --- PatchMatch iterations ---
        for iteration in range(self.n_iters):
            # Forward sweep
            for y in range(1, H - 1):
                for x in range(1, W - 1):
                    # Propagate from left / top
                    for dy, dx in [(0, -1), (-1, 0)]:
                        ny, nx = y + dy, x + dx
                        d_prop = depth[ny, nx]
                        c_prop = eval_depth(y, x, d_prop)
                        if c_prop > cost[y, x]:
                            depth[y, x] = d_prop
                            cost[y, x]  = c_prop
                    # Random search
                    r = (d_max - d_min) / 2
                    while r > 0.01:
                        d_rand = np.clip(
                            depth[y, x] + np.random.uniform(-r, r),
                            d_min, d_max
                        )
                        c_rand = eval_depth(y, x, d_rand)
                        if c_rand > cost[y, x]:
                            depth[y, x] = d_rand
                            cost[y, x]  = c_rand
                        r /= 2

            # Backward sweep (alternate direction)
            for y in range(H - 2, 0, -1):
                for x in range(W - 2, 0, -1):
                    for dy, dx in [(0, 1), (1, 0)]:
                        ny, nx = y + dy, x + dx
                        d_prop = depth[ny, nx]
                        c_prop = eval_depth(y, x, d_prop)
                        if c_prop > cost[y, x]:
                            depth[y, x] = d_prop
                            cost[y, x]  = c_prop

        # Mask out low-confidence pixels
        depth[cost < 0.1] = 0.0
        conf = np.clip((cost + 1) / 2, 0, 1).astype(np.float32)
        conf[depth == 0] = 0.0

        return DepthMap(
            image_id=ref_cam.image_id,
            depth=depth,
            confidence=conf,
        )


class SGMDepthEstimator(DepthEstimatorBase):
    """
    Semi-Global Matching depth estimator using OpenCV StereoSGBM.

    Works best for forward-facing stereo pairs (small baseline).
    Projects source images into the reference frame via homography
    for each neighbouring view, then aggregates.
    """

    def __init__(
        self,
        min_disp: int = 0,
        num_disp: int = 128,
        block_size: int = 5,
        n_src_views: int = 2,
        image_scale: float = 1.0,
    ):
        self.min_disp  = min_disp
        self.num_disp  = num_disp
        self.block_size = block_size
        self.n_src_views = n_src_views
        self.scale = image_scale

        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8  * 3 * block_size ** 2,
            P2=32 * 3 * block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def estimate(self, ref_cam, src_cams, reconstruction) -> DepthMap:
        d_min, d_max = _depth_range_from_sparse(
            ref_cam, reconstruction.sparse_points
        )

        img_ref_bgr = _load_color(ref_cam.image_path)
        H, W = img_ref_bgr.shape[:2]
        if self.scale != 1.0:
            W2 = int(W * self.scale); H2 = int(H * self.scale)
            img_ref_bgr = cv2.resize(img_ref_bgr, (W2, H2))
            H, W = H2, W2

        img_ref_gray = cv2.cvtColor(img_ref_bgr, cv2.COLOR_BGR2GRAY)

        depth_accum = np.zeros((H, W), dtype=np.float64)
        weight_accum = np.zeros((H, W), dtype=np.float64)

        K_ref = ref_cam.K.copy()
        if self.scale != 1.0:
            K_ref[:2] *= self.scale

        for src_cam in src_cams[:self.n_src_views]:
            img_src = _load_color(src_cam.image_path)
            if self.scale != 1.0:
                img_src = cv2.resize(img_src, (W, H))

            # Rectify src into ref frame via fundamental-matrix-based stereo rectification
            R_rel = src_cam.R @ ref_cam.R.T
            t_rel = src_cam.t - R_rel @ ref_cam.t

            K_src = src_cam.K.copy()
            if self.scale != 1.0:
                K_src[:2] *= self.scale

            R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
                K_ref, ref_cam.dist_coeffs,
                K_src,  src_cam.dist_coeffs,
                (W, H), R_rel, t_rel,
                flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
            )
            map1_x, map1_y = cv2.initUndistortRectifyMap(
                K_ref, ref_cam.dist_coeffs, R1, P1, (W, H), cv2.CV_32FC1
            )
            map2_x, map2_y = cv2.initUndistortRectifyMap(
                K_src, src_cam.dist_coeffs, R2, P2, (W, H), cv2.CV_32FC1
            )

            rect_ref = cv2.remap(img_ref_gray, map1_x, map1_y, cv2.INTER_LINEAR)
            rect_src = cv2.remap(
                cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY),
                map2_x, map2_y, cv2.INTER_LINEAR
            )

            disp = self.sgbm.compute(rect_ref, rect_src).astype(np.float32) / 16.0

            # Convert disparity → depth using Q matrix
            pts_3d = cv2.reprojectImageTo3D(disp, Q)
            depth_sgm = pts_3d[:, :, 2].astype(np.float32)

            valid = (depth_sgm > d_min) & (depth_sgm < d_max) & (disp > self.min_disp)

            # Warp depth back to original (un-rectified) reference frame
            # map1_x/map1_y map rectified pixels -> original reference pixels.
            # Scatter valid rectified depths into original image coordinates.
            depth_rect = depth_sgm * valid.astype(np.float32)
            x_orig = np.rint(map1_x).astype(np.int32)
            y_orig = np.rint(map1_y).astype(np.int32)
            in_bounds = (
                valid
                & (x_orig >= 0) & (x_orig < W)
                & (y_orig >= 0) & (y_orig < H)
            )

            depth_warped = np.zeros((H, W), dtype=np.float32)
            contrib_count = np.zeros((H, W), dtype=np.float32)
            np.add.at(depth_warped, (y_orig[in_bounds], x_orig[in_bounds]), depth_rect[in_bounds])
            np.add.at(contrib_count, (y_orig[in_bounds], x_orig[in_bounds]), 1.0)
            depth_warped = np.where(
                contrib_count > 0,
                depth_warped / np.maximum(contrib_count, 1e-8),
                0.0,
            ).astype(np.float32)

            weight = (depth_warped > 0).astype(np.float64)
            depth_accum  += depth_warped.astype(np.float64) * weight
            weight_accum += weight

        depth_out = np.where(
            weight_accum > 0,
            (depth_accum / np.maximum(weight_accum, 1e-8)).astype(np.float32),
            0.0,
        )

        conf = np.clip(weight_accum / max(1, len(src_cams[:self.n_src_views])),
                       0, 1).astype(np.float32)

        return DepthMap(
            image_id=ref_cam.image_id,
            depth=depth_out,
            confidence=conf,
        )


class PlaneSweepCPU(DepthEstimatorBase):
    """
    Plane Sweep Stereo (CPU).

    Sweeps a set of fronto-parallel depth planes, warps source images
    into the reference frame at each depth, and picks the depth with
    the minimum photo-consistency cost (NCC).
    """

    def __init__(
        self,
        n_planes: int = 64,
        patch_half: int = 4,
        n_src_views: int = 4,
        image_scale: float = 1.0,
    ):
        self.n_planes    = n_planes
        self.patch_half  = patch_half
        self.n_src_views = n_src_views
        self.scale       = image_scale

    def estimate(self, ref_cam, src_cams, reconstruction) -> DepthMap:
        d_min, d_max = _depth_range_from_sparse(
            ref_cam, reconstruction.sparse_points
        )
        depths_to_try = np.linspace(d_min, d_max, self.n_planes)

        img_ref = _load_gray(ref_cam.image_path)
        H, W = img_ref.shape
        if self.scale != 1.0:
            W2 = int(W * self.scale); H2 = int(H * self.scale)
            img_ref = cv2.resize(img_ref, (W2, H2))
            H, W = H2, W2

        src_imgs = [_load_gray(c.image_path, W, H) for c in src_cams[:self.n_src_views]]

        K_ref = ref_cam.K.copy()
        if self.scale != 1.0:
            K_ref[:2] *= self.scale

        K_ref_inv = np.linalg.inv(K_ref)
        best_cost  = np.full((H, W), -1.0, dtype=np.float32)
        best_depth = np.zeros((H, W), dtype=np.float32)

        # Pixel grid (homogeneous)
        ys, xs = np.mgrid[0:H, 0:W]
        pix_h = np.stack([xs.ravel(), ys.ravel(), np.ones(H * W)], axis=0)  # (3, H*W)
        rays  = K_ref_inv @ pix_h   # (3, H*W) — normalised directions

        for d in depths_to_try:
            pts3d_ref = d * rays   # (3, H*W) in camera space
            # To world
            pts3d_w = (ref_cam.R.T @ (pts3d_ref.T - ref_cam.t).T).T  # (H*W, 3)

            cost_sum = np.zeros(H * W, dtype=np.float32)
            n_valid  = np.zeros(H * W, dtype=np.int32)

            for cam_s, img_s in zip(src_cams[:self.n_src_views], src_imgs):
                K_s = cam_s.K.copy()
                if self.scale != 1.0:
                    K_s[:2] *= self.scale
                # Project pts3d_w into source camera
                pts_c = (cam_s.R @ pts3d_w.T).T + cam_s.t   # (H*W, 3)
                valid_depth = pts_c[:, 2] > 0
                px = np.full((H * W, 2), -1.0)
                p  = (K_s @ pts_c[valid_depth].T).T
                px[valid_depth, 0] = p[:, 0] / p[:, 2]
                px[valid_depth, 1] = p[:, 1] / p[:, 2]

                # Bilinear sample source image at projected positions
                map_x = px[:, 0].reshape(H, W).astype(np.float32)
                map_y = px[:, 1].reshape(H, W).astype(np.float32)
                warped = cv2.remap(img_s, map_x, map_y, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)

                # NCC via sliding window
                ph = self.patch_half
                ref_blur  = cv2.blur(img_ref,  (2*ph+1, 2*ph+1))
                warp_blur = cv2.blur(warped, (2*ph+1, 2*ph+1))
                ref_sq_blur  = cv2.blur(img_ref**2,   (2*ph+1, 2*ph+1))
                warp_sq_blur = cv2.blur(warped**2,  (2*ph+1, 2*ph+1))
                cross_blur   = cv2.blur(img_ref * warped, (2*ph+1, 2*ph+1))

                var_r  = ref_sq_blur  - ref_blur**2
                var_s  = warp_sq_blur - warp_blur**2
                cov    = cross_blur   - ref_blur * warp_blur
                denom  = np.sqrt(np.maximum(var_r * var_s, 1e-8))
                ncc    = np.clip(cov / denom, -1, 1)

                in_bounds = valid_depth.reshape(H, W) & \
                            (map_x >= 0) & (map_x < W - 1) & \
                            (map_y >= 0) & (map_y < H - 1)

                cost_sum += ncc.ravel() * in_bounds.ravel()
                n_valid  += in_bounds.ravel().astype(np.int32)

            mean_cost = np.where(n_valid > 0, cost_sum / np.maximum(n_valid, 1), -1.0)
            mean_cost = mean_cost.reshape(H, W)

            improve = mean_cost > best_cost
            best_cost[improve]  = mean_cost[improve]
            best_depth[improve] = d

        best_depth[best_cost < 0.1] = 0.0
        conf = np.clip((best_cost + 1) / 2, 0, 1).astype(np.float32)
        conf[best_depth == 0] = 0.0

        return DepthMap(
            image_id=ref_cam.image_id,
            depth=best_depth,
            confidence=conf,
        )


# ============================================================
#  GPU IMPLEMENTATIONS (PyTorch — CUDA / MPS / CPU fallback)
# ============================================================

class PlaneSweepGPU(DepthEstimatorBase):
    """
    Plane Sweep Stereo accelerated with PyTorch.
    Automatically uses CUDA → MPS → CPU depending on hardware.

    The entire cost volume is computed as batched tensor operations,
    giving a 10–50× speedup over the CPU version on modern GPUs.
    """

    def __init__(
        self,
        n_planes: int = 128,
        patch_half: int = 4,
        n_src_views: int = 4,
        image_scale: float = 1.0,
    ):
        self.n_planes    = n_planes
        self.patch_half  = patch_half
        self.n_src_views = n_src_views
        self.scale       = image_scale
        self._device     = None    # lazy init

    def _get_device(self):
        if self._device is not None:
            return self._device
        try:
            import torch
            dev = ACCELERATOR.torch_device
            self._device = torch.device(dev)
            print(f"  [PlaneSweepGPU] using torch device: {dev}")
        except ImportError:
            self._device = "cpu_fallback"
        return self._device

    def estimate(self, ref_cam, src_cams, reconstruction) -> DepthMap:
        device = self._get_device()

        # Fallback to CPU if torch unavailable
        if device == "cpu_fallback":
            print("  [PlaneSweepGPU] PyTorch not available, falling back to PlaneSweepCPU")
            return PlaneSweepCPU(
                n_planes=self.n_planes,
                patch_half=self.patch_half,
                n_src_views=self.n_src_views,
                image_scale=self.scale,
            ).estimate(ref_cam, src_cams, reconstruction)

        import torch
        import torch.nn.functional as F

        d_min, d_max = _depth_range_from_sparse(
            ref_cam, reconstruction.sparse_points
        )
        depths_to_try = np.linspace(d_min, d_max, self.n_planes)

        img_ref = _load_gray(ref_cam.image_path)
        H, W = img_ref.shape
        if self.scale != 1.0:
            W2 = int(W * self.scale); H2 = int(H * self.scale)
            img_ref = cv2.resize(img_ref, (W2, H2))
            H, W = H2, W2

        src_imgs = [_load_gray(c.image_path, W, H) for c in src_cams[:self.n_src_views]]

        K_ref = ref_cam.K.copy()
        if self.scale != 1.0:
            K_ref[:2] *= self.scale
        K_ref_inv = np.linalg.inv(K_ref)

        # Move to GPU
        ref_t = torch.from_numpy(img_ref).to(device).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        src_ts = [
            torch.from_numpy(s).to(device).unsqueeze(0).unsqueeze(0)
            for s in src_imgs
        ]

        # Pixel rays in reference frame
        ys, xs = np.mgrid[0:H, 0:W]
        pix_h = np.stack([xs.ravel(), ys.ravel(), np.ones(H * W)], axis=0)
        rays  = torch.from_numpy((K_ref_inv @ pix_h).T.astype(np.float32)).to(device)  # (H*W, 3)

        best_cost  = torch.full((H, W), -1.0, device=device, dtype=torch.float32)
        best_depth = torch.zeros((H, W), device=device, dtype=torch.float32)

        ph = self.patch_half
        kernel = torch.ones(1, 1, 2*ph+1, 2*ph+1, device=device) / (2*ph+1)**2

        for d in depths_to_try:
            d_t = float(d)
            pts3d_cam = d_t * rays        # (H*W, 3)
            # world coordinates
            R_ref_t = torch.from_numpy(ref_cam.R.T.astype(np.float32)).to(device)
            t_ref   = torch.from_numpy(ref_cam.t.astype(np.float32)).to(device)
            pts3d_w = (R_ref_t @ (pts3d_cam - t_ref).T).T   # (H*W, 3)

            cost_sum = torch.zeros(H * W, device=device, dtype=torch.float32)
            n_valid  = torch.zeros(H * W, device=device, dtype=torch.float32)

            for cam_s, src_t in zip(src_cams[:self.n_src_views], src_ts):
                K_s = cam_s.K.copy()
                if self.scale != 1.0:
                    K_s[:2] *= self.scale
                K_st = torch.from_numpy(K_s.astype(np.float32)).to(device)
                R_st = torch.from_numpy(cam_s.R.astype(np.float32)).to(device)
                t_st = torch.from_numpy(cam_s.t.astype(np.float32)).to(device)

                # Project world → source image
                pts_c = (R_st @ pts3d_w.T).T + t_st          # (H*W, 3)
                valid = pts_c[:, 2] > 0
                px = torch.full((H * W, 2), -1.0, device=device)
                if valid.any():
                    proj = (K_st @ pts_c[valid].T).T
                    px[valid, 0] = proj[:, 0] / proj[:, 2]
                    px[valid, 1] = proj[:, 1] / proj[:, 2]

                # Normalise to [-1,1] for grid_sample
                gx = (px[:, 0] / (W - 1)) * 2 - 1
                gy = (px[:, 1] / (H - 1)) * 2 - 1
                grid = torch.stack([gx, gy], dim=-1).reshape(1, H, W, 2)

                # Sample source image
                warped = F.grid_sample(src_t, grid, mode='bilinear',
                                       padding_mode='zeros', align_corners=True)

                # NCC via convolution
                def local_mean(x):
                    return F.conv2d(x, kernel, padding=ph)

                ref_mean   = local_mean(ref_t)
                warp_mean  = local_mean(warped)
                ref_var    = local_mean(ref_t**2)   - ref_mean**2
                warp_var   = local_mean(warped**2)  - warp_mean**2
                cross      = local_mean(ref_t * warped) - ref_mean * warp_mean
                ncc = (cross / (torch.sqrt(ref_var.clamp(1e-8) * warp_var.clamp(1e-8)))).clamp(-1, 1)

                in_bounds = valid & (px[:, 0] >= 0) & (px[:, 0] < W - 1) & \
                                    (px[:, 1] >= 0) & (px[:, 1] < H - 1)

                cost_sum += ncc.reshape(H * W) * in_bounds.float()
                n_valid  += in_bounds.float()

            mean_cost = torch.where(n_valid > 0, cost_sum / n_valid.clamp(1), torch.full_like(cost_sum, -1.0))
            mean_cost = mean_cost.reshape(H, W)

            improve = mean_cost > best_cost
            best_cost  = torch.where(improve, mean_cost, best_cost)
            best_depth = torch.where(improve, torch.full_like(best_depth, d_t), best_depth)

        # Back to numpy
        depth_np = best_depth.cpu().numpy().astype(np.float32)
        cost_np  = best_cost.cpu().numpy().astype(np.float32)
        depth_np[cost_np < 0.1] = 0.0
        conf_np = np.clip((cost_np + 1) / 2, 0, 1).astype(np.float32)
        conf_np[depth_np == 0] = 0.0

        return DepthMap(
            image_id=ref_cam.image_id,
            depth=depth_np,
            confidence=conf_np,
        )


class PatchMatchGPU(DepthEstimatorBase):
    """
    GPU-accelerated PatchMatch via PyTorch.
    Runs random initialisation + propagation + refinement on GPU tensors.
    Falls back to PatchMatchCPU if PyTorch is unavailable.
    """

    def __init__(
        self,
        n_iters: int = 3,
        patch_half: int = 4,
        n_src_views: int = 4,
        image_scale: float = 1.0,
    ):
        self.n_iters    = n_iters
        self.patch_half = patch_half
        self.n_src_views = n_src_views
        self.scale      = image_scale

    def estimate(self, ref_cam, src_cams, reconstruction) -> DepthMap:
        try:
            import torch
            device = torch.device(ACCELERATOR.torch_device)
        except ImportError:
            print("  [PatchMatchGPU] falling back to PatchMatchCPU")
            return PatchMatchCPU(
                n_iters=self.n_iters,
                patch_half=self.patch_half,
                n_src_views=self.n_src_views,
                image_scale=self.scale,
            ).estimate(ref_cam, src_cams, reconstruction)

        import torch
        import torch.nn.functional as F

        d_min, d_max = _depth_range_from_sparse(
            ref_cam, reconstruction.sparse_points
        )

        img_ref = _load_gray(ref_cam.image_path)
        H, W = img_ref.shape
        if self.scale != 1.0:
            W2 = int(W * self.scale); H2 = int(H * self.scale)
            img_ref = cv2.resize(img_ref, (W2, H2))
            H, W = H2, W2

        src_imgs = [_load_gray(c.image_path, W, H) for c in src_cams[:self.n_src_views]]

        K_ref = ref_cam.K.copy()
        if self.scale != 1.0:
            K_ref[:2] *= self.scale
        K_ref_inv_t = torch.from_numpy(np.linalg.inv(K_ref).astype(np.float32)).to(device)

        ref_t  = torch.from_numpy(img_ref).to(device)
        src_ts = [torch.from_numpy(s).to(device) for s in src_imgs]

        # Pixel grid
        ys_t, xs_t = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij',
        )
        ones = torch.ones(H, W, device=device, dtype=torch.float32)
        pix_h = torch.stack([xs_t, ys_t, ones], dim=-1).reshape(-1, 3)  # (H*W,3)
        rays  = (K_ref_inv_t @ pix_h.T).T                               # (H*W,3)

        # Initialise random depth
        depth = torch.empty(H, W, device=device).uniform_(d_min, d_max)
        cost  = torch.full((H, W), -1.0, device=device)

        def eval_batch(depth_flat):
            """Evaluate NCC cost for a flat (H*W,) depth tensor."""
            pts3d_cam = depth_flat.unsqueeze(1) * rays   # (H*W, 3)
            R_ref_t_t = torch.from_numpy(ref_cam.R.T.astype(np.float32)).to(device)
            t_ref_t   = torch.from_numpy(ref_cam.t.astype(np.float32)).to(device)
            pts3d_w   = (R_ref_t_t @ (pts3d_cam - t_ref_t).T).T

            total_cost = torch.full((H * W,), -1.0, device=device)
            n_valid    = torch.zeros(H * W, device=device)

            ph = self.patch_half
            kernel = torch.ones(1, 1, 2*ph+1, 2*ph+1, device=device) / (2*ph+1)**2

            for cam_s, src_t in zip(src_cams[:self.n_src_views], src_ts):
                K_s = cam_s.K.copy()
                if self.scale != 1.0:
                    K_s[:2] *= self.scale
                K_st = torch.from_numpy(K_s.astype(np.float32)).to(device)
                R_st = torch.from_numpy(cam_s.R.astype(np.float32)).to(device)
                t_st = torch.from_numpy(cam_s.t.astype(np.float32)).to(device)

                pts_c = (R_st @ pts3d_w.T).T + t_st
                valid = pts_c[:, 2] > 0
                px = torch.full((H*W, 2), -1.0, device=device)
                if valid.any():
                    proj = (K_st @ pts_c[valid].T).T
                    px[valid, 0] = proj[:, 0] / proj[:, 2]
                    px[valid, 1] = proj[:, 1] / proj[:, 2]

                in_bounds = valid & (px[:,0]>=0)&(px[:,0]<W-1)&(px[:,1]>=0)&(px[:,1]<H-1)

                gx = (px[:, 0] / (W-1)) * 2 - 1
                gy = (px[:, 1] / (H-1)) * 2 - 1
                grid = torch.stack([gx, gy], dim=-1).reshape(1, H, W, 2)
                src_4d = src_t.unsqueeze(0).unsqueeze(0)
                ref_4d = ref_t.unsqueeze(0).unsqueeze(0)
                warped = F.grid_sample(src_4d, grid, mode='bilinear',
                                       padding_mode='zeros', align_corners=True)

                def lmean(x):
                    return F.conv2d(x, kernel, padding=ph)
                rm = lmean(ref_4d); wm = lmean(warped)
                ncc = (lmean(ref_4d*warped) - rm*wm) / (
                    torch.sqrt((lmean(ref_4d**2)-rm**2).clamp(1e-8) *
                               (lmean(warped**2)-wm**2).clamp(1e-8))
                )
                ncc = ncc.reshape(H*W).clamp(-1, 1)

                total_cost = torch.where(in_bounds & (ncc > total_cost), ncc, total_cost)
                n_valid += in_bounds.float()

            return total_cost.reshape(H, W)

        # --- PatchMatch iterations ---
        for it in range(self.n_iters):
            new_cost = eval_batch(depth.reshape(-1))
            improve = new_cost > cost
            cost = torch.where(improve, new_cost, cost)

            # Spatial propagation: shift depth candidates
            for shift_y, shift_x in [(0, -1), (-1, 0), (0, 1), (1, 0)]:
                d_prop = torch.roll(depth, shifts=(shift_y, shift_x), dims=(0, 1))
                c_prop = eval_batch(d_prop.reshape(-1))
                improve = c_prop > cost
                cost  = torch.where(improve, c_prop, cost)
                depth = torch.where(improve, d_prop, depth)

            # Random perturbation
            radius = (d_max - d_min) * 0.5 * (0.5 ** it)
            d_rand = (depth + torch.empty_like(depth).uniform_(-radius, radius)).clamp(d_min, d_max)
            c_rand = eval_batch(d_rand.reshape(-1))
            improve = c_rand > cost
            cost  = torch.where(improve, c_rand, cost)
            depth = torch.where(improve, d_rand, depth)

        depth_np = depth.cpu().numpy().astype(np.float32)
        cost_np  = cost.cpu().numpy().astype(np.float32)
        depth_np[cost_np < 0.1] = 0.0
        conf_np = np.clip((cost_np + 1) / 2, 0, 1).astype(np.float32)
        conf_np[depth_np == 0] = 0.0

        return DepthMap(image_id=ref_cam.image_id, depth=depth_np, confidence=conf_np)


class MVSNetEstimator(DepthEstimatorBase):
    """
    Placeholder for deep-learning MVS networks (MVSNet, IterMVS, TransMVSNet …).
    Supply a `model` object implementing:
        depth, conf = model.infer(ref_image, src_images, proj_matrices, depth_range)
    """

    def __init__(self, model=None):
        if model is None:
            raise ValueError("MVSNetEstimator requires a model. Pass model=<your network>.")
        self.model = model

    def estimate(self, ref_cam, src_cams, reconstruction) -> DepthMap:
        d_min, d_max = _depth_range_from_sparse(ref_cam, reconstruction.sparse_points)
        ref_img = _load_color(ref_cam.image_path)
        src_imgs = [_load_color(c.image_path) for c in src_cams]
        depth, conf = self.model.infer(ref_img, src_imgs, ref_cam, src_cams, (d_min, d_max))
        return DepthMap(image_id=ref_cam.image_id, depth=depth, confidence=conf)


# ============================================================
#  PIPELINE STEP FUNCTION
# ============================================================

def estimate_depth_maps(
    reconstruction: MVSReconstruction,
    estimator: Optional[DepthEstimatorBase] = None,
    n_src_views: int = 4,
    image_ids: Optional[List[int]] = None,
) -> Dict[int, DepthMap]:
    """
    Step 1 — Estimate depth maps for all (or selected) registered images.

    Parameters
    ----------
    reconstruction : MVSReconstruction
    estimator      : DepthEstimatorBase (default: PlaneSweepCPU)
    n_src_views    : number of source views per reference image
    image_ids      : subset of image IDs to process (default: all)

    Returns
    -------
    depth_maps : dict image_id → DepthMap
    """
    if estimator is None:
        if ACCELERATOR.has_cuda or ACCELERATOR.has_mps:
            estimator = PlaneSweepGPU(n_planes=64, image_scale=0.5)
        else:
            estimator = PlaneSweepCPU(n_planes=32, image_scale=0.5)

    ids = image_ids if image_ids is not None else list(reconstruction.cameras.keys())
    depth_maps: Dict[int, DepthMap] = {}

    print(f"[Step 1] Depth estimation ({estimator.__class__.__name__}) — {len(ids)} images")

    for ref_id in ids:
        ref_cam = reconstruction.cameras[ref_id]
        src_cams = _select_src_views(ref_cam, reconstruction.cameras, n_src_views)

        dm = estimator.estimate(ref_cam, src_cams, reconstruction)
        depth_maps[ref_id] = dm
        reconstruction.depth_maps[ref_id] = dm

        valid = dm.valid_count()
        total = dm.width * dm.height
        print(f"  [img {ref_id:03d}] valid={valid}/{total} ({100*valid/max(total,1):.1f}%)")

    return depth_maps
