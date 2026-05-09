"""
mvs_types.py
------------
Shared data structures for the MVS pipeline.
All pipeline steps communicate through these types.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# GPU / accelerator context  (detected once at import time)
# ---------------------------------------------------------------------------

@dataclass
class AcceleratorInfo:
    """
    Runtime hardware discovery result.
    Used by GPU-variant implementations to pick the right backend.
    """
    has_cuda: bool = False
    has_mps: bool = False        # Apple Silicon
    has_opencl: bool = False
    torch_device: str = "cpu"   # "cuda", "mps", or "cpu"
    cuda_device_count: int = 0
    opencl_platform: Optional[str] = None


def detect_accelerator() -> AcceleratorInfo:
    """Probe available GPU backends without crashing if none present."""
    info = AcceleratorInfo()

    # --- PyTorch / CUDA ---
    try:
        import torch
        if torch.cuda.is_available():
            info.has_cuda = True
            info.torch_device = "cuda"
            info.cuda_device_count = torch.cuda.device_count()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info.has_mps = True
            info.torch_device = "mps"
    except Exception:
        pass

    # --- OpenCL (via pyopencl, optional) ---
    try:
        import pyopencl as cl
        platforms = cl.get_platforms()
        if platforms:
            info.has_opencl = True
            info.opencl_platform = platforms[0].name
    except (ImportError, Exception):
        pass

    return info


ACCELERATOR = detect_accelerator()


# ---------------------------------------------------------------------------
# Camera (reuse-compatible with sfm_types, but self-contained)
# ---------------------------------------------------------------------------

@dataclass
class MVSCamera:
    """Intrinsics + extrinsics for one image."""
    image_id:    int
    image_path:  str
    width:       int
    height:      int
    K:           np.ndarray    # (3,3) float64
    dist_coeffs: np.ndarray    # (5,)  float64
    R:           np.ndarray    # (3,3) float64  world→camera
    t:           np.ndarray    # (3,)  float64  world→camera

    @property
    def P(self) -> np.ndarray:
        """3×4 projection matrix  P = K [R | t]"""
        return self.K @ np.hstack([self.R, self.t.reshape(3, 1)])

    @property
    def centre(self) -> np.ndarray:
        """World-space camera centre  C = -R^T t"""
        return -self.R.T @ self.t

    def project(self, pts3d: np.ndarray) -> np.ndarray:
        """
        Project (N,3) world points → (N,2) pixel coords.
        Returns NaN for points behind the camera.
        """
        pts_c = (self.R @ pts3d.T).T + self.t          # (N,3) camera space
        valid = pts_c[:, 2] > 0
        px = np.full((len(pts3d), 2), np.nan)
        if valid.any():
            proj, _ = __import__("cv2").projectPoints(
                pts3d[valid].reshape(-1, 1, 3),
                __import__("cv2").Rodrigues(self.R)[0],
                self.t.reshape(3, 1),
                self.K, self.dist_coeffs,
            )
            px[valid] = proj.reshape(-1, 2)
        return px


# ---------------------------------------------------------------------------
# Depth Map
# ---------------------------------------------------------------------------

@dataclass
class DepthMap:
    """Per-pixel depth for one image."""
    image_id:   int
    depth:      np.ndarray          # (H,W) float32  — metres; 0 = invalid
    confidence: Optional[np.ndarray] = None  # (H,W) float32  [0,1]
    normal_map: Optional[np.ndarray] = None  # (H,W,3) float32 camera-space normals
    width:      int = 0
    height:     int = 0

    def __post_init__(self):
        if self.width == 0:
            self.width = self.depth.shape[1]
        if self.height == 0:
            self.height = self.depth.shape[0]

    def valid_mask(self) -> np.ndarray:
        """Boolean mask: True where depth > 0."""
        return self.depth > 0

    def valid_count(self) -> int:
        return int(self.valid_mask().sum())


# ---------------------------------------------------------------------------
# Dense Point
# ---------------------------------------------------------------------------

@dataclass
class DensePoint:
    """A single point in the dense cloud."""
    xyz:        np.ndarray   # (3,) float64
    color:      np.ndarray   # (3,) uint8   RGB
    normal:     Optional[np.ndarray] = None   # (3,) float64
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Dense Point Cloud
# ---------------------------------------------------------------------------

@dataclass
class DensePointCloud:
    """
    The main output of the MVS pipeline.
    Stores points as parallel arrays for efficiency.
    """
    points:     np.ndarray                    # (N,3) float64
    colors:     np.ndarray                    # (N,3) uint8
    normals:    Optional[np.ndarray] = None   # (N,3) float64
    confidences: Optional[np.ndarray] = None  # (N,)  float32

    def __len__(self) -> int:
        return len(self.points)

    def append(self, other: "DensePointCloud") -> "DensePointCloud":
        """Concatenate two clouds."""
        pts = np.vstack([self.points, other.points])
        col = np.vstack([self.colors, other.colors])
        nrm = None
        if self.normals is not None or other.normals is not None:
            self_normals = self.normals
            other_normals = other.normals
            if self_normals is None:
                self_normals = np.full((len(self.points), 3), np.nan, dtype=np.float64)
            if other_normals is None:
                other_normals = np.full((len(other.points), 3), np.nan, dtype=np.float64)
            nrm = np.vstack([self_normals, other_normals])
        conf = None
        if self.confidences is not None or other.confidences is not None:
            self_conf = self.confidences
            other_conf = other.confidences
            if self_conf is None:
                self_conf = np.full((len(self.points),), np.nan, dtype=np.float32)
            if other_conf is None:
                other_conf = np.full((len(other.points),), np.nan, dtype=np.float32)
            conf = np.concatenate([self_conf, other_conf])
        return DensePointCloud(pts, col, nrm, conf)

    @staticmethod
    def empty() -> "DensePointCloud":
        return DensePointCloud(
            points=np.empty((0, 3), dtype=np.float64),
            colors=np.empty((0, 3), dtype=np.uint8),
        )


# ---------------------------------------------------------------------------
# MVS Reconstruction  (living state object)
# ---------------------------------------------------------------------------

@dataclass
class MVSReconstruction:
    """
    Holds the full MVS state.
    Every pipeline step reads from / writes to this object.
    """
    # --- inputs (populated from SfM output) ---
    cameras:       Dict[int, MVSCamera]    = field(default_factory=dict)
    sparse_points: Optional[np.ndarray]   = None   # (M,3) from SfM
    sparse_colors: Optional[np.ndarray]   = None   # (M,3) uint8

    # --- intermediate ---
    depth_maps:    Dict[int, DepthMap]     = field(default_factory=dict)
    fused_depth:   Dict[int, DepthMap]     = field(default_factory=dict)

    # --- output ---
    dense_cloud:   DensePointCloud         = field(default_factory=DensePointCloud.empty)

    # --- meta ---
    accelerator:   AcceleratorInfo         = field(default_factory=detect_accelerator)
