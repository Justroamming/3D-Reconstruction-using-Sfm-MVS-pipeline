"""
sfm_types.py
------------
Shared data structures for the SfM pipeline.
All pipeline steps communicate through these types.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def normalize_pair_key(image_id_a: int, image_id_b: int) -> Tuple[int, int]:
    """Normalize pair key so (a,b) and (b,a) map to the same canonical form."""
    return (min(image_id_a, image_id_b), max(image_id_a, image_id_b))


# ---------------------------------------------------------------------------
# Feature Detection / Description
# ---------------------------------------------------------------------------

@dataclass
class Keypoints:
    """Detected keypoints for a single image."""
    image_id: int
    points: np.ndarray          # (N, 2) float32 — pixel (x, y)
    sizes: np.ndarray           # (N,)   float32 — keypoint diameter
    angles: np.ndarray          # (N,)   float32 — orientation in degrees
    responses: np.ndarray       # (N,)   float32 — detector response / score
    descriptors: np.ndarray     # (N, D) float32 or uint8 — feature vectors


# ---------------------------------------------------------------------------
# Feature Matching
# ---------------------------------------------------------------------------

@dataclass
class ImageMatches:
    """Raw matches between two images (before geometric verification)."""
    image_id_a: int
    image_id_b: int
    # Each row: [idx_in_A, idx_in_B]
    matches: np.ndarray         # (M, 2) int32


# ---------------------------------------------------------------------------
# Geometric Verification
# ---------------------------------------------------------------------------

@dataclass
class VerifiedMatches:
    """Geometrically verified matches between two images."""
    image_id_a: int
    image_id_b: int
    matches: np.ndarray         # (M, 2) int32  — inlier matches
    F: Optional[np.ndarray]     # (3, 3) Fundamental matrix (or None)
    H: Optional[np.ndarray]     # (3, 3) Homography (or None)
    inlier_ratio: float         # fraction of original matches kept
    num_inliers: int


# ---------------------------------------------------------------------------
# Camera Model
# ---------------------------------------------------------------------------

@dataclass
class Camera:
    """Intrinsic parameters for one camera model."""
    camera_id: int
    model: str                  # "pinhole", "radial", "opencv", …
    width: int
    height: int
    K: np.ndarray               # (3, 3) intrinsic matrix
    dist_coeffs: np.ndarray     # distortion coefficients (k1,k2,p1,p2,…)


# ---------------------------------------------------------------------------
# Registered Image (camera pose)
# ---------------------------------------------------------------------------

@dataclass
class RegisteredImage:
    """A camera that has been placed in the reconstruction."""
    image_id: int
    camera_id: int
    R: np.ndarray               # (3, 3) rotation    — world→camera
    t: np.ndarray               # (3,)   translation — world→camera
    # Projection matrix P = K [R | t]
    @property
    def P(self) -> np.ndarray:
        """Return [R | t] (3×4 matrix). Caller multiplies by K to get P = K[R|t]."""
        return np.hstack([self.R, self.t.reshape(3, 1)])


# ---------------------------------------------------------------------------
# 3-D Point
# ---------------------------------------------------------------------------

@dataclass
class Point3D:
    """A reconstructed 3-D point with its 2-D track."""
    point3d_id: int
    xyz: np.ndarray             # (3,)  float64
    color: np.ndarray           # (3,)  uint8  RGB
    error: float                # mean reprojection error (pixels)
    # track: list of (image_id, keypoint_idx) observations
    track: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Reconstruction  (the living state object passed through the pipeline)
# ---------------------------------------------------------------------------

@dataclass
class Reconstruction:
    """
    Holds the full incremental SfM state.
    Every pipeline step reads from and/or writes to this object.
    """
    # --- inputs ---
    image_paths: list = field(default_factory=list)   # ordered list of file paths
    cameras: dict = field(default_factory=dict)        # camera_id → Camera
    keypoints: dict = field(default_factory=dict)      # image_id  → Keypoints

    # --- matching ---
    image_matches: dict = field(default_factory=dict)      # (id_a,id_b) → ImageMatches
    verified_matches: dict = field(default_factory=dict)   # (id_a,id_b) → VerifiedMatches

    # --- reconstruction ---
    registered_images: dict = field(default_factory=dict)  # image_id → RegisteredImage
    points3d: dict = field(default_factory=dict)           # point3d_id → Point3D

    # --- bookkeeping ---
    next_point_id: int = 0
    component_id: Optional[int] = None   # NEW: informational, which seed this grew from

    def new_point_id(self) -> int:
        pid = self.next_point_id
        self.next_point_id += 1
        return pid

    def spawn_component(self, component_id: Optional[int] = None) -> "Reconstruction":
        """
        Create a new, independent Reconstruction that shares this dataset's
        global, read-only data (image_paths, cameras, keypoints, matches) but
        starts with EMPTY per-component state (registered_images, points3d).

        Used for multi-seed incremental reconstruction: each component grows
        its own registered_images/points3d independently while reading from
        the same underlying feature/match data. Safe because Steps 7-10 never
        mutate cameras/keypoints/verified_matches, only registered_images and
        points3d (which are fresh, unshared dicts here).
        """
        comp = Reconstruction()
        comp.image_paths = self.image_paths
        comp.cameras = self.cameras
        comp.keypoints = self.keypoints
        comp.image_matches = self.image_matches
        comp.verified_matches = self.verified_matches
        comp.component_id = component_id
        return comp
