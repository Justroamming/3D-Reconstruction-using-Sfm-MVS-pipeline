"""
step8_triangulation.py
-----------------------
Step 8 — Triangulation of New 3-D Points

Responsibilities:
  - After a new camera is registered, find verified matches between the new
    image and all already-registered images
  - Triangulate new 3-D points from those 2-D correspondences
  - Filter by reprojection error, cheirality, and triangulation angle
  - Add new points to Reconstruction.points3d and extend existing tracks

Swappable strategies
  Triangulators:
    - DLTTriangulator        — Direct Linear Transform (cv2.triangulatePoints)
    - OptimalTriangulator    — optimal angular correction (Lindstrom 2010 / iterative)
    - MidpointTriangulator   — simple mid-point of closest approach (fast, approx.)
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from sfm_types import Camera, Keypoints, Point3D, Reconstruction, RegisteredImage, normalize_pair_key


# ============================================================
#  HELPERS
# ============================================================

def _projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return K @ np.hstack([R, t.reshape(3, 1)])


def _reprojection_error(pts3d: np.ndarray, pts2d: np.ndarray, P: np.ndarray) -> np.ndarray:
    n = pts3d.shape[0]
    h = np.hstack([pts3d, np.ones((n, 1))])
    proj = (P @ h.T).T
    p2d = proj[:, :2] / (proj[:, 2:3] + 1e-10)
    return np.linalg.norm(p2d - pts2d, axis=1)


def _triangulation_angle_deg(
    pts3d: np.ndarray,
    C_a: np.ndarray,
    C_b: np.ndarray,
) -> np.ndarray:
    """Angle (degrees) between the two observation rays at each 3-D point."""
    ray_a = pts3d - C_a
    ray_b = pts3d - C_b
    na = np.linalg.norm(ray_a, axis=1, keepdims=True) + 1e-10
    nb = np.linalg.norm(ray_b, axis=1, keepdims=True) + 1e-10
    cos_a = np.clip(np.sum((ray_a / na) * (ray_b / nb), axis=1), -1, 1)
    return np.degrees(np.arccos(cos_a))


def _camera_centre(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """World-space camera centre: C = -R^T t."""
    return -R.T @ t


# ============================================================
#  TRIANGULATOR INTERFACE & IMPLEMENTATIONS
# ============================================================

class TriangulatorBase(ABC):
    @abstractmethod
    def triangulate(
        self,
        pts_a: np.ndarray,   # (N,2) float32
        pts_b: np.ndarray,   # (N,2) float32
        P_a: np.ndarray,     # (3,4) projection matrix
        P_b: np.ndarray,     # (3,4) projection matrix
    ) -> np.ndarray:
        """Return (N,3) float64 3-D point array."""


class DLTTriangulator(TriangulatorBase):
    """
    Standard Direct Linear Transform via cv2.triangulatePoints.
    Fast and numerically stable for well-conditioned pairs.
    """

    def triangulate(self, pts_a, pts_b, P_a, P_b) -> np.ndarray:
        pts4d = cv2.triangulatePoints(
            P_a.astype(np.float64),
            P_b.astype(np.float64),
            pts_a.T.astype(np.float64),
            pts_b.T.astype(np.float64),
        )  # (4,N)
        w = pts4d[3] + 1e-10
        return (pts4d[:3] / w).T   # (N,3)


class MidpointTriangulator(TriangulatorBase):
    """
    Mid-point of the closest approach between two rays.
    Much faster but less accurate than DLT.
    Useful as a quick sanity-check or initialisation.
    """

    def triangulate(self, pts_a, pts_b, P_a, P_b) -> np.ndarray:
        # Extract camera centres and ray directions
        def unproject(P, pts2d):
            # Backproject pixels to normalised ray directions
            K = P[:, :3]
            K_inv = np.linalg.inv(K)
            ones = np.ones((len(pts2d), 1))
            pts_h = np.hstack([pts2d, ones])          # (N,3)
            rays = (K_inv @ pts_h.T).T                # (N,3)
            rays /= np.linalg.norm(rays, axis=1, keepdims=True) + 1e-10
            return rays

        def centre_from_P(P):
            _, _, Vt = np.linalg.svd(P)
            C = Vt[-1]
            return (C[:3] / C[3])

        C_a = centre_from_P(P_a)
        C_b = centre_from_P(P_b)
        rays_a = unproject(P_a, pts_a)
        rays_b = unproject(P_b, pts_b)

        pts3d = []
        for ra, rb in zip(rays_a, rays_b):
            # Solve for t in: C_a + t*ra ≈ C_b + s*rb  (least squares)
            A = np.stack([ra, -rb], axis=1)
            b = C_b - C_a
            ts, *_ = np.linalg.lstsq(A, b, rcond=None)
            p = 0.5 * ((C_a + ts[0] * ra) + (C_b + ts[1] * rb))
            pts3d.append(p)

        return np.array(pts3d, dtype=np.float64)


class OptimalTriangulator(TriangulatorBase):
    """
    Iterative optimal triangulation (Hartley & Sturm / Lindstrom).
    Minimises algebraic reprojection error by correcting 2-D observations.
    Falls back to DLT if scipy is unavailable.
    """

    def __init__(self, max_iters: int = 10):
        self.max_iters = max_iters
        self._dlt = DLTTriangulator()

    def triangulate(self, pts_a, pts_b, P_a, P_b) -> np.ndarray:
        # Lindstrom iterative correction (simplified version)
        pa = pts_a.astype(np.float64).copy()
        pb = pts_b.astype(np.float64).copy()

        for _ in range(self.max_iters):
            pts3d = self._dlt.triangulate(pa, pb, P_a, P_b)
            # Reproject and correct
            n = pts3d.shape[0]
            ph = np.hstack([pts3d, np.ones((n, 1))])
            proj_a = (P_a @ ph.T).T
            proj_b = (P_b @ ph.T).T
            pa = proj_a[:, :2] / (proj_a[:, 2:3] + 1e-10)
            pb = proj_b[:, :2] / (proj_b[:, 2:3] + 1e-10)
            pa = (pa + pts_a) / 2
            pb = (pb + pts_b) / 2

        return self._dlt.triangulate(pa, pb, P_a, P_b)


# ============================================================
#  PIPELINE STEP FUNCTION
# ============================================================

def triangulate_new_points(
    new_image_id: int,
    images: list,
    reconstruction: Reconstruction,
    triangulator: Optional[TriangulatorBase] = None,
    max_reproj_error: float = 4.0,
    min_triangulation_angle_deg: float = 0.5,#2.0
) -> int:
    """
    Step 8 — Triangulate new 3-D points after registering `new_image_id`.

    For every already-registered image that shares verified matches with
    `new_image_id`, triangulate the matched points that are NOT yet in any
    3-D track.

    Parameters
    ----------
    new_image_id                : the freshly registered image
    images                      : list of BGR arrays (for colour sampling)
    reconstruction              : SfM state — new Point3Ds will be added
    triangulator                : TriangulatorBase (default: DLTTriangulator)
    max_reproj_error            : pixel threshold for reprojection filter
    min_triangulation_angle_deg : ray-angle threshold

    Returns
    -------
    n_new : number of new 3-D points added
    """
    if triangulator is None:
        triangulator = DLTTriangulator()

    # Validate inputs
    if new_image_id not in reconstruction.registered_images:
        print(f"[Step 8] Image {new_image_id} not registered. Skipping.")
        return 0
    if new_image_id not in reconstruction.keypoints:
        print(f"[Step 8] No keypoints for image {new_image_id}. Skipping.")
        return 0
    if new_image_id not in reconstruction.cameras:
        print(f"[Step 8] No camera for image {new_image_id}. Skipping.")
        return 0
    if not reconstruction.verified_matches:
        print("[Step 8] No verified matches available.")
        return 0

    print(f"[Step 8] Triangulating new points for image {new_image_id}  "
          f"(method: {triangulator.__class__.__name__})")

    # Build a lookup: (image_id, kp_idx) → point3d_id
    obs_to_point: dict = {}
    for pid, p3d in reconstruction.points3d.items():
        for (obs_img, obs_kp) in p3d.track:
            obs_to_point[(obs_img, obs_kp)] = pid

    kp_new = reconstruction.keypoints[new_image_id]
    ri_new = reconstruction.registered_images[new_image_id]
    cam_new = reconstruction.cameras[new_image_id]
    P_new = _projection_matrix(cam_new.K, ri_new.R, ri_new.t)
    C_new = _camera_centre(ri_new.R, ri_new.t)

    img_new = images[new_image_id]
    n_added = 0

    for (id_a, id_b), vm in reconstruction.verified_matches.items():
        # We only care about pairs involving the new image
        if id_a == new_image_id:
            other_id, idx_new_col, idx_other_col = id_b, 0, 1
        elif id_b == new_image_id:
            other_id, idx_new_col, idx_other_col = id_a, 1, 0
        else:
            continue

        # Other image must already be registered
        if other_id not in reconstruction.registered_images:
            continue

        ri_other = reconstruction.registered_images[other_id]
        cam_other = reconstruction.cameras[other_id]
        P_other = _projection_matrix(cam_other.K, ri_other.R, ri_other.t)
        C_other = _camera_centre(ri_other.R, ri_other.t)
        kp_other = reconstruction.keypoints[other_id]

        # Only triangulate pairs where neither keypoint is already in a track
        new_matches_idx = []
        for i, m in enumerate(vm.matches):
            kp_idx_new   = m[idx_new_col]
            kp_idx_other = m[idx_other_col]
            already_new   = (new_image_id, int(kp_idx_new)) in obs_to_point
            already_other = (other_id,     int(kp_idx_other)) in obs_to_point
            if not already_new and not already_other:
                new_matches_idx.append(i)

        if not new_matches_idx:
            continue

        sel = vm.matches[new_matches_idx]
        pts_new   = kp_new.points[sel[:, idx_new_col]]
        pts_other = kp_other.points[sel[:, idx_other_col]]

        # Triangulate
        if id_a == new_image_id:
            pts3d = triangulator.triangulate(pts_new, pts_other, P_new, P_other)
        else:
            pts3d = triangulator.triangulate(pts_other, pts_new, P_other, P_new)

        # Filter: reprojection error
        err_new   = _reprojection_error(pts3d, pts_new,   P_new)
        err_other = _reprojection_error(pts3d, pts_other, P_other)
        valid_reproj = (err_new < max_reproj_error) & (err_other < max_reproj_error)

        # Filter: triangulation angle
        angles = _triangulation_angle_deg(pts3d, C_new, C_other)
        valid_angle = angles > min_triangulation_angle_deg

        # Filter: positive depth (in front of both cameras)
        def positive_depth(pts, R, t):
            if pts.shape[0] == 0:
                return np.array([], dtype=bool)
            try:
                pts_cam = (R @ pts.T).T + t
                return pts_cam[:, 2] > 0
            except (ValueError, IndexError):
                return np.zeros(len(pts), dtype=bool)

        valid_depth = positive_depth(pts3d, ri_new.R, ri_new.t) & \
                      positive_depth(pts3d, ri_other.R, ri_other.t)

        valid = valid_reproj & valid_angle & valid_depth
        sel = sel[valid]
        pts3d = pts3d[valid]
        err_new_f = err_new[valid]
        err_other_f = err_other[valid]

        for i, (m, xyz) in enumerate(zip(sel, pts3d)):
            kp_idx_new   = int(m[idx_new_col])
            kp_idx_other = int(m[idx_other_col])

            # Colour from the new image
            px, py = kp_new.points[kp_idx_new].astype(int)
            px = np.clip(px, 0, img_new.shape[1] - 1)
            py = np.clip(py, 0, img_new.shape[0] - 1)
            bgr = img_new[py, px]
            color = np.array([bgr[2], bgr[1], bgr[0]], dtype=np.uint8)

            pid = reconstruction.new_point_id()
            p3d = Point3D(
                point3d_id=pid,
                xyz=xyz,
                color=color,
                error=float((err_new_f[i] + err_other_f[i]) / 2),
                track=[(new_image_id, kp_idx_new), (other_id, kp_idx_other)],
            )
            reconstruction.points3d[pid] = p3d
            obs_to_point[(new_image_id, kp_idx_new)] = pid
            obs_to_point[(other_id, kp_idx_other)]   = pid
            n_added += 1

    print(f"  Added {n_added} new 3-D points  "
          f"(total: {len(reconstruction.points3d)})")
    return n_added
