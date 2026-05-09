"""
step10_outlier_filtering.py
----------------------------
Step 10 — Outlier Filtering & Point Cloud Cleaning

Responsibilities:
  - Remove 3-D points with high reprojection error
  - Remove points seen by too few cameras (short tracks)
  - Remove points with poor triangulation geometry (small ray angles)
  - Optionally remove statistical outliers (distance from cloud centroid)
  - Update Reconstruction.points3d in-place

Swappable strategies
  - ReprojectionFilter        — threshold on mean reprojection error
  - TrackLengthFilter         — minimum number of observations
  - TriangulationAngleFilter  — minimum ray angle across all track pairs
  - StatisticalOutlierFilter  — distance-based outlier removal (like PCL SOR)
  - ChainFilter               — apply multiple filters in sequence
"""

import numpy as np
import cv2
from abc import ABC, abstractmethod
from typing import List, Optional, Set

from sfm_types import Point3D, Reconstruction


# ============================================================
#  HELPERS
# ============================================================

def _rot_to_rvec(R):
    rvec, _ = cv2.Rodrigues(R)
    return rvec.ravel()


def _reproj_error_for_point(p3d: Point3D, reconstruction: Reconstruction) -> float:
    """Compute mean reprojection error for a single Point3D."""
    errors = []
    for (img_id, kp_idx) in p3d.track:
        ri  = reconstruction.registered_images.get(img_id)
        cam = reconstruction.cameras.get(img_id)
        kp  = reconstruction.keypoints.get(img_id)
        if ri is None or cam is None or kp is None:
            continue
        if kp_idx >= len(kp.points):
            continue
        proj, _ = cv2.projectPoints(
            p3d.xyz.reshape(1, 1, 3),
            _rot_to_rvec(ri.R).reshape(3, 1),
            ri.t.reshape(3, 1),
            cam.K, cam.dist_coeffs,
        )
        err = np.linalg.norm(proj.reshape(2) - kp.points[kp_idx])
        errors.append(err)
    return float(np.mean(errors)) if errors else 0.0


def _min_triangulation_angle(p3d: Point3D, reconstruction: Reconstruction) -> float:
    """
    Return the MINIMUM pairwise ray-angle (degrees) among all camera pairs
    that observe this point. (We want this to be large enough for good triangulation.)
    Points triangulated from cameras with small baseline have small min angle.
    """
    cam_centres = []
    for (img_id, _) in p3d.track:
        ri = reconstruction.registered_images.get(img_id)
        if ri is None:
            continue
        C = -ri.R.T @ ri.t
        cam_centres.append(C)

    if len(cam_centres) < 2:
        return 0.0

    min_angle = 180.0
    xyz = p3d.xyz
    for i in range(len(cam_centres)):
        for j in range(i + 1, len(cam_centres)):
            ra = xyz - cam_centres[i]
            rb = xyz - cam_centres[j]
            na = np.linalg.norm(ra) + 1e-10
            nb = np.linalg.norm(rb) + 1e-10
            cos_a = np.clip(np.dot(ra / na, rb / nb), -1, 1)
            angle = float(np.degrees(np.arccos(cos_a)))
            min_angle = min(min_angle, angle)

    return min_angle


# ============================================================
#  FILTER INTERFACE
# ============================================================

class PointFilterBase(ABC):
    @abstractmethod
    def filter(self, reconstruction: Reconstruction) -> Set[int]:
        """Return the set of point3d_ids that should be REMOVED."""


# ============================================================
#  CONCRETE FILTERS
# ============================================================

class ReprojectionFilter(PointFilterBase):
    """Remove points whose mean reprojection error exceeds `max_error` pixels."""

    def __init__(self, max_error: float = 4.0):
        self.max_error = max_error

    def filter(self, reconstruction: Reconstruction) -> Set[int]:
        remove = set()
        for pid, p3d in reconstruction.points3d.items():
            err = _reproj_error_for_point(p3d, reconstruction)
            if err > self.max_error:
                remove.add(pid)
        return remove


class TrackLengthFilter(PointFilterBase):
    """Remove points observed in fewer than `min_track_length` images."""

    def __init__(self, min_track_length: int = 2):
        self.min_len = min_track_length

    def filter(self, reconstruction: Reconstruction) -> Set[int]:
        remove = set()
        for pid, p3d in reconstruction.points3d.items():
            # Count only observations from registered cameras
            valid_obs = sum(
                1 for (img_id, _) in p3d.track
                if img_id in reconstruction.registered_images
            )
            if valid_obs < self.min_len:
                remove.add(pid)
        return remove


class TriangulationAngleFilter(PointFilterBase):
    """Remove points with maximum pairwise ray angle below `min_angle_deg`."""

    def __init__(self, min_angle_deg: float = 2.0):
        self.min_angle = min_angle_deg

    def filter(self, reconstruction: Reconstruction) -> Set[int]:
        remove = set()
        for pid, p3d in reconstruction.points3d.items():
            angle = _min_triangulation_angle(p3d, reconstruction)
            if angle < self.min_angle:
                remove.add(pid)
        return remove


class StatisticalOutlierFilter(PointFilterBase):
    """
    Remove points that lie more than `std_ratio` standard deviations
    from the mean distance to their k nearest neighbours.
    Similar to PCL StatisticalOutlierRemoval.
    """

    def __init__(self, k: int = 20, std_ratio: float = 2.0):
        self.k = k
        self.std_ratio = std_ratio

    def filter(self, reconstruction: Reconstruction) -> Set[int]:
        if len(reconstruction.points3d) < self.k + 1:
            return set()

        point_ids = list(reconstruction.points3d.keys())
        pts = np.array([reconstruction.points3d[pid].xyz for pid in point_ids])

        # Pairwise distances (naïve; use KDTree for large clouds)
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(pts)
            dists, _ = tree.query(pts, k=self.k + 1)
            mean_dists = dists[:, 1:].mean(axis=1)   # exclude self (dist=0)
        except ImportError:
            # Fallback: compute full pairwise (slow for large clouds)
            mean_dists = np.zeros(len(pts))
            for i, p in enumerate(pts):
                d = np.linalg.norm(pts - p, axis=1)
                d_sorted = np.sort(d)[1: self.k + 1]
                mean_dists[i] = d_sorted.mean()

        mu = mean_dists.mean()
        sigma = mean_dists.std()
        threshold = mu + self.std_ratio * sigma

        remove = set()
        for i, pid in enumerate(point_ids):
            if mean_dists[i] > threshold:
                remove.add(pid)
        return remove


class ChainFilter(PointFilterBase):
    """Apply multiple filters in sequence; union of all removed sets."""

    def __init__(self, filters: List[PointFilterBase]):
        self.filters = filters

    def filter(self, reconstruction: Reconstruction) -> Set[int]:
        remove: Set[int] = set()
        for f in self.filters:
            remove |= f.filter(reconstruction)
        return remove


# ============================================================
#  PIPELINE STEP FUNCTION
# ============================================================

def filter_outliers(
    reconstruction: Reconstruction,
    point_filter: Optional[PointFilterBase] = None,
) -> int:
    """
    Step 10 — Remove outlier 3-D points from the reconstruction.

    Parameters
    ----------
    reconstruction : SfM state (modified in-place)
    point_filter   : PointFilterBase (default: ChainFilter with reproj + track + angle)

    Returns
    -------
    n_removed : number of points removed
    """
    # Validate state
    if not reconstruction.points3d:
        print("[Step 10] No points to filter.")
        return 0
    
    if point_filter is None:
        point_filter = ChainFilter([
            ReprojectionFilter(max_error=4.0),
            TrackLengthFilter(min_track_length=2),
            TriangulationAngleFilter(min_angle_deg=2.0),
        ])

    before = len(reconstruction.points3d)
    to_remove = point_filter.filter(reconstruction)

    for pid in to_remove:
        del reconstruction.points3d[pid]

    after = len(reconstruction.points3d)
    print(f"[Step 10] Outlier filtering ({point_filter.__class__.__name__}): "
          f"removed {before - after} points  ({before} -> {after})")
    return before - after
