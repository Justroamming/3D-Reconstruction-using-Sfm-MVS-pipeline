"""
step6_initial_reconstruction.py
---------------------------------
Step 6 — Seed Pair Selection & Initial Two-View Reconstruction

Responsibilities:
  - Choose the best initial image pair (wide baseline, many verified matches)
  - Recover relative pose R, t from Essential/Fundamental matrix
  - Triangulate initial 3-D points
  - Register both seed images into Reconstruction

Swappable strategies
  SeedPairSelectors:
    - MaxInliersSeedSelector     — pair with most verified inliers
    - HomographyRatioSeedSelector— prefer pairs NOT well explained by homography
                                   (avoids degenerate planar scenes)

  InitialPoseRecoverers:
    - EssentialMatrixRecoverer   — decompose E (known K)
    - FundamentalRecoverer       — self-calibration via F (unknown K, less stable)
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from sfm_types import Camera, Keypoints, Point3D, Reconstruction, RegisteredImage, VerifiedMatches, normalize_pair_key


# ============================================================
#  SEED PAIR SELECTION
# ============================================================

class SeedPairSelectorBase(ABC):
    @abstractmethod
    def select(self, reconstruction: Reconstruction) -> Tuple[int, int]:
        """Return (image_id_a, image_id_b) of the best initial pair."""


class MaxInliersSeedSelector(SeedPairSelectorBase):
    """Choose the pair with the most verified inlier matches."""

    def select(self, reconstruction: Reconstruction) -> Tuple[int, int]:
        best_pair = max(
            reconstruction.verified_matches.keys(),
            key=lambda k: reconstruction.verified_matches[k].num_inliers,
        )
        return best_pair


class HomographyRatioSeedSelector(SeedPairSelectorBase):
    """
    Prefer pairs with a LOW homography-inlier ratio (non-planar scene).
    Among those, pick the one with most verified matches.

    Pairs where >85 % of matches fit a homography are skipped
    because they give unstable Essential/Fundamental decompositions.
    """

    def __init__(self, homography_ratio_thresh: float = 0.85, ransac_thresh: float = 4.0):
        self.ratio_thresh = homography_ratio_thresh
        self.ransac_thresh = ransac_thresh

    def _homography_ratio(self, kp_a: Keypoints, kp_b: Keypoints,
                          matches: np.ndarray) -> float:
        pts_a = kp_a.points[matches[:, 0]]
        pts_b = kp_b.points[matches[:, 1]]
        if len(pts_a) < 4:
            return 1.0
        _, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, self.ransac_thresh)
        if mask is None:
            return 0.0
        return float(mask.sum()) / len(mask)

    def select(self, reconstruction: Reconstruction) -> Tuple[int, int]:
        candidates = []
        for pair, vm in reconstruction.verified_matches.items():
            kp_a = reconstruction.keypoints[vm.image_id_a]
            kp_b = reconstruction.keypoints[vm.image_id_b]
            h_ratio = self._homography_ratio(kp_a, kp_b, vm.matches)
            if h_ratio < self.ratio_thresh:
                candidates.append((pair, vm.num_inliers, h_ratio))

        if not candidates:
            print("[SeedSelector] Warning: all pairs are planar - falling back to max inliers.")
            return MaxInliersSeedSelector().select(reconstruction)

        # Among non-planar pairs, pick the one with the most inliers
        best = max(candidates, key=lambda x: x[1])
        print(f"  Seed pair {best[0]}  inliers={best[1]}  H-ratio={best[2]:.2f}")
        return best[0]


# ============================================================
#  INITIAL POSE RECOVERY
# ============================================================

class InitialPoseRecovererBase(ABC):
    @abstractmethod
    def recover_pose(
        self,
        kp_a: Keypoints,
        kp_b: Keypoints,
        vm: VerifiedMatches,
        cam_a: Camera,
        cam_b: Camera,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        """
        Returns
        -------
        R       : (3,3) rotation matrix (None on failure)
        t       : (3,) unit translation vector (None on failure)
        mask    : bool array — which matches pass the cheirality check
        """


class EssentialMatrixRecoverer(InitialPoseRecovererBase):
    """
    Recover R, t by:
      1. Computing Essential matrix from verified matches + K
      2. Decomposing E with cheirality check (cv2.recoverPose)
    """

    def __init__(self, ransac_thresh: float = 1.0, confidence: float = 0.999):
        self.thresh = ransac_thresh
        self.confidence = confidence

    def recover_pose(self, kp_a, kp_b, vm, cam_a, cam_b):
        pts_a = kp_a.points[vm.matches[:, 0]]
        pts_b = kp_b.points[vm.matches[:, 1]]

        E, mask_e = cv2.findEssentialMat(
            pts_a, pts_b,
            cameraMatrix=cam_a.K,
            method=cv2.RANSAC,
            prob=self.confidence,
            threshold=self.thresh,
        )
        if E is None:
            return None, None, np.zeros(len(pts_a), bool)

        mask_e = mask_e.ravel().astype(bool)
        n_inliers, R, t, mask_pose = cv2.recoverPose(
            E,
            pts_a[mask_e], pts_b[mask_e],
            cameraMatrix=cam_a.K,
        )
        # Propagate cheirality mask back to full match set
        full_mask = np.zeros(len(pts_a), dtype=bool)
        full_mask[np.where(mask_e)[0][mask_pose.ravel().astype(bool)]] = True

        return R, t.ravel(), full_mask


class FundamentalRecoverer(InitialPoseRecovererBase):
    """
    Self-calibration path when K is uncertain.
    Decomposes F into an Essential matrix using the supplied K estimate,
    then proceeds as EssentialMatrixRecoverer.
    NOTE: inherently less stable; prefer EssentialMatrixRecoverer when K is known.
    """

    def __init__(self, ransac_thresh: float = 3.0, confidence: float = 0.999):
        self.ransac_thresh = ransac_thresh
        self.confidence = confidence

    def recover_pose(self, kp_a, kp_b, vm, cam_a, cam_b):
        # Use EssentialMatrixRecoverer with passed parameters
        recoverer = EssentialMatrixRecoverer(
            ransac_thresh=self.ransac_thresh,
            confidence=self.confidence
        )
        return recoverer.recover_pose(kp_a, kp_b, vm, cam_a, cam_b)


# ============================================================
#  TRIANGULATION HELPER (used here and in Step 8)
# ============================================================

def triangulate_points(
    pts_a: np.ndarray,      # (N,2) observations in image A
    pts_b: np.ndarray,      # (N,2) observations in image B
    P_a: np.ndarray,        # (3,4) projection matrix for A = K[R|t]
    P_b: np.ndarray,        # (3,4) projection matrix for B = K[R|t]
) -> np.ndarray:
    """
    DLT triangulation via cv2.triangulatePoints.

    Returns
    -------
    pts3d : (N,3) float64 array of world-space 3-D points
    """
    if pts_a.size == 0 or pts_b.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    P_a = np.ascontiguousarray(P_a, dtype=np.float64)
    P_b = np.ascontiguousarray(P_b, dtype=np.float64)
    pts_a = np.ascontiguousarray(pts_a, dtype=np.float64).T
    pts_b = np.ascontiguousarray(pts_b, dtype=np.float64).T

    pts4d = cv2.triangulatePoints(P_a, P_b, pts_a, pts_b)   # (4,N)
    w = pts4d[3]
    pts3d = (pts4d[:3] / w).T                                     # (N,3)
    return pts3d


def _projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build 3×4 projection matrix P = K [R | t]."""
    Rt = np.hstack([R, t.reshape(3, 1)])
    return K @ Rt


def _reprojection_errors(
    pts3d: np.ndarray,
    pts2d: np.ndarray,
    P: np.ndarray,
) -> np.ndarray:
    """Return per-point reprojection error in pixels."""
    n = pts3d.shape[0]
    pts3d_h = np.hstack([pts3d, np.ones((n, 1))])          # (N,4)
    proj = (P @ pts3d_h.T).T                                # (N,3)
    proj_2d = proj[:, :2] / proj[:, 2:3]                   # (N,2)
    return np.linalg.norm(proj_2d - pts2d, axis=1)          # (N,)


# ============================================================
#  PIPELINE STEP FUNCTION
# ============================================================

def initialize_reconstruction(
    images: list,
    reconstruction: Reconstruction,
    seed_selector: Optional[SeedPairSelectorBase] = None,
    pose_recoverer: Optional[InitialPoseRecovererBase] = None,
    max_reproj_error: float = 4.0,
    min_triangulation_angle_deg: float = 0.5,
) -> bool:
    """
    Step 6 — Select seed pair and build initial two-view reconstruction.

    Parameters
    ----------
    images                      : list of BGR arrays (from Step 1)
    reconstruction              : SfM state — will gain 2 RegisteredImages + initial Point3Ds
    seed_selector               : SeedPairSelectorBase (default: HomographyRatioSeedSelector)
    pose_recoverer              : InitialPoseRecovererBase (default: EssentialMatrixRecoverer)
    max_reproj_error            : discard triangulated points above this pixel error
    min_triangulation_angle_deg : discard points with small ray angle (poorly conditioned)

    Returns
    -------
    success : bool — False if no valid seed pair could be found
    """
    if seed_selector is None:
        seed_selector = HomographyRatioSeedSelector()
    if pose_recoverer is None:
        pose_recoverer = EssentialMatrixRecoverer()

    # Validate inputs
    if not reconstruction.verified_matches:
        print("[Step 6] No verified matches available. Cannot initialize.")
        return False

    print(f"[Step 6] Initial reconstruction - "
          f"seed: {seed_selector.__class__.__name__}, "
          f"pose: {pose_recoverer.__class__.__name__}")

    # --- 6a. Seed pair selection ---
    try:
        selected_pair = normalize_pair_key(*seed_selector.select(reconstruction))
    except (ValueError, KeyError) as e:
        print(f"[Step 6] Seed selection failed: {e}")
        return False

    candidate_pairs = list(reconstruction.verified_matches.keys())
    candidate_pairs = [selected_pair] + [pair for pair in candidate_pairs if pair != selected_pair]
    candidate_pairs.sort(key=lambda pair: reconstruction.verified_matches[pair].num_inliers, reverse=True)
    if selected_pair in candidate_pairs:
        candidate_pairs.remove(selected_pair)
        candidate_pairs.insert(0, selected_pair)

    for pair_key in candidate_pairs:
        vm = reconstruction.verified_matches[pair_key]
        id_a, id_b = pair_key

        # Validate that required data exists
        if id_a not in reconstruction.cameras or id_b not in reconstruction.cameras:
            continue
        if id_a not in reconstruction.keypoints or id_b not in reconstruction.keypoints:
            continue

        cam_a = reconstruction.cameras[id_a]
        cam_b = reconstruction.cameras[id_b]
        kp_a = reconstruction.keypoints[id_a]
        kp_b = reconstruction.keypoints[id_b]

        print(f"  Seed pair: img {id_a} <-> img {id_b}  ({vm.num_inliers} verified matches)")

        # --- 6b. Recover initial pose ---
        R, t, cheirality_mask = pose_recoverer.recover_pose(kp_a, kp_b, vm, cam_a, cam_b)

        if R is None:
            print("  ERROR: pose recovery failed.")
            continue

        # Image A is the origin: R=I, t=0
        R_a = np.eye(3)
        t_a = np.zeros(3)
        R_b = R
        t_b = t

        P_a = _projection_matrix(cam_a.K, R_a, t_a)
        P_b = _projection_matrix(cam_b.K, R_b, t_b)

        # --- 6c. Triangulate initial points ---
        valid_matches = vm.matches[cheirality_mask]
        if len(valid_matches) == 0:
            print("  WARNING: no cheirality-consistent matches for this pair; trying another pair.")
            continue

        pts_a_2d = kp_a.points[valid_matches[:, 0]]
        pts_b_2d = kp_b.points[valid_matches[:, 1]]

        pts3d = triangulate_points(pts_a_2d, pts_b_2d, P_a, P_b)
        if len(pts3d) == 0:
            print("  WARNING: no seed correspondences available for triangulation; trying another pair.")
            continue

        # Filter by reprojection error
        err_a = _reprojection_errors(pts3d, pts_a_2d, P_a)
        err_b = _reprojection_errors(pts3d, pts_b_2d, P_b)
        valid_reproj = (err_a < max_reproj_error) & (err_b < max_reproj_error)

        # Filter by triangulation angle
        # Direction from camera centre to point
        C_a = -R_a.T @ t_a
        C_b = -R_b.T @ t_b
        ray_a = pts3d - C_a
        ray_b = pts3d - C_b
        norms_a = np.linalg.norm(ray_a, axis=1, keepdims=True) + 1e-10
        norms_b = np.linalg.norm(ray_b, axis=1, keepdims=True) + 1e-10
        cos_angle = np.sum((ray_a / norms_a) * (ray_b / norms_b), axis=1)
        angles_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        valid_angle = angles_deg > min_triangulation_angle_deg

        valid = valid_reproj & valid_angle
        pts3d = pts3d[valid]
        valid_matches = valid_matches[valid]

        if len(pts3d) == 0:
            print("  WARNING: seed pair produced no valid 3-D points after filtering; trying another pair.")
            continue

        # --- 6d. Register seed images and store points ---
        reconstruction.registered_images[id_a] = RegisteredImage(id_a, id_a, R_a, t_a)
        reconstruction.registered_images[id_b] = RegisteredImage(id_b, id_b, R_b, t_b)

        # Colour from image A
        img_a = images[id_a]

        for i, (match, xyz) in enumerate(zip(valid_matches, pts3d)):
            kp_idx_a = match[0]
            kp_idx_b = match[1]
            px, py = kp_a.points[kp_idx_a].astype(int)
            px = np.clip(px, 0, img_a.shape[1] - 1)
            py = np.clip(py, 0, img_a.shape[0] - 1)
            bgr = img_a[py, px]
            color = np.array([bgr[2], bgr[1], bgr[0]], dtype=np.uint8)  # BGR→RGB

            pid = reconstruction.new_point_id()
            reconstruction.points3d[pid] = Point3D(
                point3d_id=pid,
                xyz=xyz,
                color=color,
                error=float((err_a[valid][i] + err_b[valid][i]) / 2),
                track=[(id_a, int(kp_idx_a)), (id_b, int(kp_idx_b))],
            )

        print(f"  Registered images: {id_a}, {id_b}")
        print(f"  Triangulated {len(reconstruction.points3d)} initial 3-D points "
              f"(from {cheirality_mask.sum()} candidates)")

        return True

    print("[Step 6] No seed pair produced valid triangulated points.")
    return False
