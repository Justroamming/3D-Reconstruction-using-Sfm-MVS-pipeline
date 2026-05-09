"""
step7_camera_registration.py
-----------------------------
Step 7 — Incremental Camera Registration (PnP)

Responsibilities:
  - Determine which un-registered image to add next (next-best-view)
  - Find 2D–3D correspondences between the new image and existing 3-D points
  - Estimate the new camera pose with PnP + RANSAC
  - Register the camera into Reconstruction

Swappable strategies
  NextViewSelectors:
    - MaxCorrespondencesSelector  — image with most 2D↔3D correspondences
    - ScoreBasedSelector          — weighted score: #correspondences × coverage

  PnPSolvers:
    - EPnPRANSAC     — EPnP algorithm in RANSAC loop (fast, general)
    - IterativePnP   — Levenberg-Marquardt iterative (accurate for small sets)
    - P3PRANSAC      — minimal 3-point solver in RANSAC (good for many outliers)
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple

from sfm_types import Camera, Keypoints, Point3D, Reconstruction, RegisteredImage, normalize_pair_key


# ============================================================
#  BUILD 2-D / 3-D CORRESPONDENCE MAP
# ============================================================

def build_2d3d_correspondences(
    image_id: int,
    reconstruction: Reconstruction,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
        Build 2D-3D correspondences for an unregistered image.

        A keypoint in `image_id` is linked to a 3-D point when:
            1) the keypoint is in a verified match with a registered image keypoint, and
            2) that registered-image keypoint already belongs to a Point3D track.

    Returns
    -------
    pts2d     : (N,2) float32 — 2-D keypoint positions in the new image
    pts3d     : (N,3) float64 — corresponding 3-D point coordinates
    point_ids : list of point3d_id for each correspondence
    """
    if image_id not in reconstruction.keypoints:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 3), dtype=np.float64),
            [],
        )

    pts2d_list, pts3d_list, pid_list = [], [], []
    kp = reconstruction.keypoints[image_id]

    # Existing observation -> point id map.
    obs_to_pid: Dict[Tuple[int, int], int] = {}
    for pid, p3d in reconstruction.points3d.items():
        for (obs_img_id, kp_idx) in p3d.track:
            obs_to_pid[(int(obs_img_id), int(kp_idx))] = pid

    # Candidate image keypoint -> point id.
    # If conflicting mappings appear, mark keypoint ambiguous and discard it.
    cand_kp_to_pid: Dict[int, int] = {}
    ambiguous_kp: Set[int] = set()

    for reg_id in reconstruction.registered_images.keys():
        if reg_id == image_id:
            continue

        pair_key = normalize_pair_key(image_id, reg_id)
        vm = reconstruction.verified_matches.get(pair_key)
        if vm is None or len(vm.matches) == 0:
            continue

        if pair_key[0] == image_id:
            idx_cand, idx_reg = 0, 1
        else:
            idx_cand, idx_reg = 1, 0

        for m in vm.matches:
            kp_idx_cand = int(m[idx_cand])
            kp_idx_reg = int(m[idx_reg])
            pid = obs_to_pid.get((reg_id, kp_idx_reg))
            if pid is None:
                continue

            prev = cand_kp_to_pid.get(kp_idx_cand)
            if prev is None:
                cand_kp_to_pid[kp_idx_cand] = pid
            elif prev != pid:
                ambiguous_kp.add(kp_idx_cand)

    used_pid: Set[int] = set()
    for kp_idx_cand, pid in sorted(cand_kp_to_pid.items()):
        if kp_idx_cand in ambiguous_kp:
            continue
        if kp_idx_cand >= len(kp.points):
            continue
        if pid in used_pid:
            continue

        p3d = reconstruction.points3d.get(pid)
        if p3d is None:
            continue

        pts2d_list.append(kp.points[kp_idx_cand])
        pts3d_list.append(p3d.xyz)
        pid_list.append(pid)
        used_pid.add(pid)

    if not pts2d_list:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 3), dtype=np.float64),
            [],
        )

    return (
        np.array(pts2d_list, dtype=np.float32),
        np.array(pts3d_list, dtype=np.float64),
        pid_list,
    )


# ============================================================
#  NEXT-BEST-VIEW SELECTORS
# ============================================================

class NextViewSelectorBase(ABC):
    @abstractmethod
    def select(
        self,
        reconstruction: Reconstruction,
        registered_ids: Set[int],
    ) -> Optional[int]:
        """
        Return the image_id of the next image to register,
        or None if no suitable candidate exists.
        """


class MaxCorrespondencesSelector(NextViewSelectorBase):
    """
    Pick the unregistered image that shares the most 2D↔3D correspondences
    with the current reconstruction.
    """

    def __init__(self, min_correspondences: int = 12):
        self.min_corr = min_correspondences

    def select(self, reconstruction, registered_ids):
        unregistered = [
            i for i in range(len(reconstruction.image_paths))
            if i not in registered_ids
        ]

        best_id, best_n = None, self.min_corr - 1
        for img_id in unregistered:
            pts2d, pts3d, _ = build_2d3d_correspondences(img_id, reconstruction)
            if len(pts2d) > best_n:
                best_n = len(pts2d)
                best_id = img_id

        return best_id


class ScoreBasedSelector(NextViewSelectorBase):
    """
    Score = #correspondences × spatial_coverage_factor.
    Spatial coverage is the fraction of the image area covered by visible 3-D points,
    rewarding images where the points are spread out (better conditioning).
    """

    def __init__(self, min_correspondences: int = 12):
        self.min_corr = min_correspondences

    def _coverage(self, pts2d: np.ndarray, w: int, h: int) -> float:
        if len(pts2d) == 0:
            return 0.0
        # Divide into 4×4 grid, count occupied cells
        gx = np.clip((pts2d[:, 0] / w * 4).astype(int), 0, 3)
        gy = np.clip((pts2d[:, 1] / h * 4).astype(int), 0, 3)
        occupied = len(set(zip(gx, gy)))
        return occupied / 16.0

    def select(self, reconstruction, registered_ids):
        unregistered = [
            i for i in range(len(reconstruction.image_paths))
            if i not in registered_ids
        ]

        best_id, best_score = None, -1.0
        for img_id in unregistered:
            pts2d, pts3d, _ = build_2d3d_correspondences(img_id, reconstruction)
            if len(pts2d) < self.min_corr:
                continue
            cam = reconstruction.cameras.get(img_id)
            w = cam.width if cam else 1
            h = cam.height if cam else 1
            score = len(pts2d) * self._coverage(pts2d, w, h)
            if score > best_score:
                best_score = score
                best_id = img_id

        return best_id


# ============================================================
#  PNP SOLVERS
# ============================================================

class PnPSolverBase(ABC):
    @abstractmethod
    def solve(
        self,
        pts3d: np.ndarray,      # (N,3)
        pts2d: np.ndarray,      # (N,2)
        K: np.ndarray,          # (3,3)
        dist: np.ndarray,       # distortion coefficients
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        """
        Returns
        -------
        R        : (3,3) or None
        t        : (3,)  or None
        inlier_mask : (N,) bool
        """


class EPnPRANSAC(PnPSolverBase):
    """EPnP in a RANSAC loop. Good general-purpose choice."""

    def __init__(self, reproj_thresh: float = 10.0, confidence: float = 0.999,
                 max_iters: int = 1000):
        self.thresh = reproj_thresh
        self.confidence = confidence
        self.max_iters = max_iters

    def solve(self, pts3d, pts2d, K, dist):
        if len(pts3d) < 4:
            return None, None, np.zeros(len(pts3d), bool)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d.astype(np.float64),
            pts2d.astype(np.float64),
            K, dist,
            flags=cv2.SOLVEPNP_EPNP,
            reprojectionError=self.thresh,
            confidence=self.confidence,
            iterationsCount=self.max_iters,
        )
        if not success or inliers is None:
            return None, None, np.zeros(len(pts3d), bool)

        R, _ = cv2.Rodrigues(rvec)
        mask = np.zeros(len(pts3d), dtype=bool)
        mask[inliers.ravel()] = True
        return R, tvec.ravel(), mask


class IterativePnP(PnPSolverBase):
    """
    Iterative Levenberg-Marquardt PnP.
    No built-in RANSAC — pre-filter or use with many inliers.
    """

    def solve(self, pts3d, pts2d, K, dist):
        if len(pts3d) < 4:
            return None, None, np.zeros(len(pts3d), bool)

        success, rvec, tvec = cv2.solvePnP(
            pts3d.astype(np.float64),
            pts2d.astype(np.float64),
            K, dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None, None, np.zeros(len(pts3d), bool)

        R, _ = cv2.Rodrigues(rvec)
        return R, tvec.ravel(), np.ones(len(pts3d), dtype=bool)


class P3PRANSAC(PnPSolverBase):
    """Minimal 3-point solver in RANSAC. Robust to many outliers."""

    def __init__(self, reproj_thresh: float = 8.0, confidence: float = 0.999,
                 max_iters: int = 1000):
        self.thresh = reproj_thresh
        self.confidence = confidence
        self.max_iters = max_iters

    def solve(self, pts3d, pts2d, K, dist):
        if len(pts3d) < 4:
            return None, None, np.zeros(len(pts3d), bool)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d.astype(np.float64),
            pts2d.astype(np.float64),
            K, dist,
            flags=cv2.SOLVEPNP_P3P,
            reprojectionError=self.thresh,
            confidence=self.confidence,
            iterationsCount=self.max_iters,
        )
        if not success or inliers is None:
            return None, None, np.zeros(len(pts3d), bool)

        R, _ = cv2.Rodrigues(rvec)
        mask = np.zeros(len(pts3d), dtype=bool)
        mask[inliers.ravel()] = True
        return R, tvec.ravel(), mask


# ============================================================
#  PIPELINE STEP FUNCTION
# ============================================================

def register_next_image(
    reconstruction: Reconstruction,
    view_selector: Optional[NextViewSelectorBase] = None,
    pnp_solver: Optional[PnPSolverBase] = None,
    min_inliers: int = 30,#6
) -> Optional[int]:
    """
    Step 7 — Register the next best unregistered image.

    Parameters
    ----------
    reconstruction : SfM state — a new RegisteredImage will be added
    view_selector  : NextViewSelectorBase (default: MaxCorrespondencesSelector)
    pnp_solver     : PnPSolverBase (default: EPnPRANSAC)
    min_inliers    : minimum PnP inliers to accept registration

    Returns
    -------
    image_id of the newly registered image, or None if none can be added.
    """
    if view_selector is None:
        view_selector = MaxCorrespondencesSelector()
    if pnp_solver is None:
        pnp_solver = EPnPRANSAC()

    registered_ids = set(reconstruction.registered_images.keys())

    def fallback_select(blocked_ids: Set[int]) -> Optional[int]:
        """
        Fallback when the configured selector returns None.
        Chooses the unblocked image with the most correspondences if it still
        has enough points for a meaningful PnP attempt.
        """
        min_corr = max(min_inliers, 4)
        best_id = None
        best_n = 0

        for img_id in range(len(reconstruction.image_paths)):
            if img_id in blocked_ids:
                continue

            pts2d, _, _ = build_2d3d_correspondences(img_id, reconstruction)
            n_corr = len(pts2d)
            if n_corr >= min_corr and n_corr > best_n:
                best_n = n_corr
                best_id = img_id

        if best_id is not None:
            print(f"[Step 7] Fallback candidate: image {best_id} "
                  f"with {best_n} correspondences")
        return best_id

    failed_ids: Set[int] = set()

    while True:
        # --- 7a. Select next image ---
        blocked_ids = registered_ids | failed_ids
        image_id = view_selector.select(reconstruction, blocked_ids)
        if image_id is None:
            image_id = fallback_select(blocked_ids)
        if image_id is None:
            if failed_ids:
                print(f"[Step 7] No suitable next image found "
                      f"(tried {len(failed_ids)} candidates).")
            else:
                print("[Step 7] No suitable next image found.")
            return None

        # --- 7b. Build 2D-3D correspondences ---
        pts2d, pts3d, _ = build_2d3d_correspondences(image_id, reconstruction)

        print(f"[Step 7] Registering image {image_id}  "
              f"({len(pts2d)} 2D↔3D correspondences)")

        # --- 7c. Solve PnP ---
        cam = reconstruction.cameras[image_id]
        R, t, inlier_mask = pnp_solver.solve(pts3d, pts2d, cam.K, cam.dist_coeffs)

        if R is None or inlier_mask.sum() < min_inliers:
            print(f"  FAILED: only {0 if R is None else inlier_mask.sum()} inliers "
                  f"(need {min_inliers}). Trying another image.")
            failed_ids.add(image_id)
            continue

        # --- 7d. Register camera ---
        reconstruction.registered_images[image_id] = RegisteredImage(
            image_id=image_id,
            camera_id=image_id,
            R=R,
            t=t,
        )

        print(f"  SUCCESS: {inlier_mask.sum()} inliers / {len(pts2d)} correspondences")
        return image_id
