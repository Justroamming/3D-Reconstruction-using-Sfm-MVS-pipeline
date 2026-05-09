"""
step9_bundle_adjustment.py
--------------------------
Step 9 — Bundle Adjustment

Responsibilities:
  - Jointly optimise all camera poses AND 3-D point positions
  - Minimise reprojection error across all observations
  - Support local BA (only recently added cameras) and full global BA
  - Populate/update Reconstruction.registered_images and points3d

Swappable strategies
  - SciPyBundleAdjuster    — pure-Python sparse Levenberg-Marquardt (scipy.optimize)
                             No extra dependencies. Slower but portable.
  - LocalBundleAdjuster    — optimise only the last N registered cameras + their points
  - StubBundleAdjuster     — no-op passthrough (disable BA, useful for debugging)

Note: Production SfM systems use Ceres Solver (C++) or g2o.
      A Python binding for Ceres (pyceres) can drop in here when available.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Set

import cv2

from sfm_types import Reconstruction


# ============================================================
#  HELPERS — ROTATION REPRESENTATION
# ============================================================

def _rot_to_rvec(R: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(R)
    return rvec.ravel()


def _rvec_to_rot(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    return R


def _pack_params(reconstruction: Reconstruction, image_ids: list, point_ids: list) -> np.ndarray:
    """Pack camera poses + 3-D points into a flat parameter vector."""
    cam_params = []
    for iid in image_ids:
        ri = reconstruction.registered_images[iid]
        rvec = _rot_to_rvec(ri.R)          # 3 params
        tvec = ri.t                         # 3 params
        cam_params.append(np.hstack([rvec, tvec]))   # 6 per camera

    pt_params = []
    for pid in point_ids:
        pt_params.append(reconstruction.points3d[pid].xyz)  # 3 per point

    return np.hstack([np.hstack(cam_params), np.hstack(pt_params)])


def _unpack_params(
    params: np.ndarray,
    image_ids: list,
    point_ids: list,
    reconstruction: Reconstruction,
):
    """Write optimised parameters back into the reconstruction."""
    try:
        n_cams = len(image_ids)
        expected_size = n_cams * 6 + len(point_ids) * 3
        if len(params) != expected_size:
            raise ValueError(
                f"Parameter size mismatch: expected {expected_size}, got {len(params)}"
            )
        
        cam_block = params[: n_cams * 6].reshape(n_cams, 6)
        pt_block  = params[n_cams * 6 :].reshape(len(point_ids), 3)
    except (ValueError, RuntimeError) as e:
        print(f"[_unpack_params] Error reshaping parameters: {e}")
        return

    for i, iid in enumerate(image_ids):
        ri = reconstruction.registered_images[iid]
        ri.R = _rvec_to_rot(cam_block[i, :3])
        ri.t = cam_block[i, 3:6]

    for i, pid in enumerate(point_ids):
        reconstruction.points3d[pid].xyz = pt_block[i]


# ============================================================
#  INTERFACE
# ============================================================

class BundleAdjusterBase(ABC):
    @abstractmethod
    def run(
        self,
        reconstruction: Reconstruction,
        fixed_image_ids: Optional[Set[int]] = None,
    ) -> float:
        """
        Optimise in-place.

        Parameters
        ----------
        reconstruction  : SfM state (modified in-place)
        fixed_image_ids : camera IDs whose pose is held fixed (e.g. the origin)

        Returns
        -------
        mean_reproj_error : float — average reprojection error after optimisation
        """


# ============================================================
#  CONCRETE ADJUSTERS
# ============================================================

class StubBundleAdjuster(BundleAdjusterBase):
    """No-op adjuster — skip BA entirely. Useful for debugging the pipeline."""

    def run(self, reconstruction, fixed_image_ids=None) -> float:
        print("[Step 9] Bundle adjustment SKIPPED (StubBundleAdjuster)")
        return _compute_mean_reproj_error(reconstruction)


class SciPyBundleAdjuster(BundleAdjusterBase):
    """
    Sparse Levenberg-Marquardt bundle adjustment via scipy.optimize.least_squares.

    Uses analytic sparsity structure (Jacobian sparsity pattern) for speed.
    Suitable for small-to-medium reconstructions (<= ~50 cameras, ~20 k points).
    """

    def __init__(
        self,
        max_nfev: int = 200,
        ftol: float = 1e-4,
        xtol: float = 1e-6,
        loss: str = "huber",          # 'linear' | 'huber' | 'cauchy' | 'soft_l1'
        verbose: int = 0,
    ):
        self.max_nfev = max_nfev
        self.ftol = ftol
        self.xtol = xtol
        self.loss = loss
        self.verbose = verbose

    def run(self, reconstruction, fixed_image_ids=None) -> float:
        try:
            from scipy.optimize import least_squares
            from scipy.sparse import lil_matrix
        except ImportError:
            print("[SciPyBundleAdjuster] scipy not available — skipping BA.")
            return _compute_mean_reproj_error(reconstruction)

        if fixed_image_ids is None:
            fixed_image_ids = set()

        image_ids = list(reconstruction.registered_images.keys())
        point_ids = list(reconstruction.points3d.keys())

        if len(image_ids) < 2 or len(point_ids) < 5:
            print("[Step 9] Not enough cameras/points for BA.")
            return _compute_mean_reproj_error(reconstruction)

        # Build observation list: (cam_idx, pt_idx, observed_2d, K)
        observations = _collect_observations(reconstruction, image_ids, point_ids)
        if len(observations) == 0:
            return _compute_mean_reproj_error(reconstruction)

        x0 = _pack_params(reconstruction, image_ids, point_ids)
        n_cams = len(image_ids)
        n_pts  = len(point_ids)

        # Jacobian sparsity
        n_residuals = len(observations) * 2
        n_params    = n_cams * 6 + n_pts * 3
        jac_sparsity = _build_sparsity(observations, n_cams, n_pts, image_ids, point_ids)

        # Indices of fixed camera parameters
        fixed_mask = np.zeros(n_params, dtype=bool)
        for i, iid in enumerate(image_ids):
            if iid in fixed_image_ids:
                fixed_mask[i * 6: (i + 1) * 6] = True

        def residuals(params):
            # Write temp params (respect fixed cameras)
            p = params.copy()
            p[fixed_mask] = x0[fixed_mask]
            res = _compute_residuals(p, observations, n_cams, image_ids, point_ids)
            return res

        result = least_squares(
            residuals,
            x0,
            jac_sparsity=jac_sparsity,
            method="trf",
            loss=self.loss,
            ftol=self.ftol,
            xtol=self.xtol,
            max_nfev=self.max_nfev,
            verbose=self.verbose,
        )

        _unpack_params(result.x, image_ids, point_ids, reconstruction)

        mean_err = _compute_mean_reproj_error(reconstruction)
        print(f"[Step 9] BA done  cost: {result.cost:.4f}  "
              f"mean reproj err: {mean_err:.3f} px  "
              f"iters: {result.nfev}")
        return mean_err


class LocalBundleAdjuster(BundleAdjusterBase):
    """
    Optimise only the last `window` registered cameras and their observed 3-D points.
    Much faster per-iteration than full BA; good for large scenes.
    """

    def __init__(self, window: int = 5, inner: Optional[BundleAdjusterBase] = None):
        self.window = window
        self.inner = inner or SciPyBundleAdjuster()

    def run(self, reconstruction, fixed_image_ids=None) -> float:
        all_ids = list(reconstruction.registered_images.keys())
        
        if not all_ids:
            print("[LocalBA] No registered images. Skipping BA.")
            return _compute_mean_reproj_error(reconstruction)
        
        local_ids = set(all_ids[-self.window:])

        # Find points observed by local cameras
        local_point_ids = set()
        for pid, p3d in reconstruction.points3d.items():
            if any(obs_id in local_ids for obs_id, _ in p3d.track):
                local_point_ids.add(pid)

        if not local_point_ids:
            print(f"[LocalBA] No points in local window [{len(local_ids)} cams]. Skipping BA.")
            return _compute_mean_reproj_error(reconstruction)

        # Build a mini-reconstruction view
        mini = _sub_reconstruction(reconstruction, local_ids, local_point_ids)

        # Fix the oldest camera in the window as anchor
        fixed = {all_ids[-(self.window)]} if fixed_image_ids is None else fixed_image_ids

        print(f"[Step 9] Local BA: {len(local_ids)} cams, {len(local_point_ids)} points")
        err = self.inner.run(mini, fixed_image_ids=fixed)

        # Write back
        for iid in local_ids:
            reconstruction.registered_images[iid] = mini.registered_images[iid]
        for pid in local_point_ids:
            reconstruction.points3d[pid] = mini.points3d[pid]

        return err


# ============================================================
#  INTERNAL HELPERS
# ============================================================

def _collect_observations(reconstruction, image_ids, point_ids):
    """Return list of (cam_idx, pt_idx, pt2d, K, dist)."""
    iid_to_idx = {iid: i for i, iid in enumerate(image_ids)}
    pid_to_idx = {pid: i for i, pid in enumerate(point_ids)}

    obs = []
    for pid in point_ids:
        p3d = reconstruction.points3d[pid]
        for (obs_img_id, kp_idx) in p3d.track:
            if obs_img_id not in iid_to_idx:
                continue
            cam = reconstruction.cameras.get(obs_img_id)
            if cam is None:
                continue
            kp = reconstruction.keypoints.get(obs_img_id)
            if kp is None or kp_idx >= len(kp.points):
                continue
            obs.append((
                iid_to_idx[obs_img_id],
                pid_to_idx[pid],
                kp.points[kp_idx].astype(np.float64),
                cam.K,
                cam.dist_coeffs,
            ))
    return obs


def _compute_residuals(params, observations, n_cams, image_ids, point_ids):
    cam_block = params[:n_cams * 6].reshape(n_cams, 6)
    pt_block  = params[n_cams * 6:].reshape(len(point_ids), 3)
    res = []
    for (ci, pi, pt2d, K, dist) in observations:
        rvec = cam_block[ci, :3]
        tvec = cam_block[ci, 3:6]
        xyz  = pt_block[pi]
        proj, _ = cv2.projectPoints(
            xyz.reshape(1, 1, 3),
            rvec.reshape(3, 1),
            tvec.reshape(3, 1),
            K, dist,
        )
        proj = proj.reshape(2)
        res.extend((proj - pt2d).tolist())
    return np.array(res, dtype=np.float64)


def _build_sparsity(observations, n_cams, n_pts, image_ids, point_ids):
    from scipy.sparse import lil_matrix
    n_res = len(observations) * 2
    n_par = n_cams * 6 + n_pts * 3
    S = lil_matrix((n_res, n_par), dtype=np.int8)
    for i, (ci, pi, *_) in enumerate(observations):
        S[2*i:2*i+2, ci*6:(ci+1)*6] = 1
        S[2*i:2*i+2, n_cams*6 + pi*3: n_cams*6 + (pi+1)*3] = 1
    return S.tocsr()


def _compute_mean_reproj_error(reconstruction: Reconstruction) -> float:
    errors = []
    for pid, p3d in reconstruction.points3d.items():
        for (img_id, kp_idx) in p3d.track:
            ri = reconstruction.registered_images.get(img_id)
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


def _sub_reconstruction(reconstruction: Reconstruction, image_ids: set, point_ids: set):
    """Shallow copy of a Reconstruction limited to the given IDs."""
    from sfm_types import Reconstruction as R
    mini = R()
    mini.image_paths = reconstruction.image_paths
    
    # Copy cameras with validation
    missing_cameras = image_ids - set(reconstruction.cameras.keys())
    if missing_cameras:
        print(f"[_sub_reconstruction] Warning: {len(missing_cameras)} cameras missing")
    mini.cameras = {i: reconstruction.cameras[i] for i in image_ids if i in reconstruction.cameras}
    
    # Copy keypoints with validation
    missing_keypoints = image_ids - set(reconstruction.keypoints.keys())
    if missing_keypoints:
        print(f"[_sub_reconstruction] Warning: {len(missing_keypoints)} keypoint sets missing")
    mini.keypoints = {i: reconstruction.keypoints[i] for i in image_ids if i in reconstruction.keypoints}
    
    mini.registered_images = {i: reconstruction.registered_images[i] for i in image_ids}
    mini.points3d = {p: reconstruction.points3d[p] for p in point_ids}
    mini.verified_matches = reconstruction.verified_matches
    mini.next_point_id = reconstruction.next_point_id
    return mini


# ============================================================
#  PIPELINE STEP FUNCTION
# ============================================================

def run_bundle_adjustment(
    reconstruction: Reconstruction,
    adjuster: Optional[BundleAdjusterBase] = None,
    fixed_image_ids: Optional[Set[int]] = None,
) -> float:
    """
    Step 9 — Run bundle adjustment on the current reconstruction.

    Parameters
    ----------
    reconstruction  : SfM state (modified in-place)
    adjuster        : BundleAdjusterBase (default: SciPyBundleAdjuster)
    fixed_image_ids : camera IDs to hold fixed (default: first registered image)

    Returns
    -------
    mean_reproj_error : float
    """
    # Validate reconstruction state
    if not reconstruction.registered_images:
        print("[Step 9] No registered images. Cannot run BA.")
        return 0.0
    if not reconstruction.points3d:
        print("[Step 9] No 3-D points. Cannot run BA.")
        return 0.0
    
    if adjuster is None:
        adjuster = SciPyBundleAdjuster()

    if fixed_image_ids is None:
        # Fix the very first camera to remove gauge ambiguity
        first_id = next(iter(reconstruction.registered_images))
        fixed_image_ids = {first_id}

    return adjuster.run(reconstruction, fixed_image_ids=fixed_image_ids)
