"""
step6b_component_merging.py
----------------------------
Step 6b — Multi-Component Merging

After multiple independent seeds have each grown their own Reconstruction
("component"), this step finds pairs of components that observe overlapping
physical structure, estimates the similarity transform (rotation + translation
+ uniform scale) that aligns one component's coordinate frame onto the
other's, and folds them together.

Responsibilities:
  - Find 3-D point correspondences shared between two components, via
    verified 2-D matches between an image registered in component A and an
    image registered in component B
  - Estimate the Sim(3) alignment transform with RANSAC (Umeyama's method)
  - Apply the transform and merge camera poses / points into one component
  - Repeat pairwise until no more components can be merged
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from sfm_types import Reconstruction, VerifiedMatches


# ============================================================
#  UMEYAMA SIMILARITY ALIGNMENT
# ============================================================

def _umeyama(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Umeyama (1991) closed-form least-squares similarity transform.
    Finds R (3x3), t (3,), s (scalar) minimizing sum ||dst_i - (s*R@src_i + t)||^2.
    """
    n, dim = src.shape
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0

    R = U @ S @ Vt
    var_src = (src_c ** 2).sum() / n
    s = float(np.trace(np.diag(D) @ S) / var_src) if var_src > 1e-12 else 1.0
    t = mu_dst - s * (R @ mu_src)

    return R, t, s


def _umeyama_ransac(
    src: np.ndarray,
    dst: np.ndarray,
    dist_thresh: float,
    n_iters: int = 2000,
    min_sample: int = 4,
    rng: Optional[np.random.Generator] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, float, np.ndarray]]:
    """
    RANSAC wrapper around Umeyama alignment.

    Returns (R, t, s, inlier_mask) for the best-scoring model, or None if
    there isn't enough data or no model clears `min_sample` inliers.
    """
    n = src.shape[0]
    if n < min_sample:
        return None
    if rng is None:
        rng = np.random.default_rng()

    best_inliers = None
    best_count = -1

    for _ in range(n_iters):
        idx = rng.choice(n, size=min_sample, replace=False)
        try:
            R, t, s = _umeyama(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue

        pred = s * (R @ src.T).T + t
        errs = np.linalg.norm(pred - dst, axis=1)
        inliers = errs < dist_thresh

        if inliers.sum() > best_count:
            best_count = int(inliers.sum())
            best_inliers = inliers

    if best_inliers is None or best_count < min_sample:
        return None

    # Refit using all inliers for a cleaner final estimate
    R, t, s = _umeyama(src[best_inliers], dst[best_inliers])
    pred = s * (R @ src.T).T + t
    errs = np.linalg.norm(pred - dst, axis=1)
    final_inliers = errs < dist_thresh

    return R, t, s, final_inliers


def apply_similarity_transform(component: Reconstruction, R: np.ndarray, t: np.ndarray, s: float) -> None:
    """
    Apply world-space similarity transform xyz' = s*R@xyz + t to every
    camera pose and 3-D point in `component`, in place.

    Camera centre transforms as C' = s*R@C + t (a point in the world).
    Camera rotation transforms as R_cw' = R_cw @ R^T (axes only rotate,
    unaffected by translation/scale — pinhole projection ratios x/Z, y/Z are
    invariant to a uniform rescale of the whole scene including camera
    position, so this exactly preserves every pixel projection).
    """
    for ri in component.registered_images.values():
        C = -ri.R.T @ ri.t
        C_new = s * (R @ C) + t
        R_new = ri.R @ R.T
        t_new = -R_new @ C_new
        ri.R = R_new
        ri.t = t_new

    for p3d in component.points3d.values():
        p3d.xyz = s * (R @ p3d.xyz) + t


# ============================================================
#  CROSS-COMPONENT CORRESPONDENCE FINDING
# ============================================================

def _build_obs_to_pid(component: Reconstruction) -> Dict[Tuple[int, int], int]:
    obs_to_pid = {}
    for pid, p3d in component.points3d.items():
        for (img_id, kp_idx) in p3d.track:
            obs_to_pid[(int(img_id), int(kp_idx))] = pid
    return obs_to_pid


def find_cross_component_correspondences(
    comp_a: Reconstruction,
    comp_b: Reconstruction,
    global_verified_matches: Dict[Tuple[int, int], VerifiedMatches],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find pairs of 3-D points (one from comp_a, one from comp_b) that
    correspond to the same physical keypoint, via verified 2-D matches
    between an image registered in comp_a and an image registered in comp_b.

    Returns
    -------
    pts_a : (N,3) points in comp_a's coordinate frame
    pts_b : (N,3) points in comp_b's coordinate frame  (same physical points)
    """
    obs_to_pid_a = _build_obs_to_pid(comp_a)
    obs_to_pid_b = _build_obs_to_pid(comp_b)

    imgs_a = set(comp_a.registered_images.keys())
    imgs_b = set(comp_b.registered_images.keys())

    pts_a_list, pts_b_list = [], []

    for vm in global_verified_matches.values():
        # Determine, from vm's own stored ids (not the dict key ordering),
        # which column belongs to comp_a and which to comp_b.
        if vm.image_id_a in imgs_a and vm.image_id_b in imgs_b:
            a_img, b_img, col_a, col_b = vm.image_id_a, vm.image_id_b, 0, 1
        elif vm.image_id_b in imgs_a and vm.image_id_a in imgs_b:
            a_img, b_img, col_a, col_b = vm.image_id_b, vm.image_id_a, 1, 0
        else:
            continue

        for m in vm.matches:
            pid_a = obs_to_pid_a.get((a_img, int(m[col_a])))
            pid_b = obs_to_pid_b.get((b_img, int(m[col_b])))
            if pid_a is None or pid_b is None:
                continue
            pts_a_list.append(comp_a.points3d[pid_a].xyz)
            pts_b_list.append(comp_b.points3d[pid_b].xyz)

    if not pts_a_list:
        return np.empty((0, 3)), np.empty((0, 3))

    return np.array(pts_a_list, dtype=np.float64), np.array(pts_b_list, dtype=np.float64)


# ============================================================
#  MERGER INTERFACE & IMPLEMENTATION
# ============================================================

class ComponentMergerBase(ABC):
    @abstractmethod
    def merge(
        self,
        comp_a: Reconstruction,
        comp_b: Reconstruction,
        global_verified_matches: Dict[Tuple[int, int], VerifiedMatches],
    ) -> Optional[Reconstruction]:
        """
        Attempt to merge comp_b into comp_a's coordinate frame (comp_a is
        mutated in place). Returns comp_a on success, or None if not enough
        shared structure was found to align them.
        """


class UmeyamaRANSACMerger(ComponentMergerBase):
    """Umeyama + RANSAC similarity alignment, self-scaling inlier threshold."""

    def __init__(
        self,
        min_correspondences: int = 12,
        inlier_thresh_ratio: float = 0.05,
        min_inliers: int = 8,
        ransac_iters: int = 2000,
    ):
        self.min_correspondences = min_correspondences
        self.inlier_thresh_ratio = inlier_thresh_ratio
        self.min_inliers = min_inliers
        self.ransac_iters = ransac_iters

    def merge(self, comp_a, comp_b, global_verified_matches):
        pts_a, pts_b = find_cross_component_correspondences(comp_a, comp_b, global_verified_matches)
        if len(pts_a) < self.min_correspondences:
            return None

        # Self-scaling inlier threshold: a fraction of comp_a's point spread,
        # since the two components' absolute units are otherwise unknown.
        scale_ref = np.linalg.norm(pts_a - pts_a.mean(axis=0), axis=1).mean() + 1e-9
        dist_thresh = self.inlier_thresh_ratio * scale_ref

        result = _umeyama_ransac(pts_b, pts_a, dist_thresh=dist_thresh, n_iters=self.ransac_iters)
        if result is None:
            return None
        R, t, s, inliers = result
        if inliers.sum() < self.min_inliers:
            return None

        print(f"[Merge] {len(comp_b.registered_images)}-cam component -> "
              f"{len(comp_a.registered_images)}-cam component  "
              f"({int(inliers.sum())}/{len(pts_a)} correspondences, scale={s:.3f})")

        apply_similarity_transform(comp_b, R, t, s)

        # Fold comp_b's points into comp_a with fresh, non-colliding point IDs
        for pid, p3d in comp_b.points3d.items():
            new_pid = comp_a.new_point_id()
            p3d.point3d_id = new_pid
            comp_a.points3d[new_pid] = p3d

        for img_id, ri in comp_b.registered_images.items():
            comp_a.registered_images[img_id] = ri

        return comp_a


# ============================================================
#  PIPELINE STEP FUNCTION
# ============================================================

def merge_components(
    components: List[Reconstruction],
    global_verified_matches: Dict[Tuple[int, int], VerifiedMatches],
    merger: Optional[ComponentMergerBase] = None,
) -> List[Reconstruction]:
    """
    Step 6b — Greedily merge components that share enough cross-component
    structure. Returns the resulting list of components (largest first);
    if more than one remains, no bridging structure was found between them.
    """
    if merger is None:
        merger = UmeyamaRANSACMerger()

    comps = list(components)
    merged_any = True

    while merged_any and len(comps) > 1:
        merged_any = False
        comps.sort(key=lambda c: len(c.registered_images), reverse=True)

        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                result = merger.merge(comps[i], comps[j], global_verified_matches)
                if result is not None:
                    del comps[j]
                    merged_any = True
                    break
            if merged_any:
                break

    print(f"[Merge] Final: {len(comps)} component(s), "
          f"sizes={[len(c.registered_images) for c in comps]}")
    return comps