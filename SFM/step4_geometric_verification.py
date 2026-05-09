"""
step4_geometric_verification.py
--------------------------------
Step 4 — Geometric Verification

Responsibilities:
  - Filter putative matches using a geometric model
  - Estimate Fundamental Matrix F or Homography H between each pair
  - Retain only geometrically consistent inliers
  - Populate Reconstruction.verified_matches

Swappable strategies
  - FundamentalRANSAC   — estimate F with standard RANSAC
  - FundamentalMAGSAC   — estimate F with MAGSAC++ (OpenCV ≥ 4.7 with contrib)
  - HomographyRANSAC    — estimate H with RANSAC (good for planar scenes)
  - EssentialRANSAC     — estimate E directly if intrinsics are known
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from sfm_types import ImageMatches, Keypoints, Reconstruction, VerifiedMatches, normalize_pair_key


# ============================================================
#  INTERFACE
# ============================================================

class GeometricVerifierBase(ABC):
    @abstractmethod
    def verify(
        self,
        kp_a: Keypoints,
        kp_b: Keypoints,
        matches: ImageMatches,
    ) -> VerifiedMatches:
        """
        Given raw matches and keypoints, return a VerifiedMatches with
        only the geometrically consistent inliers.
        """


# ============================================================
#  CONCRETE VERIFIERS
# ============================================================

def _pts_from_matches(
    kp_a: Keypoints,
    kp_b: Keypoints,
    matches: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract cleaned (N,2) point arrays for matched keypoints."""
    match_idx = np.asarray(matches, dtype=np.int32)
    if match_idx.size == 0:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty, np.empty((0,), dtype=bool)

    if match_idx.ndim != 2 or match_idx.shape[1] != 2:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty, np.empty((0,), dtype=bool)

    pts_a = np.asarray(kp_a.points[match_idx[:, 0]], dtype=np.float32).reshape(-1, 2)
    pts_b = np.asarray(kp_b.points[match_idx[:, 1]], dtype=np.float32).reshape(-1, 2)

    finite_mask = np.isfinite(pts_a).all(axis=1) & np.isfinite(pts_b).all(axis=1)
    return pts_a[finite_mask], pts_b[finite_mask], finite_mask


def _expand_inlier_mask(
    num_matches: int,
    valid_mask: np.ndarray,
    model_mask: np.ndarray,
) -> Optional[np.ndarray]:
    """Map an OpenCV mask back to the original match list."""
    if model_mask is None:
        return None

    model_mask = np.asarray(model_mask).ravel().astype(bool)
    if len(model_mask) == num_matches:
        return model_mask

    if len(model_mask) == int(valid_mask.sum()):
        expanded = np.zeros(num_matches, dtype=bool)
        expanded[valid_mask] = model_mask
        return expanded

    return None


class FundamentalRANSAC(GeometricVerifierBase):
    """
    Estimate the Fundamental Matrix with RANSAC.
    Works without any knowledge of camera intrinsics.
    """

    def __init__(
        self,
        ransac_reproj_thresh: float = 4.0,#3.0
        confidence: float = 0.999,
        max_iters: int = 10000,
    ):
        self.thresh = ransac_reproj_thresh
        self.confidence = confidence
        self.max_iters = max_iters

    def verify(self, kp_a, kp_b, matches):
        raw_matches = np.asarray(matches.matches, dtype=np.int32)
        pts_a, pts_b, valid_mask = _pts_from_matches(kp_a, kp_b, raw_matches)
        F = None
        inlier_mask = None

        if len(pts_a) >= 8:
            try:
                F, mask = cv2.findFundamentalMat(
                    pts_a, pts_b,
                    method=cv2.FM_RANSAC,
                    ransacReprojThreshold=self.thresh,
                    confidence=self.confidence,
                    maxIters=self.max_iters,
                )
            except cv2.error as exc:
                print(
                    f"[Step 4] Warning: FundamentalRANSAC failed for "
                    f"pair {matches.image_id_a}-{matches.image_id_b}: {exc}"
                )
                F, mask = None, None

            if mask is not None:
                inlier_mask = _expand_inlier_mask(len(raw_matches), valid_mask, mask)

        if inlier_mask is None or inlier_mask.sum() == 0:
            return VerifiedMatches(
                image_id_a=matches.image_id_a,
                image_id_b=matches.image_id_b,
                matches=np.empty((0, 2), dtype=np.int32),
                F=None, H=None,
                inlier_ratio=0.0, num_inliers=0,
            )

        inlier_matches = matches.matches[inlier_mask]
        return VerifiedMatches(
            image_id_a=matches.image_id_a,
            image_id_b=matches.image_id_b,
            matches=inlier_matches,
            F=F, H=None,
            inlier_ratio=inlier_mask.mean(),
            num_inliers=int(inlier_mask.sum()),
        )


class HomographyRANSAC(GeometricVerifierBase):
    """
    Estimate a Homography with RANSAC.
    Best for mostly-planar scenes or pure camera rotation.
    """

    def __init__(
        self,
        ransac_reproj_thresh: float = 4.0,
        confidence: float = 0.999,
        max_iters: int = 10000,
    ):
        self.thresh = ransac_reproj_thresh
        self.confidence = confidence
        self.max_iters = max_iters

    def verify(self, kp_a, kp_b, matches):
        raw_matches = np.asarray(matches.matches, dtype=np.int32)
        pts_a, pts_b, valid_mask = _pts_from_matches(kp_a, kp_b, raw_matches)
        H = None
        inlier_mask = None

        if len(pts_a) >= 4:
            try:
                H, mask = cv2.findHomography(
                    pts_a, pts_b,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=self.thresh,
                    confidence=self.confidence,
                    maxIters=self.max_iters,
                )
            except cv2.error as exc:
                print(
                    f"[Step 4] Warning: HomographyRANSAC failed for "
                    f"pair {matches.image_id_a}-{matches.image_id_b}: {exc}"
                )
                H, mask = None, None

            if mask is not None:
                inlier_mask = _expand_inlier_mask(len(raw_matches), valid_mask, mask)

        if inlier_mask is None or inlier_mask.sum() == 0:
            return VerifiedMatches(
                image_id_a=matches.image_id_a,
                image_id_b=matches.image_id_b,
                matches=np.empty((0, 2), dtype=np.int32),
                F=None, H=None,
                inlier_ratio=0.0, num_inliers=0,
            )

        inlier_matches = matches.matches[inlier_mask]
        return VerifiedMatches(
            image_id_a=matches.image_id_a,
            image_id_b=matches.image_id_b,
            matches=inlier_matches,
            F=None, H=H,
            inlier_ratio=inlier_mask.mean(),
            num_inliers=int(inlier_mask.sum()),
        )


class EssentialRANSAC(GeometricVerifierBase):
    """
    Estimate the Essential Matrix with RANSAC.
    Requires known (or estimated) camera intrinsics K.
    Decomposes to R, t as a bonus.
    """

    def __init__(
        self,
        K: np.ndarray,
        ransac_reproj_thresh: float = 4.0,#1.0
        confidence: float = 0.999,
        max_iters: int = 10000,
    ):
        # Validate K matrix
        if not isinstance(K, np.ndarray):
            raise TypeError("K must be a numpy array")
        if K.shape != (3, 3):
            raise ValueError(f"K must be 3×3, got shape {K.shape}")
        if np.linalg.matrix_rank(K) < 3:
            raise ValueError("K matrix is singular (not invertible)")
        
        self.K = K
        self.thresh = ransac_reproj_thresh
        self.confidence = confidence
        self.max_iters = max_iters

    def verify(self, kp_a, kp_b, matches):
        raw_matches = np.asarray(matches.matches, dtype=np.int32)
        pts_a, pts_b, valid_mask = _pts_from_matches(kp_a, kp_b, raw_matches)
        F = None
        inlier_mask = None

        if len(pts_a) >= 5:
            try:
                E, mask = cv2.findEssentialMat(
                    pts_a, pts_b,
                    cameraMatrix=self.K,
                    method=cv2.RANSAC,
                    prob=self.confidence,
                    threshold=self.thresh,
                    maxIters=self.max_iters,
                )
            except cv2.error as exc:
                print(
                    f"[Step 4] Warning: EssentialRANSAC failed for "
                    f"pair {matches.image_id_a}-{matches.image_id_b}: {exc}"
                )
                E, mask = None, None

            if mask is not None:
                inlier_mask = _expand_inlier_mask(len(raw_matches), valid_mask, mask)
                # Derive F from E for bookkeeping
                K_inv = np.linalg.inv(self.K)
                F = K_inv.T @ E @ K_inv if E is not None else None

        if inlier_mask is None or inlier_mask.sum() == 0:
            return VerifiedMatches(
                image_id_a=matches.image_id_a,
                image_id_b=matches.image_id_b,
                matches=np.empty((0, 2), dtype=np.int32),
                F=None, H=None,
                inlier_ratio=0.0, num_inliers=0,
            )

        inlier_matches = matches.matches[inlier_mask]
        return VerifiedMatches(
            image_id_a=matches.image_id_a,
            image_id_b=matches.image_id_b,
            matches=inlier_matches,
            F=F, H=None,
            inlier_ratio=inlier_mask.mean(),
            num_inliers=int(inlier_mask.sum()),
        )


class MAGSACVerifier(GeometricVerifierBase):
    """
    MAGSAC++ — marginalised RANSAC, more robust to noise.
    Requires opencv-contrib (cv2.USAC_MAGSAC flag).
    Falls back to standard RANSAC if unavailable.
    """

    def __init__(
        self,
        ransac_reproj_thresh: float = 4.0,#3.0
        confidence: float = 0.999,
        max_iters: int = 10000,
    ):
        self.thresh = ransac_reproj_thresh
        self.confidence = confidence
        self.max_iters = max_iters

        # Check MAGSAC availability
        self._method = getattr(cv2, "USAC_MAGSAC", cv2.FM_RANSAC)
        if self._method == cv2.FM_RANSAC:
            print("[MAGSACVerifier] MAGSAC not available, falling back to RANSAC.")

    def verify(self, kp_a, kp_b, matches):
        raw_matches = np.asarray(matches.matches, dtype=np.int32)
        pts_a, pts_b, valid_mask = _pts_from_matches(kp_a, kp_b, raw_matches)
        F = None
        inlier_mask = None

        if len(pts_a) >= 8:
            try:
                F, mask = cv2.findFundamentalMat(
                    pts_a, pts_b,
                    method=self._method,
                    ransacReprojThreshold=self.thresh,
                    confidence=self.confidence,
                    maxIters=self.max_iters,
                )
            except cv2.error as exc:
                print(
                    f"[Step 4] Warning: MAGSACVerifier failed for "
                    f"pair {matches.image_id_a}-{matches.image_id_b}: {exc}"
                )
                F, mask = None, None

            if mask is not None:
                inlier_mask = _expand_inlier_mask(len(raw_matches), valid_mask, mask)

        if inlier_mask is None or inlier_mask.sum() == 0:
            return VerifiedMatches(
                image_id_a=matches.image_id_a,
                image_id_b=matches.image_id_b,
                matches=np.empty((0, 2), dtype=np.int32),
                F=None, H=None,
                inlier_ratio=0.0, num_inliers=0,
            )

        inlier_matches = matches.matches[inlier_mask]
        return VerifiedMatches(
            image_id_a=matches.image_id_a,
            image_id_b=matches.image_id_b,
            matches=inlier_matches,
            F=F, H=None,
            inlier_ratio=inlier_mask.mean(),
            num_inliers=int(inlier_mask.sum()),
        )


# ============================================================
#  PIPELINE STEP FUNCTION
# ============================================================

def verify_matches(
    reconstruction: Reconstruction,
    verifier: Optional[GeometricVerifierBase] = None,
    min_inliers: int = 15,
) -> Dict[Tuple[int, int], VerifiedMatches]:
    """
    Step 4 — Geometric verification of all matched pairs.

    Parameters
    ----------
    reconstruction : SfM state — verified_matches dict will be populated
    verifier       : GeometricVerifierBase implementation (default: FundamentalRANSAC)
    min_inliers    : discard pairs with fewer inliers than this threshold

    Returns
    -------
    verified : dict mapping (id_a, id_b) → VerifiedMatches
    """
    if verifier is None:
        verifier = FundamentalRANSAC()

    # Validate inputs
    if not reconstruction.image_matches:
        print("[Step 4] No matched pairs to verify.")
        return {}
    
    print(f"[Step 4] Geometric verification — method: {verifier.__class__.__name__}")

    verified: Dict[Tuple[int, int], VerifiedMatches] = {}

    for (id_a, id_b), image_match in reconstruction.image_matches.items():
        # Validate keypoints exist
        if id_a not in reconstruction.keypoints or id_b not in reconstruction.keypoints:
            continue
        
        kp_a = reconstruction.keypoints[id_a]
        kp_b = reconstruction.keypoints[id_b]

        vm = verifier.verify(kp_a, kp_b, image_match)

        if vm.num_inliers < min_inliers:
            continue

        # Use normalized pair key for consistent storage
        pair_key = normalize_pair_key(id_a, id_b)
        verified[pair_key] = vm
        reconstruction.verified_matches[pair_key] = vm

    skipped = len(reconstruction.image_matches) - len(verified)
    print(f"  Verified {len(verified)} pairs  (dropped {skipped} with < {min_inliers} inliers)")
    for (a, b), vm in sorted(verified.items()):
        print(f"  [{a:03d}↔{b:03d}]  {vm.num_inliers:4d} inliers  "
              f"ratio={vm.inlier_ratio:.2f}")

    return verified
