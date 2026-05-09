"""
step3_feature_matching.py
-------------------------
Step 3 — Feature Matching

Responsibilities:
  - Decide which image pairs to match (pair selection strategy)
  - Match descriptors between selected pairs
  - Apply ratio test / cross-check filtering
  - Populate Reconstruction.image_matches

Swappable strategies
  Pair selection:
    ExhaustivePairSelector     — all N*(N-1)/2 pairs
    SequentialPairSelector     — consecutive frames  (video / ordered capture)
    VocabTreePairSelector      — placeholder for image-retrieval-based selection

  Matchers:
    BFMatcher                  — brute-force (L2 for float, Hamming for binary)
    FLANNMatcher               — FLANN approximate nearest-neighbour (float only)
    SuperGlueMatcher           — deep-learning placeholder
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from sfm_types import ImageMatches, Keypoints, Reconstruction, normalize_pair_key


# ============================================================
#  PAIR SELECTION INTERFACE & IMPLEMENTATIONS
# ============================================================

class PairSelectorBase(ABC):
    @abstractmethod
    def select_pairs(self, n_images: int) -> List[Tuple[int, int]]:
        """Return a list of (id_a, id_b) pairs with id_a < id_b."""


class ExhaustivePairSelector(PairSelectorBase):
    """All N*(N-1)/2 pairs. Accurate but O(N²) cost."""

    def select_pairs(self, n_images: int) -> List[Tuple[int, int]]:
        return [(i, j) for i in range(n_images) for j in range(i + 1, n_images)]


class SequentialPairSelector(PairSelectorBase):
    """
    Match each image to the next `window` images.
    Best for video or spatially ordered capture.
    """

    def __init__(self, window: int = 5):
        self.window = window

    def select_pairs(self, n_images: int) -> List[Tuple[int, int]]:
        pairs = []
        for i in range(n_images):
            for j in range(i + 1, min(i + 1 + self.window, n_images)):
                pairs.append((i, j))
        return pairs


class VocabTreePairSelector(PairSelectorBase):
    """
    Placeholder for vocabulary-tree / image-retrieval-based pair selection.
    Supply a `retrieval_model` with:
        top_k_ids = retrieval_model.query(image_id, k)  →  list of int
    """

    def __init__(self, retrieval_model=None, top_k: int = 20):
        if retrieval_model is None:
            raise ValueError("VocabTreePairSelector requires a retrieval_model.")
        if not hasattr(retrieval_model, 'query'):
            raise ValueError(
                "retrieval_model must have a query(image_id, k) method "
                "that returns a list of image IDs."
            )
        self.model = retrieval_model
        self.top_k = top_k

    def select_pairs(self, n_images: int) -> List[Tuple[int, int]]:
        pairs = set()
        for i in range(n_images):
            for j in self.model.query(i, self.top_k):
                a, b = min(i, j), max(i, j)
                if a != b:
                    pairs.add((a, b))
        return sorted(pairs)


# ============================================================
#  MATCHER INTERFACE & IMPLEMENTATIONS
# ============================================================

class DescriptorMatcherBase(ABC):
    @abstractmethod
    def match(
        self,
        kp_a: Keypoints,
        kp_b: Keypoints,
    ) -> np.ndarray:
        """
        Return (M, 2) int32 array of [idx_in_A, idx_in_B].
        Already filtered (ratio test / cross-check).
        """


def _is_binary_descriptor(descs: np.ndarray) -> bool:
    return descs.dtype == np.uint8


class BFMatcher(DescriptorMatcherBase):
    """
    Brute-force matcher.
    - Float descriptors (SIFT, AKAZE-float)  → L2 norm + Lowe ratio test
    - Binary descriptors (ORB, AKAZE-binary) → Hamming + cross-check
    """

    def __init__(self, ratio_thresh: float = 0.8, cross_check: bool = False): #0.75 ratio_thresh: float = 0.75, cross_check: bool = False
        self.ratio_thresh = ratio_thresh
        self.cross_check = cross_check

    def match(self, kp_a: Keypoints, kp_b: Keypoints) -> np.ndarray:
        desc_a, desc_b = kp_a.descriptors, kp_b.descriptors
        if len(desc_a) == 0 or len(desc_b) == 0:
            return np.empty((0, 2), dtype=np.int32)

        binary = _is_binary_descriptor(desc_a)
        norm = cv2.NORM_HAMMING if binary else cv2.NORM_L2

        if binary or self.cross_check:
            # Cross-check: only mutual nearest neighbours
            bf = cv2.BFMatcher(norm, crossCheck=True)
            raw = bf.match(desc_a, desc_b)
            if len(raw) == 0:
                return np.empty((0, 2), dtype=np.int32)
            matches = np.array([[m.queryIdx, m.trainIdx] for m in raw], dtype=np.int32)
        else:
            # kNN + Lowe ratio test
            bf = cv2.BFMatcher(norm, crossCheck=False)
            raw = bf.knnMatch(desc_a, desc_b, k=2)
            matches = []
            for pair in raw:
                if len(pair) == 2 and pair[0].distance < self.ratio_thresh * pair[1].distance:
                    matches.append([pair[0].queryIdx, pair[0].trainIdx])
            matches = np.array(matches, dtype=np.int32) if matches else np.empty((0, 2), dtype=np.int32)

        return matches


class FLANNMatcher(DescriptorMatcherBase):
    """
    FLANN approximate nearest-neighbour matcher.
    Only works with float descriptors (e.g. SIFT).
    Uses Lowe ratio test.
    """

    def __init__(self, ratio_thresh: float = 0.75, n_trees: int = 5, n_checks: int = 50):
        self.ratio_thresh = ratio_thresh
        self.index_params = {"algorithm": 1, "trees": n_trees}    # FLANN_INDEX_KDTREE
        self.search_params = {"checks": n_checks}

    def match(self, kp_a: Keypoints, kp_b: Keypoints) -> np.ndarray:
        desc_a = kp_a.descriptors.astype(np.float32)
        desc_b = kp_b.descriptors.astype(np.float32)

        if len(desc_a) < 2 or len(desc_b) < 2:
            return np.empty((0, 2), dtype=np.int32)

        flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)
        raw = flann.knnMatch(desc_a, desc_b, k=2)

        matches = []
        for pair in raw:
            if len(pair) == 2 and pair[0].distance < self.ratio_thresh * pair[1].distance:
                matches.append([pair[0].queryIdx, pair[0].trainIdx])
        return np.array(matches, dtype=np.int32) if matches else np.empty((0, 2), dtype=np.int32)


class SuperGlueMatcher(DescriptorMatcherBase):
    """
    Placeholder for SuperGlue deep-learning matcher.
    Supply a `model` with:
        matches01 = model.match(kp_a, kp_b)  →  (N,) int array (-1 = unmatched)
    """

    def __init__(self, model=None):
        if model is None:
            raise ValueError("SuperGlueMatcher requires a SuperGlue model instance.")
        self.model = model

    def match(self, kp_a: Keypoints, kp_b: Keypoints) -> np.ndarray:
        matches01 = self.model.match(kp_a, kp_b)   # (N,) with -1 for unmatched
        valid = matches01 >= 0
        idx_a = np.where(valid)[0]
        idx_b = matches01[valid]
        return np.stack([idx_a, idx_b], axis=1).astype(np.int32)


# ============================================================
#  PIPELINE STEP FUNCTION
# ============================================================

def match_features(
    reconstruction: Reconstruction,
    pair_selector: Optional[PairSelectorBase] = None,
    matcher: Optional[DescriptorMatcherBase] = None,
    min_matches: int = 15,#20
) -> Dict[Tuple[int, int], ImageMatches]:
    """
    Step 3 — Match features between selected image pairs.

    Parameters
    ----------
    reconstruction : SfM state — image_matches dict will be populated
    pair_selector  : PairSelectorBase implementation (default: ExhaustivePairSelector)
    matcher        : DescriptorMatcherBase implementation (default: BFMatcher)
    min_matches    : discard pairs with fewer matches than this threshold

    Returns
    -------
    image_matches : dict mapping (id_a, id_b) → ImageMatches
    """
    if pair_selector is None:
        pair_selector = ExhaustivePairSelector()
    if matcher is None:
        matcher = BFMatcher()

    # Validate inputs
    if not reconstruction.image_paths:
        raise ValueError("No images in reconstruction")
    if not reconstruction.keypoints:
        raise ValueError("No keypoints extracted. Run Step 2 first.")
    
    n_images = len(reconstruction.image_paths)
    if len(reconstruction.keypoints) < n_images:
        raise ValueError(
            f"Missing keypoints: expected {n_images}, got {len(reconstruction.keypoints)}"
        )

    pairs = pair_selector.select_pairs(n_images)

    print(f"[Step 3] Feature matching — pair selector: {pair_selector.__class__.__name__}, "
          f"matcher: {matcher.__class__.__name__}")
    print(f"  Matching {len(pairs)} pairs …")

    image_matches: Dict[Tuple[int, int], ImageMatches] = {}

    for id_a, id_b in pairs:
        kp_a = reconstruction.keypoints[id_a]
        kp_b = reconstruction.keypoints[id_b]
        matched = matcher.match(kp_a, kp_b)

        if len(matched) < min_matches:
            continue

        # Normalize pair key for consistent storage
        pair_key = normalize_pair_key(id_a, id_b)
        im = ImageMatches(image_id_a=id_a, image_id_b=id_b, matches=matched)
        image_matches[pair_key] = im
        reconstruction.image_matches[pair_key] = im

    kept = len(image_matches)
    skipped = len(pairs) - kept
    print(f"  Kept {kept} pairs  (skipped {skipped} with < {min_matches} matches)")
    for (a, b), im in sorted(image_matches.items()):
        print(f"  [{a:03d}↔{b:03d}]  {len(im.matches):5d} matches")

    return image_matches
