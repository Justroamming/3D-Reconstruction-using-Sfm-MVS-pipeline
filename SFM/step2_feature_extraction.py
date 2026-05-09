"""
step2_feature_extraction.py
---------------------------
Step 2 — Feature Detection & Description

Responsibilities:
  - Detect keypoints in every image
  - Compute descriptors for each keypoint
  - Populate Reconstruction.keypoints

Swappable strategies
  - SIFTExtractor      — classic SIFT  (cv2.SIFT)
  - ORBExtractor       — fast binary   (cv2.ORB)
  - AKAZEExtractor     — AKAZE         (cv2.AKAZE)
  - SURFExtractor      — SURF (needs opencv-contrib)
  - SuperPointExtractor— deep-learning placeholder (bring your own weights)
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from sfm_types import Keypoints, Reconstruction


# ============================================================
#  INTERFACE
# ============================================================

class FeatureExtractorBase(ABC):
    """
    Detect keypoints AND compute descriptors for a single image.
    Returns a Keypoints object.
    """

    @abstractmethod
    def extract(self, image: np.ndarray, image_id: int) -> Keypoints:
        """
        Parameters
        ----------
        image    : BGR uint8 array
        image_id : index of this image in the reconstruction

        Returns
        -------
        Keypoints dataclass
        """


# ============================================================
#  HELPERS
# ============================================================

def _cv2kp_to_arrays(
    kps: list,
    descs: Optional[np.ndarray],
    image_id: int,
) -> Keypoints:
    """Convert a list of cv2.KeyPoint + descriptor array into our Keypoints type."""
    if len(kps) == 0:
        return Keypoints(
            image_id=image_id,
            points=np.empty((0, 2), dtype=np.float32),
            sizes=np.empty((0,), dtype=np.float32),
            angles=np.empty((0,), dtype=np.float32),
            responses=np.empty((0,), dtype=np.float32),
            descriptors=np.empty((0, 128), dtype=np.float32),
        )

    pts = np.array([kp.pt for kp in kps], dtype=np.float32)            # (N,2)
    sizes = np.array([kp.size for kp in kps], dtype=np.float32)
    angles = np.array([kp.angle for kp in kps], dtype=np.float32)
    responses = np.array([kp.response for kp in kps], dtype=np.float32)

    if descs is None:
        # Default to 128-dim to match SIFT descriptor size
        descs = np.zeros((len(kps), 128), dtype=np.float32)

    return Keypoints(
        image_id=image_id,
        points=pts,
        sizes=sizes,
        angles=angles,
        responses=responses,
        descriptors=descs,
    )


# ============================================================
#  CONCRETE EXTRACTORS
# ============================================================

class SIFTExtractor(FeatureExtractorBase):
    """
    Scale-Invariant Feature Transform (SIFT).
    Float descriptors of dimension 128. Works best for reconstruction quality.
    """

    def __init__(
        self,
        n_features: int = 32768, #5000
        n_octave_layers: int = 4, #3
        contrast_threshold: float = 0.04, #0.04
        edge_threshold: float = 10.0,
        sigma: float = 1.6,
    ):
        self.sift = cv2.SIFT_create(
            nfeatures=n_features,
            nOctaveLayers=n_octave_layers,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
            sigma=sigma,
        )

    def extract(self, image: np.ndarray, image_id: int) -> Keypoints:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kps, descs = self.sift.detectAndCompute(gray, None)
        return _cv2kp_to_arrays(kps, descs, image_id)


class ORBExtractor(FeatureExtractorBase):
    """
    Oriented FAST and Rotated BRIEF (ORB).
    Binary descriptors of dimension 32 bytes (256 bits).
    Fast but less accurate than SIFT for reconstruction.
    """

    def __init__(
        self,
        n_features: int = 8000,
        scale_factor: float = 1.2,
        n_levels: int = 8,
        edge_threshold: int = 31,
        patch_size: int = 31,
        fast_threshold: int = 20,
    ):
        self.orb = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=scale_factor,
            nlevels=n_levels,
            edgeThreshold=edge_threshold,
            patchSize=patch_size,
            fastThreshold=fast_threshold,
        )

    def extract(self, image: np.ndarray, image_id: int) -> Keypoints:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kps, descs = self.orb.detectAndCompute(gray, None)
        return _cv2kp_to_arrays(kps, descs, image_id)


class AKAZEExtractor(FeatureExtractorBase):
    """
    Accelerated-KAZE (AKAZE).
    Binary descriptors (MLDB). Good balance of speed and accuracy.
    """

    def __init__(
        self,
        descriptor_type: int = cv2.AKAZE_DESCRIPTOR_MLDB,
        descriptor_size: int = 0,       # 0 = full size
        descriptor_channels: int = 3,
        threshold: float = 0.001,
        n_octaves: int = 4,
        n_octave_layers: int = 4,
    ):
        self.akaze = cv2.AKAZE_create(
            descriptor_type=descriptor_type,
            descriptor_size=descriptor_size,
            descriptor_channels=descriptor_channels,
            threshold=threshold,
            nOctaves=n_octaves,
            nOctaveLayers=n_octave_layers,
        )

    def extract(self, image: np.ndarray, image_id: int) -> Keypoints:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kps, descs = self.akaze.detectAndCompute(gray, None)
        return _cv2kp_to_arrays(kps, descs, image_id)


class SuperPointExtractor(FeatureExtractorBase):
    """
    Placeholder for SuperPoint (deep-learning detector + descriptor).
    To activate: supply a `model` object that implements
        pts, descs = model.run(gray_float32)  →  (N,2), (N,256)
    """

    def __init__(self, model=None):
        if model is None:
            raise ValueError(
                "SuperPointExtractor requires a SuperPoint model. "
                "Pass model=<your SuperPoint instance>."
            )
        if not hasattr(model, 'run'):
            raise ValueError(
                "SuperPoint model must have a run(image) method "
                "that returns (points, descriptors) arrays."
            )
        self.model = model

    def extract(self, image: np.ndarray, image_id: int) -> Keypoints:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        pts, descs = self.model.run(gray)   # user-supplied model
        n = pts.shape[0]
        return Keypoints(
            image_id=image_id,
            points=pts[:, :2].astype(np.float32),
            sizes=np.ones(n, dtype=np.float32),
            angles=np.zeros(n, dtype=np.float32),
            responses=pts[:, 2].astype(np.float32) if pts.shape[1] > 2 else np.ones(n, dtype=np.float32),
            descriptors=descs.astype(np.float32),
        )


# ============================================================
#  PIPELINE STEP FUNCTION
# ============================================================

def extract_features(
    images: List[np.ndarray],
    reconstruction: Reconstruction,
    extractor: Optional[FeatureExtractorBase] = None,
) -> List[Keypoints]:
    """
    Step 2 — Detect keypoints and compute descriptors for all images.

    Parameters
    ----------
    images         : list of BGR uint8 arrays (from Step 1)
    reconstruction : SfM state — keypoints dict will be populated
    extractor      : FeatureExtractorBase implementation (default: SIFTExtractor)

    Returns
    -------
    keypoints_list : list of Keypoints, one per image (also stored in reconstruction)
    """
    if extractor is None:
        extractor = SIFTExtractor()

    if not images:
        raise ValueError("No images provided to extract_features")
    
    # Validate image format
    for i, img in enumerate(images):
        if not isinstance(img, np.ndarray):
            raise TypeError(f"Image {i} is not a numpy array")
        if len(img.shape) != 3 or img.shape[2] != 3:
            raise ValueError(f"Image {i} must be BGR (H×W×3), got shape {img.shape}")

    keypoints_list: List[Keypoints] = []

    for image_id, image in enumerate(images):
        kp = extractor.extract(image, image_id)
        keypoints_list.append(kp)
        reconstruction.keypoints[image_id] = kp

    print(f"[Step 2] Feature extraction — method: {extractor.__class__.__name__}")
    for kp in keypoints_list:
        print(f"  [img {kp.image_id:03d}] {len(kp.points):6d} keypoints")

    return keypoints_list
