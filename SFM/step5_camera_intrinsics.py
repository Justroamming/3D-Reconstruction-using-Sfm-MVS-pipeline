"""
step5_camera_intrinsics.py
--------------------------
Step 5 — Camera Intrinsics Estimation

Responsibilities:
  - Determine the intrinsic camera matrix K and distortion coefficients
  - Support multiple sources: EXIF, calibration file, or estimation from image size
  - Populate Reconstruction.cameras

Swappable strategies
  - EXIFIntrinsicsEstimator   — derive K from EXIF focal length + sensor size
  - CalibrationFileEstimator  — load K + dist from a JSON/YAML calibration file
  - PrincipalPointEstimator   — assume fx=fy from EXIF, cx/cy at image centre (no dist)
  - MockIntrinsicsEstimator   — unit-test helper with a fixed K
"""

import json
import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from sfm_types import Camera, Reconstruction


def _validate_K_matrix(K: np.ndarray) -> None:
    """Validate that K is a valid 3x3 intrinsic matrix."""
    if not isinstance(K, np.ndarray):
        raise TypeError("K must be a numpy array")
    if K.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got shape {K.shape}")
    if np.linalg.matrix_rank(K) < 3:
        raise ValueError("K matrix is singular")


# ============================================================
#  INTERFACE
# ============================================================

class IntrinsicsEstimatorBase(ABC):
    @abstractmethod
    def estimate(
        self,
        image_id: int,
        image_width: int,
        image_height: int,
        exif: dict,
    ) -> Camera:
        """
        Return a Camera with intrinsic matrix K and distortion coefficients.

        Parameters
        ----------
        image_id     : image index
        image_width  : image width in pixels
        image_height : image height in pixels
        exif         : dict from Step 1 loader (focal_length_mm, sensor_width_mm, …)
        """


# ============================================================
#  CONCRETE ESTIMATORS
# ============================================================

class EXIFIntrinsicsEstimator(IntrinsicsEstimatorBase):
    """
    Derive focal length in pixels from EXIF metadata.

        fx = (focal_length_mm / sensor_width_mm) * image_width_px

    Principal point at image centre. No distortion assumed.
    Falls back to a heuristic (focal = 1.2 * max(w, h)) if EXIF is incomplete.
    """

    def __init__(self, assume_no_distortion: bool = True):
        self.assume_no_distortion = assume_no_distortion

    def estimate(self, image_id, image_width, image_height, exif) -> Camera:
        fl_mm = exif.get("focal_length_mm")
        sw_mm = exif.get("sensor_width_mm")

        if fl_mm and sw_mm and sw_mm > 0:
            fx = (fl_mm / sw_mm) * image_width
        else:
            # Heuristic: focal ≈ 1.2 × max dimension
            fx = 1.2 * max(image_width, image_height)
            print(f"  [img {image_id:03d}] No EXIF focal length — using heuristic fx={fx:.1f} px")

        fy = fx
        cx = image_width / 2.0
        cy = image_height / 2.0

        K = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1],
        ], dtype=np.float64)

        dist = np.zeros(5, dtype=np.float64)

        return Camera(
            camera_id=image_id,
            model="pinhole",
            width=image_width,
            height=image_height,
            K=K,
            dist_coeffs=dist,
        )


class CalibrationFileEstimator(IntrinsicsEstimatorBase):
    """
    Load K and distortion from a JSON calibration file produced by, e.g.,
    OpenCV's calibrateCamera or COLMAP's camera format.

    Expected JSON format:
    {
      "camera_matrix": [[fx,0,cx],[0,fy,cy],[0,0,1]],
      "dist_coeffs": [k1, k2, p1, p2, k3],
      "width": 1920,
      "height": 1080,
      "model": "opencv"
    }

    One shared calibration is applied to all images by default.
    """

    def __init__(self, calibration_path: str):
        with open(calibration_path, "r") as f:
            data = json.load(f)
        
        if "camera_matrix" not in data:
            raise KeyError("Calibration must have camera_matrix")
        if "dist_coeffs" not in data:
            raise KeyError("Calibration must have dist_coeffs")
        
        self.K = np.array(data["camera_matrix"], dtype=np.float64)
        _validate_K_matrix(self.K)
        
        self.dist = np.array(data["dist_coeffs"], dtype=np.float64)
        self.width = data.get("width", 0)
        self.height = data.get("height", 0)
        self.model = data.get("model", "opencv")

    def estimate(self, image_id, image_width, image_height, exif) -> Camera:
        return Camera(
            camera_id=image_id,
            model=self.model,
            width=image_width or self.width,
            height=image_height or self.height,
            K=self.K.copy(),
            dist_coeffs=self.dist.copy(),
        )


class PrincipalPointEstimator(IntrinsicsEstimatorBase):
    """
    Simplest possible estimator:
      - fx = fy = image diagonal / sqrt(2)   (very rough)
      - cx, cy at image centre
      - zero distortion
    Useful when no EXIF and no calibration file are available.
    """

    def estimate(self, image_id, image_width, image_height, exif) -> Camera:
        diag = np.sqrt(image_width ** 2 + image_height ** 2)
        f = diag / np.sqrt(2)
        K = np.array([
            [f, 0, image_width / 2.0],
            [0, f, image_height / 2.0],
            [0, 0, 1.0],
        ], dtype=np.float64)
        return Camera(
            camera_id=image_id,
            model="pinhole",
            width=image_width,
            height=image_height,
            K=K,
            dist_coeffs=np.zeros(5, dtype=np.float64),
        )


class FixedIntrinsicsEstimator(IntrinsicsEstimatorBase):
    """
    Use a pre-built Camera object for every image.
    Useful when the exact K is already known.
    """

    def __init__(self, K: np.ndarray, dist_coeffs: Optional[np.ndarray] = None,
                 model: str = "pinhole"):
        _validate_K_matrix(K)
        self.K = K
        self.dist = dist_coeffs if dist_coeffs is not None else np.zeros(5, dtype=np.float64)
        self.model = model

    def estimate(self, image_id, image_width, image_height, exif) -> Camera:
        return Camera(
            camera_id=image_id,
            model=self.model,
            width=image_width,
            height=image_height,
            K=self.K.copy(),
            dist_coeffs=self.dist.copy(),
        )


# ============================================================
#  PIPELINE STEP FUNCTION
# ============================================================

def estimate_intrinsics(
    images: list,           # list of BGR arrays (to read w/h)
    exif_list: List[dict],
    reconstruction: Reconstruction,
    estimator: Optional[IntrinsicsEstimatorBase] = None,
    shared_camera: bool = False,
) -> Dict[int, Camera]:
    """
    Step 5 — Estimate camera intrinsics for every image.

    Parameters
    ----------
    images         : list of BGR arrays (from Step 1) — used only for shape
    exif_list      : list of EXIF dicts (from Step 1)
    reconstruction : SfM state — cameras dict will be populated
    estimator      : IntrinsicsEstimatorBase implementation
                     (default: EXIFIntrinsicsEstimator)
    shared_camera  : if True, estimate intrinsics once and reuse for all images
                     (assumes all images share the same camera/lens)

    Returns
    -------
    cameras : dict mapping camera_id → Camera
    """
    if estimator is None:
        estimator = EXIFIntrinsicsEstimator()

    if not images:
        raise ValueError("No images provided")
    if len(images) != len(exif_list):
        raise ValueError(f"Image count {len(images)} != EXIF count {len(exif_list)}")

    print(f"[Step 5] Intrinsics estimation — method: {estimator.__class__.__name__}  "
          f"shared_camera={shared_camera}")

    cameras: Dict[int, Camera] = {}
    shared_cam: Optional[Camera] = None

    for image_id, (image, exif) in enumerate(zip(images, exif_list)):
        h, w = image.shape[:2]

        if shared_camera and shared_cam is not None:
            cam = Camera(
                camera_id=image_id,
                model=shared_cam.model,
                width=w,
                height=h,
                K=shared_cam.K.copy(),
                dist_coeffs=shared_cam.dist_coeffs.copy(),
            )
        else:
            cam = estimator.estimate(image_id, w, h, exif)
            if shared_camera:
                shared_cam = cam

        cameras[image_id] = cam
        reconstruction.cameras[image_id] = cam

        fx = cam.K[0, 0]
        fy = cam.K[1, 1]
        cx = cam.K[0, 2]
        cy = cam.K[1, 2]

        fl_mm = exif.get("focal_length_mm")
        sw_mm = exif.get("sensor_width_mm")
        if shared_camera and shared_cam is not None and image_id > 0:
            source = "shared_camera"
        elif fl_mm and sw_mm and sw_mm > 0:
            source = "exif"
        else:
            source = "heuristic"

        print(
            f"  [img {image_id:03d}] {w}x{h}  "
            f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f} px  "
            f"model={cam.model} source={source} "
            f"f_mm={fl_mm} sensor_w_mm={sw_mm}"
        )

    return cameras
