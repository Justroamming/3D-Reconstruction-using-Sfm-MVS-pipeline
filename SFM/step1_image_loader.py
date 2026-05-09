"""
step1_image_loader.py
---------------------
Step 1 — Image Input & Preprocessing

Responsibilities:
  - Load images from a directory
  - Extract EXIF metadata (focal length, sensor size)
  - Optional resize / colour normalisation
  - Populate Reconstruction.image_paths

Swappable strategies
  - loaders: BasicImageLoader  (cv2, reads EXIF via Pillow)
  - preprocessors: NoOpPreprocessor, ResizePreprocessor
"""

import os
import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple

from sfm_types import Reconstruction

try:
    from PIL import Image as PILImage
    from PIL.ExifTags import TAGS
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


def _exif_value_to_float(value) -> Optional[float]:
    """Convert common EXIF numeric encodings into float."""
    if value is None:
        return None

    # PIL rational-like object
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        try:
            den = float(value.denominator)
            if den == 0:
                return None
            return float(value.numerator) / den
        except (TypeError, ValueError):
            return None

    # Tuple/list rational, e.g. (35, 10)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            num = float(value[0])
            den = float(value[1])
            if den == 0:
                return None
            return num / den
        except (TypeError, ValueError):
            return None

    # String rational, e.g. "35/10"
    if isinstance(value, str) and "/" in value:
        parts = value.split("/", 1)
        if len(parts) == 2:
            try:
                num = float(parts[0].strip())
                den = float(parts[1].strip())
                if den == 0:
                    return None
                return num / den
            except ValueError:
                return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ============================================================
#  INTERFACES
# ============================================================

class ImageLoaderBase(ABC):
    @abstractmethod
    def load(self, path: str) -> np.ndarray:
        """Return BGR uint8 image array."""

    @abstractmethod
    def extract_exif(self, path: str) -> dict:
        """Return dict with keys: focal_length_mm, sensor_width_mm, gps, …"""


class ImagePreprocessorBase(ABC):
    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        """Return processed BGR image."""


# ============================================================
#  CONCRETE LOADERS
# ============================================================

class BasicImageLoader(ImageLoaderBase):
    """
    Load images with OpenCV.
    Extract EXIF with Pillow (if available).
    """

    SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    _printed_pillow_warning = False

    def load(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {path}")
        return img

    def extract_exif(self, path: str) -> dict:
        meta = {
            "focal_length_mm": None,
            "sensor_width_mm": None,
            "exif_source": "none",
            "sensor_width_source": "none",
            "image_width_px": None,
            "image_height_px": None,
            "gps": None,
        }

        img = cv2.imread(path)
        if img is not None:
            meta["image_height_px"] = img.shape[0]
            meta["image_width_px"]  = img.shape[1]

        if not _HAS_PIL:
            if not BasicImageLoader._printed_pillow_warning:
                print("[Step 1] Pillow is not installed, EXIF extraction is disabled.")
                BasicImageLoader._printed_pillow_warning = True
            return meta

        try:
            with PILImage.open(path) as pil_img:
                # getexif() can omit some EXIF-subIFD tags on certain files.
                # Fall back to _getexif() if focal is not found.
                exif_data = pil_img.getexif()
                tag_map = {}
                if exif_data is not None and len(exif_data) > 0:
                    tag_map = {TAGS.get(k, k): v for k, v in exif_data.items()}
                    meta["exif_source"] = "getexif"

                if "FocalLength" not in tag_map and hasattr(pil_img, "_getexif"):
                    legacy_exif = pil_img._getexif() or {}
                    if legacy_exif:
                        meta["exif_source"] = "_getexif" if not tag_map else "getexif+_getexif"
                    for k, v in legacy_exif.items():
                        tag_name = TAGS.get(k, k)
                        if tag_name not in tag_map:
                            tag_map[tag_name] = v

                if not tag_map:
                    return meta

                # focal length
                if "FocalLength" in tag_map:
                    meta["focal_length_mm"] = _exif_value_to_float(tag_map["FocalLength"])

                # 35mm-equivalent → derive sensor width via crop factor
                if "FocalLengthIn35mmFilm" in tag_map and meta["focal_length_mm"]:
                    eq35 = _exif_value_to_float(tag_map["FocalLengthIn35mmFilm"])
                    if eq35 and eq35 > 0:
                        crop = eq35 / meta["focal_length_mm"]
                        # 35mm full-frame sensor width = 36 mm
                        meta["sensor_width_mm"] = 36.0 / crop
                        meta["sensor_width_source"] = "35mm_equivalent"

        except Exception as exc:
            print(f"[Step 1] EXIF parse failed for '{os.path.basename(path)}': {exc}")

        return meta


# ============================================================
#  CONCRETE PREPROCESSORS
# ============================================================

class NoOpPreprocessor(ImagePreprocessorBase):
    """Pass the image through unchanged."""
    def process(self, image: np.ndarray) -> np.ndarray:
        return image


class ResizePreprocessor(ImagePreprocessorBase):
    """
    Resize so that the longer side equals `max_side` pixels.
    Useful for speeding up feature detection on large images.
    """
    def __init__(self, max_side: int = 1600):
        self.max_side = max_side

    def process(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if max(h, w) <= self.max_side:
            return image
        scale = self.max_side / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


class ColourNormPreprocessor(ImagePreprocessorBase):
    """
    CLAHE histogram equalisation on the L channel (LAB space).
    Helps feature detection on unevenly lit images.
    """
    def __init__(self, clip_limit: float = 2.0, tile_grid: Tuple[int, int] = (8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)

    def process(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


class ChainPreprocessor(ImagePreprocessorBase):
    """Apply a list of preprocessors in sequence."""
    def __init__(self, steps: List[ImagePreprocessorBase]):
        self.steps = steps

    def process(self, image: np.ndarray) -> np.ndarray:
        for step in self.steps:
            image = step.process(image)
        return image


# ============================================================
#  PIPELINE STEP FUNCTION
# ============================================================

def load_images(
    image_dir: str,
    reconstruction: Reconstruction,
    loader: Optional[ImageLoaderBase] = None,
    preprocessor: Optional[ImagePreprocessorBase] = None,
    extensions: Optional[set] = None,
) -> Tuple[List[np.ndarray], List[dict]]:
    """
    Step 1 — Load and preprocess all images in `image_dir`.

    Parameters
    ----------
    image_dir       : directory containing input images
    reconstruction  : SfM state object — image_paths will be populated
    loader          : ImageLoaderBase implementation (default: BasicImageLoader)
    preprocessor    : ImagePreprocessorBase implementation (default: NoOpPreprocessor)
    extensions      : set of allowed file extensions (default: common image types)

    Returns
    -------
    images   : list of BGR uint8 arrays, one per image
    exif_list: list of EXIF dicts, one per image
    """
    if loader is None:
        loader = BasicImageLoader()
    if preprocessor is None:
        preprocessor = NoOpPreprocessor()
    if extensions is None:
        extensions = BasicImageLoader.SUPPORTED_EXTS

    # Collect and sort paths for reproducibility
    all_files = sorted(os.listdir(image_dir))
    paths = [
        os.path.join(image_dir, f)
        for f in all_files
        if os.path.splitext(f)[1].lower() in extensions
    ]

    if not paths:
        raise ValueError(f"No images found in {image_dir!r} with extensions {extensions}")

    images: List[np.ndarray] = []
    exif_list: List[dict] = []

    for path in paths:
        img = loader.load(path)
        img = preprocessor.process(img)
        exif = loader.extract_exif(path)
        images.append(img)
        exif_list.append(exif)

    # Update reconstruction
    reconstruction.image_paths = paths

    print(f"[Step 1] Loaded {len(images)} images from '{image_dir}'")
    for i, (p, e) in enumerate(zip(paths, exif_list)):
        fl = e.get("focal_length_mm")
        sw = e.get("sensor_width_mm")
        sz = f"{e.get('image_width_px')}x{e.get('image_height_px')}"
        print(
            f"  [{i:03d}] {os.path.basename(p):40s}  size={sz} "
            f" focal={fl} mm  sensor_w={sw} mm"
        )

    return images, exif_list
