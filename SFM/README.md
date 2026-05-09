# Incremental SfM Pipeline

A modular, plug-and-play Structure-from-Motion pipeline in Python.
Every step exposes a clean base class so you can swap in a different
algorithm without touching any other part of the code.

---

## File Structure

```
sfm/
├── sfm_types.py                   # Shared data structures (Reconstruction, Camera, …)
├── step1_image_loader.py          # Image loading & preprocessing
├── step2_feature_extraction.py    # Keypoint detection + description
├── step3_feature_matching.py      # Descriptor matching + pair selection
├── step4_geometric_verification.py# RANSAC-based inlier filtering
├── step5_camera_intrinsics.py     # Intrinsic camera estimation
├── step6_initial_reconstruction.py# Seed pair + two-view init
├── step7_camera_registration.py   # PnP camera registration (next-best-view)
├── step8_triangulation.py         # New 3-D point triangulation
├── step9_bundle_adjustment.py     # Joint pose + point optimisation
├── step10_outlier_filtering.py    # Point cloud cleaning
├── step11_export.py               # PLY / COLMAP / NVM export
└── pipeline.py                    # Orchestrator — wires all steps together
```

---

## Quick Start

```python
from pipeline import SfMPipeline

pipeline = SfMPipeline(image_dir="my_images/", output_dir="output/")
reconstruction = pipeline.run()
```

---

## Swapping Methods

Every step has a base class and multiple concrete implementations.
Pass any implementation to `SfMPipeline(...)` as a keyword argument.

### Feature Extractor (Step 2)
| Class | Descriptor | Notes |
|---|---|---|
| `SIFTExtractor` | float-128 | Best quality, default |
| `ORBExtractor` | binary-256 | Fastest |
| `AKAZEExtractor` | binary | Good balance |
| `SuperPointExtractor` | float-256 | Deep learning, bring own weights |

```python
from step2_feature_extraction import ORBExtractor
pipeline = SfMPipeline("imgs/", feature_extractor=ORBExtractor(n_features=5000))
```

### Pair Selector (Step 3)
| Class | Notes |
|---|---|
| `ExhaustivePairSelector` | All pairs, default |
| `SequentialPairSelector(window=5)` | Ordered/video capture |
| `VocabTreePairSelector` | Scalable, needs retrieval model |

### Matcher (Step 3)
| Class | Notes |
|---|---|
| `BFMatcher` | Brute-force, default |
| `FLANNMatcher` | Faster for float descriptors |
| `SuperGlueMatcher` | Deep learning, bring own model |

### Geometric Verifier (Step 4)
| Class | Notes |
|---|---|
| `FundamentalRANSAC` | Default, no intrinsics needed |
| `HomographyRANSAC` | Planar scenes |
| `EssentialRANSAC(K)` | Needs intrinsics, most accurate |
| `MAGSACVerifier` | MAGSAC++, needs opencv-contrib |

### Intrinsics Estimator (Step 5)
| Class | Notes |
|---|---|
| `EXIFIntrinsicsEstimator` | Uses EXIF focal length, default |
| `CalibrationFileEstimator(path)` | Load JSON calibration |
| `PrincipalPointEstimator` | Heuristic, no EXIF needed |
| `FixedIntrinsicsEstimator(K)` | Known K |

### Bundle Adjuster (Step 9)
| Class | Notes |
|---|---|
| `SciPyBundleAdjuster` | Pure Python LM, default |
| `LocalBundleAdjuster(window=5)` | Faster for large scenes |
| `StubBundleAdjuster` | Disable BA (debugging) |

### Exporter (Step 11)
| Class | Output |
|---|---|
| `PLYExporter` | `sparse_cloud.ply` + `cameras.ply` |
| `COLMAPExporter` | `sparse/{cameras,images,points3D}.txt` |
| `NVMExporter` | `reconstruction.nvm` |

---

## Full Custom Example

```python
from pipeline import SfMPipeline
from step1_image_loader       import ResizePreprocessor
from step2_feature_extraction import SIFTExtractor
from step3_feature_matching   import SequentialPairSelector, FLANNMatcher
from step4_geometric_verification import EssentialRANSAC
from step5_camera_intrinsics  import CalibrationFileEstimator
from step9_bundle_adjustment  import LocalBundleAdjuster
from step11_export            import COLMAPExporter
import numpy as np

pipeline = SfMPipeline(
    image_dir  = "my_images/",
    output_dir = "output/",
    preprocessor       = ResizePreprocessor(max_side=1600),
    feature_extractor  = SIFTExtractor(n_features=10000),
    pair_selector      = SequentialPairSelector(window=5),
    matcher            = FLANNMatcher(ratio_thresh=0.75),
    intrinsics_estimator = CalibrationFileEstimator("calib.json"),
    bundle_adjuster    = LocalBundleAdjuster(window=5),
    ba_every_n_images  = 3,
    exporter           = COLMAPExporter(),
)
reconstruction = pipeline.run()
```

---

## Data Flow

```
images/
  └─ Step 1 → images[], exif_list[]
       └─ Step 2 → Reconstruction.keypoints
            └─ Step 3 → Reconstruction.image_matches
                 └─ Step 4 → Reconstruction.verified_matches
                      └─ Step 5 → Reconstruction.cameras
                           └─ Step 6 → registered_images[0,1] + initial points3d
                                └─ loop:
                                     Step 7 → registered_images[new]
                                     Step 8 → points3d (grow)
                                     Step 9 → optimise in-place
                                     Step 10 → remove outliers
                                └─ Step 11 → PLY / COLMAP / NVM files
```

---

## Dependencies

```
opencv-python >= 4.5
numpy
scipy          # for SciPyBundleAdjuster
Pillow         # for EXIF extraction (optional but recommended)
```

Install:
```bash
pip install opencv-python numpy scipy Pillow
```
