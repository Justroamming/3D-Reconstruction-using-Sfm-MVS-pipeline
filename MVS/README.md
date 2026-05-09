# MVS Dense Reconstruction Pipeline

Modular Multi-View Stereo pipeline. Takes SfM output (cameras + sparse cloud)
and produces a dense coloured point cloud. Every step has multiple swappable
implementations — CPU and GPU variants where possible.

---

## File Structure

```
mvs/
├── mvs_types.py                   # Shared data structures + GPU detection
├── mvs_step0_input_loader.py      # Load SfM output (COLMAP / NVM / direct)
├── mvs_step1_depth_estimation.py  # Per-image depth maps
├── mvs_step2_depth_refinement.py  # Bilateral / guided filter / hole fill
├── mvs_step3_depth_fusion.py      # Multi-view geometric consistency
├── mvs_step4_point_cloud.py       # Generation + Filtering + Merging + Normals
├── mvs_step8_export.py            # PLY / PCD / LAS / E57 export
└── mvs_pipeline.py                # Orchestrator
```

---

## Quick Start

```python
from mvs_pipeline import MVSPipeline

# From COLMAP sparse directory
pipeline = MVSPipeline(sfm_source="colmap_sparse/", output_dir="output/")
recon = pipeline.run()

# Directly from SfM pipeline result
from mvs_pipeline import MVSPipeline
from mvs_step0_input_loader import SfMPipelineLoader

pipeline = MVSPipeline(
    sfm_source=my_sfm_reconstruction,     # sfm_types.Reconstruction object
    output_dir="output/",
    sfm_loader=SfMPipelineLoader(image_dir="my_images/"),
)
recon = pipeline.run()
```

---

## Swappable Methods Per Step

### Step 0 — SfM Loader
| Class | Source |
|---|---|
| `COLMAPLoader` | COLMAP text sparse directory |
| `SfMPipelineLoader` | Our sfm_types.Reconstruction object |
| `NVMLoader` | VisualSFM .nvm file |
| `ManualLoader` | Raw numpy arrays |

### Step 1 — Depth Estimation
| Class | Backend | Notes |
|---|---|---|
| `PlaneSweepCPU` | CPU NumPy | Default CPU, reliable |
| `PatchMatchCPU` | CPU NumPy | Better quality, slower |
| `SGMDepthEstimator` | CPU OpenCV | Best for stereo pairs |
| `PlaneSweepGPU` | PyTorch CUDA/MPS | 10–50× faster |
| `PatchMatchGPU` | PyTorch CUDA/MPS | Best quality + GPU |
| `MVSNetEstimator` | DL model | Plug-in for MVSNet/IterMVS |

```python
from mvs_step1_depth_estimation import PatchMatchGPU
pipeline = MVSPipeline("sparse/", depth_estimator=PatchMatchGPU(n_iters=5, image_scale=0.5))
```

### Step 2 — Depth Refinement
| Class | Notes |
|---|---|
| `BilateralRefiner` | Edge-preserving smooth (CPU) |
| `ConfidenceThresholdRefiner` | Drop low-confidence pixels |
| `HoleFillRefiner` | Inpaint small holes |
| `MedianRefiner` | Remove salt-and-pepper noise |
| `GuidedFilterRefiner` | Colour-guided depth filter (CPU) |
| `BilateralRefinerGPU` | Bilateral on GPU (PyTorch) |
| `JointBilateralRefinerGPU` | Colour-guided bilateral GPU |
| `ChainRefiner([...])` | Apply multiple in sequence |

### Step 3 — Depth Fusion
| Class | Notes |
|---|---|
| `GeometricConsistencyFusion` | COLMAP-style, CPU |
| `GeometricConsistencyFusionGPU` | Batched on GPU (PyTorch) |
| `TSDFFusion` | Volumetric TSDF fusion (CPU) |
| `VotingFusion` | Per-pixel depth voting (CPU) |

### Step 4 — Point Generation
| Class | Notes |
|---|---|
| `BackProjectionGenerator` | CPU back-projection |
| `BackProjectionGeneratorGPU` | GPU back-projection (PyTorch) |

### Step 5 — Point Filtering
| Class | Notes |
|---|---|
| `StatisticalOutlierFilter` | KNN distance filtering |
| `RadiusOutlierFilter` | Remove isolated points |
| `ConfidenceFilter` | Drop below confidence threshold |
| `ChainPointFilter([...])` | Apply multiple filters |

### Step 6 — Cloud Merging
| Class | Notes |
|---|---|
| `VoxelGridMerger(voxel_size)` | Downsample to uniform density |
| `SimpleMerger` | Concatenate all views |

### Step 7 — Normal Estimation
| Class | Notes |
|---|---|
| `PCLNormalEstimator` | PCA via KNN (CPU) |
| `PCLNormalEstimatorGPU` | Batched SVD (PyTorch) |
| `DepthGradientNormalEstimator` | Smooth existing normals |

### Step 8 — Export
| Class | Format |
|---|---|
| `PLYExporter` | Binary/ASCII PLY with RGB + normals |
| `PCDExporter` | PCL PCD format |
| `LASExporter` | ASPRS LAS (needs laspy) |
| `E57Exporter` | ASTM E57 (needs pye57) |
| `MultiExporter([...])` | Multiple formats at once |

---

## GPU Support

The pipeline auto-detects available hardware at import time:
```
CUDA (NVIDIA) → torch.device("cuda")   [fastest]
MPS  (Apple)  → torch.device("mps")    [fast on M-series]
CPU fallback  → torch.device("cpu")    [always works]
```

Install PyTorch for GPU support:
```bash
# CUDA (NVIDIA):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# MPS (Apple Silicon) — standard PyTorch already includes MPS:
pip install torch torchvision

# CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## Full Custom Example

```python
from mvs_pipeline import MVSPipeline
from mvs_step1_depth_estimation import PatchMatchGPU
from mvs_step2_depth_refinement import ChainRefiner, ConfidenceThresholdRefiner, JointBilateralRefinerGPU, HoleFillRefiner
from mvs_step3_depth_fusion     import GeometricConsistencyFusionGPU
from mvs_step4_point_cloud      import VoxelGridMerger, PCLNormalEstimatorGPU
from mvs_step8_export           import MultiExporter, PLYExporter, PCDExporter

pipeline = MVSPipeline(
    sfm_source      = "colmap_sparse/",
    output_dir      = "dense_output/",
    depth_estimator = PatchMatchGPU(n_iters=5, image_scale=0.5, n_src_views=6),
    depth_refiner   = ChainRefiner([
        ConfidenceThresholdRefiner(min_confidence=0.3),
        JointBilateralRefinerGPU(kernel_size=11),
        HoleFillRefiner(max_hole_size=200),
    ]),
    depth_fusion    = GeometricConsistencyFusionGPU(
        min_consistent_views=3,
        pixel_thresh=2.0,
        rel_depth_thresh=0.04,
    ),
    cloud_merger    = VoxelGridMerger(voxel_size=0.003),
    normal_estimator = PCLNormalEstimatorGPU(k=25),
    exporter        = MultiExporter([PLYExporter(binary=True), PCDExporter()]),
)
recon = pipeline.run()
```

---

## Dependencies

```
# Required
opencv-python >= 4.5
numpy
scipy

# Optional (GPU acceleration)
torch >= 2.0

# Optional (export formats)
laspy      # LAS export
pye57      # E57 export
```
