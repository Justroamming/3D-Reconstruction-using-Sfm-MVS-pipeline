# Custom 3D Reconstruction Pipeline

This repository is a modular Python project for turning a set of images into a 3D reconstruction. It is split into two stages:

1. **SfM (Structure-from-Motion)** builds a sparse scene model and estimates camera poses.
2. **MVS (Multi-View Stereo)** uses that sparse model to produce a dense point cloud.

The code is designed so you can run the default pipeline end to end, or swap in different implementations at almost every step.

## Overview

The SfM stage loads images, extracts and matches features, verifies geometry, estimates camera intrinsics, initializes the reconstruction, registers additional cameras, triangulates points, performs bundle adjustment, removes outliers, and exports a sparse model.

The MVS stage takes the sparse reconstruction and turns it into a dense result by estimating depth maps, refining them, fusing them across views, generating points, filtering and merging the cloud, estimating normals, and exporting the final dense output.

In short:

`images -> SfM sparse reconstruction -> MVS depth estimation -> dense point cloud`

## Repository Structure

```text
SFM/
  pipeline.py                 # SfM pipeline orchestrator
  step1_image_loader.py       # Load and preprocess images
  step2_feature_extraction.py # Detect and describe keypoints
  step3_feature_matching.py   # Match descriptors between images
  step4_geometric_verification.py
  step5_camera_intrinsics.py  # Estimate camera intrinsics
  step6_initial_reconstruction.py
  step7_camera_registration.py
  step8_triangulation.py
  step9_bundle_adjustment.py
  step10_outlier_filtering.py
  step11_export.py            # Export sparse reconstruction

MVS/
  mvs_pipeline.py             # MVS pipeline orchestrator
  mvs_step0_input_loader.py   # Read SfM output into the dense pipeline
  mvs_step1_depth_estimation.py
  mvs_step2_depth_refinement.py
  mvs_step3_depth_fusion.py
  mvs_step4_point_cloud.py
  mvs_step8_export.py         # Export dense reconstruction
```

## How It Fits Together

The usual workflow is:

1. Run the SfM pipeline on a set of images.
2. Save or pass along the sparse reconstruction it produces.
3. Feed that result into the MVS pipeline.
4. Export the final dense point cloud.

## Quick Example

```python
from SFM.pipeline import SfMPipeline
from MVS.mvs_pipeline import MVSPipeline

sfm = SfMPipeline(image_dir="path/to/images", output_dir="output/")
sparse_reconstruction = sfm.run()

mvs = MVSPipeline(
    sfm_source=sparse_reconstruction,
    output_dir="mvs_output/",
)
dense_reconstruction = mvs.run()
```

Each pipeline step can be customized by passing a different strategy object into the constructor.

## Requirements

The project is written in Python and uses common computer vision and scientific computing packages such as:

- `numpy`
- `scipy`
- `opencv-python`

Some optional features may also use:

- `torch` for GPU acceleration
- `laspy` for LAS export
- `pye57` for E57 export

## Summary

This project is a modular 3D reconstruction toolkit. It is built for experimentation, so the default settings should work as a starting point while still allowing you to swap in different feature extractors, matchers, estimators, refiners, fusion methods, and exporters.