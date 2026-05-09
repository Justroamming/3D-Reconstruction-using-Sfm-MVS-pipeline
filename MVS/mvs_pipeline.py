"""
mvs_pipeline.py
---------------
MVS Pipeline Orchestrator

Wires Steps 0–8 into a single run() call.
Every step can be overridden by passing a different strategy object.

Usage — minimal (all defaults):
    from mvs_pipeline import MVSPipeline
    pipeline = MVSPipeline(sfm_source="colmap_sparse/", output_dir="mvs_output/")
    reconstruction = pipeline.run()

Usage — custom strategies:
    from mvs_pipeline import MVSPipeline
    from mvs_step1_depth_estimation   import PatchMatchGPU
    from mvs_step2_depth_refinement   import JointBilateralRefinerGPU
    from mvs_step3_depth_fusion       import GeometricConsistencyFusionGPU
    from mvs_step4_point_cloud        import VoxelGridMerger, PCLNormalEstimatorGPU
    from mvs_step8_export             import MultiExporter, PLYExporter, PCDExporter

    pipeline = MVSPipeline(
        sfm_source="colmap_sparse/",
        output_dir="mvs_output/",
        depth_estimator   = PatchMatchGPU(n_iters=5, image_scale=0.5),
        depth_fusion      = GeometricConsistencyFusionGPU(min_consistent_views=3),
        cloud_merger      = VoxelGridMerger(voxel_size=0.003),
        normal_estimator  = PCLNormalEstimatorGPU(k=25),
        exporter          = MultiExporter([PLYExporter(), PCDExporter()]),
    )
    recon = pipeline.run()
"""

from __future__ import annotations
import time
import os
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import cv2

from mvs_types import MVSReconstruction, ACCELERATOR, detect_accelerator

from mvs_step0_input_loader    import SfMLoaderBase,        load_sfm_result
from mvs_step1_depth_estimation import DepthEstimatorBase,  estimate_depth_maps
from mvs_step2_depth_refinement import DepthRefinerBase,    refine_depth_maps
from mvs_step3_depth_fusion     import DepthFusionBase,     fuse_depth_maps
from mvs_step4_point_cloud      import (
    PointGeneratorBase, generate_point_clouds,
    PointFilterBase,    filter_point_clouds,
    CloudMergerBase,    merge_point_clouds,
    NormalEstimatorBase, estimate_normals,
)
from mvs_step8_export           import DenseExporterBase,   export_dense_cloud


@dataclass
class MVSPipelineConfig:
    """All tuneable parameters and strategy overrides for the MVS pipeline."""

    # --- I/O ---
    sfm_source:   object = None    # COLMAP dir path, Reconstruction object, or NVM path
    output_dir:   str    = "/your_dense_directory/"
    save_raw_depth_maps: bool = True   # Save raw depth maps after estimation (Step 1)
    save_depth_maps: bool = True   # Save refined depth maps as images (Step 2)
    save_fused_depth_maps: bool = True  # Save fused depth maps after fusion (Step 3)

    # --- Step 0: SfM loading ---
    sfm_loader:   Optional[SfMLoaderBase] = None

    # --- Step 1: Depth estimation ---
    depth_estimator:  Optional[DepthEstimatorBase] = None
    n_src_views:      int = 4
    depth_image_ids:  Optional[list] = None   # None = all cameras

    # --- Step 2: Depth refinement ---
    depth_refiner:    Optional[DepthRefinerBase] = None

    # --- Step 3: Depth fusion ---
    depth_fusion:     Optional[DepthFusionBase] = None

    # --- Step 4: Point generation ---
    point_generator:  Optional[PointGeneratorBase] = None
    use_fused_depth:  bool = True

    # --- Step 5: Point filtering ---
    point_filter:     Optional[PointFilterBase] = None

    # --- Step 6: Cloud merging ---
    cloud_merger:     Optional[CloudMergerBase] = None

    # --- Step 7: Normal estimation ---
    normal_estimator: Optional[NormalEstimatorBase] = None
    skip_normals:     bool = False

    # --- Step 8: Export ---
    exporter:         Optional[DenseExporterBase] = None

# ============================================================
#  DEPTH MAP VISUALIZATION
# ============================================================

def _save_depth_maps(recon: MVSReconstruction, output_dir: str, cameras: dict):
    """Save refined depth maps as visualizable images (PNG + colormap)."""
    depth_output_dir = os.path.join(output_dir, "depth_maps_refined")
    confidence_output_dir = os.path.join(output_dir, "confidence_maps_refined")
    
    os.makedirs(depth_output_dir, exist_ok=True)
    if any(dm.confidence is not None for dm in recon.depth_maps.values()):
        os.makedirs(confidence_output_dir, exist_ok=True)
    
    saved_count = 0
    
    for image_id, depth_map in recon.depth_maps.items():
        # Get image name for filename
        camera = cameras.get(image_id)
        image_name = f"img_{image_id:03d}"
        if camera:
            # Try to extract a cleaner name from the image path
            img_path = camera.image_path
            if img_path:
                image_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # ─ Save depth map as 16-bit PNG (normalized) ─
        depth = depth_map.depth
        valid_mask = depth > 0
        
        if valid_mask.any():
            depth_min = depth[valid_mask].min()
            depth_max = depth[valid_mask].max()
            # Normalize to 0-65535 (16-bit range)
            depth_norm = np.zeros_like(depth, dtype=np.uint16)
            if depth_max > depth_min:
                depth_norm[valid_mask] = ((depth[valid_mask] - depth_min) / 
                                          (depth_max - depth_min) * 65535).astype(np.uint16)
            
            depth_png_path = os.path.join(depth_output_dir, f"{image_name}_depth.png")
            cv2.imwrite(depth_png_path, depth_norm)
            
            # ─ Save depth map as colorized visualization (8-bit PNG with colormap) ─
            depth_vis = np.zeros_like(depth, dtype=np.uint8)
            if depth_max > depth_min:
                depth_vis[valid_mask] = ((depth[valid_mask] - depth_min) / 
                                         (depth_max - depth_min) * 255).astype(np.uint8)
            
            depth_vis_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
            depth_vis_color[~valid_mask] = [0, 0, 0]  # Black for invalid pixels
            
            depth_vis_path = os.path.join(depth_output_dir, f"{image_name}_depth_color.png")
            cv2.imwrite(depth_vis_path, depth_vis_color)
            
            # ─ Save confidence map if available ─
            if depth_map.confidence is not None:
                conf = depth_map.confidence
                conf_norm = (np.clip(conf, 0, 1) * 255).astype(np.uint8)
                conf_color = cv2.applyColorMap(conf_norm, cv2.COLORMAP_JET)
                conf_path = os.path.join(confidence_output_dir, f"{image_name}_confidence.png")
                cv2.imwrite(conf_path, conf_color)
            
            saved_count += 1
    
    if saved_count > 0:
        print(f"[Step 2+] Saved {saved_count} depth maps to: {depth_output_dir}")
        print(f"           - depth.png (16-bit grayscale)")
        print(f"           - depth_color.png (colorized visualization)")
        if any(dm.confidence is not None for dm in recon.depth_maps.values()):
            print(f"           - confidence.png (confidence maps)")


def _save_raw_depth_maps(recon: MVSReconstruction, output_dir: str, cameras: dict):
    """Save raw (estimated) depth maps as visualizable images."""
    depth_output_dir = os.path.join(output_dir, "depth_maps_raw")
    confidence_output_dir = os.path.join(output_dir, "confidence_maps_raw")
    
    os.makedirs(depth_output_dir, exist_ok=True)
    if any(dm.confidence is not None for dm in recon.depth_maps.values()):
        os.makedirs(confidence_output_dir, exist_ok=True)
    
    saved_count = 0
    
    for image_id, depth_map in recon.depth_maps.items():
        camera = cameras.get(image_id)
        image_name = f"img_{image_id:03d}"
        if camera:
            img_path = camera.image_path
            if img_path:
                image_name = os.path.splitext(os.path.basename(img_path))[0]
        
        depth = depth_map.depth
        valid_mask = depth > 0
        
        if valid_mask.any():
            depth_min = depth[valid_mask].min()
            depth_max = depth[valid_mask].max()
            
            # 16-bit depth
            depth_norm = np.zeros_like(depth, dtype=np.uint16)
            if depth_max > depth_min:
                depth_norm[valid_mask] = ((depth[valid_mask] - depth_min) / 
                                          (depth_max - depth_min) * 65535).astype(np.uint16)
            
            depth_png_path = os.path.join(depth_output_dir, f"{image_name}_depth.png")
            cv2.imwrite(depth_png_path, depth_norm)
            
            # Colorized depth
            depth_vis = np.zeros_like(depth, dtype=np.uint8)
            if depth_max > depth_min:
                depth_vis[valid_mask] = ((depth[valid_mask] - depth_min) / 
                                         (depth_max - depth_min) * 255).astype(np.uint8)
            
            depth_vis_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
            depth_vis_color[~valid_mask] = [0, 0, 0]
            
            depth_vis_path = os.path.join(depth_output_dir, f"{image_name}_depth_color.png")
            cv2.imwrite(depth_vis_path, depth_vis_color)
            
            # Confidence map if available
            if depth_map.confidence is not None:
                conf = depth_map.confidence
                conf_norm = (np.clip(conf, 0, 1) * 255).astype(np.uint8)
                conf_color = cv2.applyColorMap(conf_norm, cv2.COLORMAP_JET)
                conf_path = os.path.join(confidence_output_dir, f"{image_name}_confidence.png")
                cv2.imwrite(conf_path, conf_color)
            
            saved_count += 1
    
    if saved_count > 0:
        print(f"[Step 1+] Saved {saved_count} raw depth maps to: {depth_output_dir}")
        print(f"           - depth.png (16-bit grayscale)")
        print(f"           - depth_color.png (colorized visualization)")
        if any(dm.confidence is not None for dm in recon.depth_maps.values()):
            print(f"           - confidence.png (confidence maps)")


def _save_fused_depth_maps(recon: MVSReconstruction, output_dir: str, cameras: dict):
    """Save fused depth maps as visualizable images."""
    depth_output_dir = os.path.join(output_dir, "depth_maps_fused")
    confidence_output_dir = os.path.join(output_dir, "confidence_maps_fused")
    
    os.makedirs(depth_output_dir, exist_ok=True)
    if any(dm.confidence is not None for dm in recon.fused_depth.values()):
        os.makedirs(confidence_output_dir, exist_ok=True)
    
    saved_count = 0
    
    for image_id, depth_map in recon.fused_depth.items():
        camera = cameras.get(image_id)
        image_name = f"img_{image_id:03d}"
        if camera:
            img_path = camera.image_path
            if img_path:
                image_name = os.path.splitext(os.path.basename(img_path))[0]
        
        depth = depth_map.depth
        valid_mask = depth > 0
        
        if valid_mask.any():
            depth_min = depth[valid_mask].min()
            depth_max = depth[valid_mask].max()
            
            # 16-bit depth
            depth_norm = np.zeros_like(depth, dtype=np.uint16)
            if depth_max > depth_min:
                depth_norm[valid_mask] = ((depth[valid_mask] - depth_min) / 
                                          (depth_max - depth_min) * 65535).astype(np.uint16)
            
            depth_png_path = os.path.join(depth_output_dir, f"{image_name}_depth_fused.png")
            cv2.imwrite(depth_png_path, depth_norm)
            
            # Colorized depth
            depth_vis = np.zeros_like(depth, dtype=np.uint8)
            if depth_max > depth_min:
                depth_vis[valid_mask] = ((depth[valid_mask] - depth_min) / 
                                         (depth_max - depth_min) * 255).astype(np.uint8)
            
            depth_vis_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
            depth_vis_color[~valid_mask] = [0, 0, 0]
            
            depth_vis_path = os.path.join(depth_output_dir, f"{image_name}_depth_fused_color.png")
            cv2.imwrite(depth_vis_path, depth_vis_color)
            
            # Confidence/consistency map
            if depth_map.confidence is not None:
                conf = depth_map.confidence
                conf_norm = (np.clip(conf, 0, 1) * 255).astype(np.uint8)
                conf_color = cv2.applyColorMap(conf_norm, cv2.COLORMAP_JET)
                conf_path = os.path.join(confidence_output_dir, f"{image_name}_consistency.png")
                cv2.imwrite(conf_path, conf_color)
            
            saved_count += 1
    
    if saved_count > 0:
        print(f"[Step 3+] Saved {saved_count} fused depth maps to: {depth_output_dir}")
        print(f"           - depth_fused.png (16-bit grayscale)")
        print(f"           - depth_fused_color.png (colorized visualization)")
        if any(dm.confidence is not None for dm in recon.fused_depth.values()):
            print(f"           - consistency.png (multi-view consistency confidence)")

def _analyze_depth_maps(recon: MVSReconstruction, stage: str = "Step 1") -> None:
    """Analyze and validate depth map statistics."""
    if not recon.depth_maps:
        print(f"[{stage}] No depth maps to analyze")
        return
    
    print(f"\n[{stage}] Depth map analysis ({len(recon.depth_maps)} maps):")
    
    min_depth_global = float('inf')
    max_depth_global = 0
    total_valid_pixels = 0
    total_pixels = 0
    depth_std_sum = 0
    
    for img_id in list(recon.depth_maps.keys())[:3]:  # Sample first 3
        dm = recon.depth_maps[img_id]
        depth = dm.depth
        valid = depth > 0
        
        total_pixels += depth.size
        total_valid_pixels += valid.sum()
        
        if valid.any():
            min_d = depth[valid].min()
            max_d = depth[valid].max()
            mean_d = depth[valid].mean()
            std_d = depth[valid].std()
            
            min_depth_global = min(min_depth_global, min_d)
            max_depth_global = max(max_depth_global, max_d)
            depth_std_sum += std_d
            
            valid_pct = 100.0 * valid.sum() / valid.size
            print(f"  [img {img_id:03d}] res={depth.shape[1]}×{depth.shape[0]} | "
                  f"valid={valid_pct:.1f}% | depth [{min_d:.2f}, {max_d:.2f}] m | "
                  f"μ={mean_d:.2f} σ={std_d:.2f}")
        else:
            print(f"  [img {img_id:03d}] res={depth.shape[1]}×{depth.shape[0]} | "
                  f"valid=0% (all zeros)")
    
    valid_pct_total = 100.0 * total_valid_pixels / total_pixels if total_pixels > 0 else 0
    print(f"  Overall: {valid_pct_total:.1f}% valid | depth range [{min_depth_global:.2f}, "
          f"{max_depth_global:.2f}] m")
    
    # Check for unreasonable values
    if max_depth_global > 10000:
        print(f"  ⚠️  WARNING: Very large depth values detected (max={max_depth_global:.2f}m)")
    if min_depth_global < 0.01:
        print(f"  ⚠️  WARNING: Very small depth values detected (min={min_depth_global:.4f}m)")


def _analyze_point_cloud(cloud, stage: str) -> None:
    """Analyze point cloud geometry."""
    if len(cloud) == 0:
        print(f"[{stage}] Point cloud is empty")
        return
    
    pts = cloud.points
    min_pt = pts.min(axis=0)
    max_pt = pts.max(axis=0)
    center = pts.mean(axis=0)
    
    # Compute statistics
    distances_to_center = np.linalg.norm(pts - center, axis=1)
    mean_dist = distances_to_center.mean()
    median_dist = np.median(distances_to_center)
    std_dist = distances_to_center.std()
    
    # Check for outliers (points > 3σ from center)
    outliers = distances_to_center > (mean_dist + 3*std_dist)
    outlier_count = outliers.sum()
    
    print(f"[{stage}] {len(cloud)} points | "
          f"bbox=[{min_pt[0]:.2f}, {max_pt[0]:.2f}] x "
          f"[{min_pt[1]:.2f}, {max_pt[1]:.2f}] x "
          f"[{min_pt[2]:.2f}, {max_pt[2]:.2f}] | "
          f"center_dist μ={mean_dist:.3f} σ={std_dist:.3f} | "
          f"outliers={outlier_count} ({100*outlier_count/len(cloud):.1f}%)")
    
    if outlier_count > len(cloud) * 0.1:
        print(f"  ⚠️  WARNING: {outlier_count} outlier points detected (>10% of cloud)")

class MVSPipeline:
    """
    Multi-View Stereo dense reconstruction pipeline.

    All strategy objects are optional; each step falls back to a sensible
    hardware-aware default (GPU if available, CPU otherwise).
    """

    def __init__(self, sfm_source, output_dir: str = "mvs_output/", **kwargs):
        self.config = MVSPipelineConfig(
            sfm_source=sfm_source,
            output_dir=output_dir,
            **kwargs,
        )
        self.reconstruction = MVSReconstruction(accelerator=detect_accelerator())

    def run(self) -> MVSReconstruction:
        """Execute the full MVS pipeline."""
        cfg   = self.config
        recon = self.reconstruction
        accel = recon.accelerator

        t0 = time.time()
        print("=" * 60)
        print("  MVS Dense Reconstruction Pipeline")
        print(f"  GPU: cuda={accel.has_cuda}  mps={accel.has_mps}  "
              f"opencl={accel.has_opencl}  device={accel.torch_device}")
        print("=" * 60)

        # ── Step 0: Load SfM result ──────────────────────────────
        load_sfm_result(cfg.sfm_source, recon, loader=cfg.sfm_loader)

        if not recon.cameras:
            print("ERROR: No cameras loaded. Aborting.")
            return recon

        # ── Step 1: Depth map estimation ─────────────────────────
        estimate_depth_maps(
            recon,
            estimator=cfg.depth_estimator,
            n_src_views=cfg.n_src_views,
            image_ids=cfg.depth_image_ids,
        )

        if not recon.depth_maps:
            print("ERROR: No depth maps produced. Aborting.")
            return recon
        
        # ── Analyze raw depth maps ───────────────────────────────
        _analyze_depth_maps(recon, "Step 1")

        # ── Save raw depth maps ──────────────────────────────────
        if cfg.save_raw_depth_maps and recon.depth_maps:
            _save_raw_depth_maps(recon, cfg.output_dir, recon.cameras)

        # ── Step 2: Depth map refinement ─────────────────────────
        refine_depth_maps(recon, refiner=cfg.depth_refiner)
         # ── Analyze refined depth maps ───────────────────────────
        _analyze_depth_maps(recon, "Step 2")

        # ── Save refined depth maps ──────────────────────────────
        if cfg.save_depth_maps and recon.depth_maps:
            _save_depth_maps(recon, cfg.output_dir, recon.cameras)

        # ── Step 3: Depth map fusion ──────────────────────────────
        if cfg.use_fused_depth:
            fuse_depth_maps(recon, fusion=cfg.depth_fusion)
        else:
            print("[Step 3] Skipped depth fusion (use_fused_depth=False)")
        # ── Save fused depth maps ────────────────────────────────
        if cfg.save_fused_depth_maps and recon.fused_depth:
            _save_fused_depth_maps(recon, cfg.output_dir, recon.cameras)

        # ── Step 4: Point cloud generation ───────────────────────
        per_view = generate_point_clouds(
            recon,
            generator=cfg.point_generator,
            use_fused=cfg.use_fused_depth,
        )
        # ── Analyze per-view point clouds (sample) ──────────────
        sample_cloud = next(iter(per_view.values())) if per_view else None
        if sample_cloud:
            _analyze_point_cloud(sample_cloud, "Step 4 (before filtering)")

        # ── Step 5: Point cloud filtering ────────────────────────
        per_view = filter_point_clouds(per_view, point_filter=cfg.point_filter)

        sample_cloud = next(iter(per_view.values())) if per_view else None
        if sample_cloud:
            _analyze_point_cloud(sample_cloud, "Step 5 (after filtering)")

        # ── Step 6: Merge & downsample ───────────────────────────
        merge_point_clouds(per_view, recon, merger=cfg.cloud_merger)
        _analyze_point_cloud(recon.dense_cloud, "Step 6 (after merge)")

        # ── Step 7: Normal estimation ─────────────────────────────
        if not cfg.skip_normals:
            estimate_normals(recon, estimator=cfg.normal_estimator)

        # ── Step 8: Export ────────────────────────────────────────
        output_path = export_dense_cloud(recon, cfg.output_dir, exporter=cfg.exporter)

        elapsed = time.time() - t0
        n_pts = len(recon.dense_cloud)
        print(f"\n{'='*60}")
        print(f"  DONE  {len(recon.cameras)} cameras | {n_pts} dense points | "
              f"[{elapsed:.1f}s]")
        print(f"  Output: {output_path}")
        print(f"{'='*60}")

        return recon


# ============================================================
#  QUICK-START EXAMPLE
# ============================================================

if __name__ == "__main__":
    import sys
    from mvs_step1_depth_estimation import PlaneSweepCPU,PlaneSweepGPU,PatchMatchGPU
    from mvs_step2_depth_refinement import ChainRefiner,BilateralRefinerGPU
    from mvs_step3_depth_fusion import GeometricConsistencyFusion,GeometricConsistencyFusionGPU
    from mvs_step4_point_cloud import ChainPointFilter, StatisticalOutlierFilter, VoxelGridMerger, PCLNormalEstimator, BackProjectionGenerator
    from mvs_step8_export import PLYExporter

    sfm_source = sys.argv[1] if len(sys.argv) > 1 else "/your_sparse_directory"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "/your_dense_directory/"
    run_mode = (sys.argv[3] if len(sys.argv) > 3 else "fast").lower()

    if run_mode == "quality":
        # Slower but denser reconstruction.
        pipeline = MVSPipeline(
            sfm_source=sfm_source,
            output_dir=output_dir,
            n_src_views=6,
            depth_estimator=PlaneSweepGPU(n_planes=96, patch_half=3, n_src_views=6, image_scale=1.0),
            depth_fusion=GeometricConsistencyFusionGPU(
                min_consistent_views=1,
                pixel_thresh=4.0,
                rel_depth_thresh=0.12,
            ),
            point_filter=ChainPointFilter([
                StatisticalOutlierFilter(
                    k=12,
                    std_ratio=3.0,
                    use_gpu=ACCELERATOR.has_cuda or ACCELERATOR.has_mps,
                ),
            ]),
            cloud_merger=VoxelGridMerger(voxel_size=0.003),
            use_fused_depth=True,
        )
    else:
        # Fast preview preset for CPU.
        pipeline = MVSPipeline(
            sfm_source=sfm_source,
            output_dir=output_dir,
            n_src_views=3,
            depth_estimator=PlaneSweepGPU(n_planes=24, patch_half=2, n_src_views=3, image_scale=0.5),
            depth_refiner = BilateralRefinerGPU(),
            depth_fusion=GeometricConsistencyFusionGPU(
                min_consistent_views=2,
                pixel_thresh=3.5,
                rel_depth_thresh=0.12,
            ),
            point_filter=ChainPointFilter([
                StatisticalOutlierFilter(),
            ]),
            cloud_merger=VoxelGridMerger(voxel_size=0.01),
            use_fused_depth=True,
        )
        

    print(f"Run mode: {run_mode} (pass 'quality' as 3rd arg for denser output)")
    recon = pipeline.run()

    # --- Example: custom strategies ---
    # from mvs_step1_depth_estimation import PlaneSweepGPU
    # from mvs_step3_depth_fusion     import GeometricConsistencyFusionGPU
    # from mvs_step4_point_cloud      import VoxelGridMerger
    # from mvs_step8_export           import MultiExporter, PLYExporter, PCDExporter
    #
    # pipeline = MVSPipeline(
    #     sfm_source    = sfm_source,
    #     output_dir    = output_dir,
    #     depth_estimator = PlaneSweepGPU(n_planes=128, image_scale=0.5),
    #     depth_fusion  = GeometricConsistencyFusionGPU(min_consistent_views=3),
    #     cloud_merger  = VoxelGridMerger(voxel_size=0.003),
    #     exporter      = MultiExporter([PLYExporter(), PCDExporter()]),
    # )
    # recon = pipeline.run()

    # --- Example: use SfMPipelineLoader for Step 0 ---
    #from SFM.pipeline             import SFMPipeline
    #from mvs_step0_input_loader  import SfMPipelineLoader
    #
    #sfm_recon = SFMPipeline(image_dir="House3d").run()
    #
    #pipeline = MVSPipeline(
    #    sfm_source = sfm_recon,
    #    output_dir = output_dir,
    #    sfm_loader = SfMPipelineLoader(image_dir="House3d"),
    #)
    #recon = pipeline.run()
