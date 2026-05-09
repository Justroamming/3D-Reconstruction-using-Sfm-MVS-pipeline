"""
pipeline.py
-----------
Incremental SfM Pipeline Orchestrator

Wires together all steps into a single run() call.
Every step can be overridden by passing a different strategy object.

Usage
-----
    from pipeline import SfMPipeline
    from step2_feature_extraction import ORBExtractor
    from step3_feature_matching import SequentialPairSelector, FLANNMatcher

    pipeline = SfMPipeline(
        image_dir="my_images/",
        output_dir="outputs/",
        feature_extractor=ORBExtractor(n_features=5000),
        pair_selector=SequentialPairSelector(window=3),
    )
    reconstruction = pipeline.run()
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from sfm_types import Reconstruction

# Step imports
from step1_image_loader      import ImageLoaderBase, ImagePreprocessorBase, load_images
from step2_feature_extraction import FeatureExtractorBase, extract_features
from step3_feature_matching   import PairSelectorBase, DescriptorMatcherBase, match_features
from step4_geometric_verification import GeometricVerifierBase, verify_matches
from step5_camera_intrinsics  import EXIFIntrinsicsEstimator, IntrinsicsEstimatorBase, estimate_intrinsics
from step6_initial_reconstruction import (
    SeedPairSelectorBase, InitialPoseRecovererBase, initialize_reconstruction
)
from step7_camera_registration import (
    NextViewSelectorBase, PnPSolverBase, register_next_image
)
from step8_triangulation      import TriangulatorBase, triangulate_new_points
from step9_bundle_adjustment   import BundleAdjusterBase, run_bundle_adjustment
from step10_outlier_filtering  import PointFilterBase, filter_outliers
from step11_export             import ExporterBase, export_reconstruction


@dataclass
class SfMPipelineConfig:
    """
    All tuneable parameters and strategy overrides for the pipeline.
    Set any field to None to use the step's built-in default.
    """

    # --- I/O ---
    image_dir:  str = "your_images_directory/"
    output_dir: str = "your_sparse_directory/"

    # --- Step 1: Image loading ---
    loader:       Optional[ImageLoaderBase]       = None
    preprocessor: Optional[ImagePreprocessorBase] = None

    # --- Step 2: Feature extraction ---
    feature_extractor: Optional[FeatureExtractorBase] = None

    # --- Step 3: Feature matching ---
    pair_selector: Optional[PairSelectorBase]        = None
    matcher:       Optional[DescriptorMatcherBase]   = None
    min_matches:   int = 20

    # --- Step 4: Geometric verification ---
    verifier:     Optional[GeometricVerifierBase] = None
    min_inliers:  int = 15

    # --- Step 5: Intrinsics ---
    intrinsics_estimator: Optional[IntrinsicsEstimatorBase] = None
    shared_camera:        bool = False

    # --- Step 6: Initial reconstruction ---
    seed_selector:   Optional[SeedPairSelectorBase]     = None
    pose_recoverer:  Optional[InitialPoseRecovererBase] = None
    max_reproj_init: float = 1.0
    min_angle_init:  float = 0.5

    # --- Step 7: Camera registration ---
    view_selector:    Optional[NextViewSelectorBase] = None
    pnp_solver:       Optional[PnPSolverBase]        = None
    min_pnp_inliers:  int = 15

    # --- Step 8: Triangulation ---
    triangulator:     Optional[TriangulatorBase] = None
    max_reproj_tri:   float = 1.0
    min_angle_tri:    float = 0.5

    # --- Step 9: Bundle adjustment ---
    bundle_adjuster:    Optional[BundleAdjusterBase] = None
    ba_every_n_images:  int = 5    # run full BA every N new images

    # --- Step 10: Outlier filtering ---
    point_filter:       Optional[PointFilterBase] = None

    # --- Step 11: Export ---
    exporter: Optional[ExporterBase] = None


class SfMPipeline:
    """
    Incremental Structure-from-Motion pipeline.

    All strategy objects are optional; each step falls back to its
    sensible default when None is passed.
    """

    def __init__(self, image_dir: str, output_dir: str = "your_sparse_directory/", **kwargs):
        self.config = SfMPipelineConfig(
            image_dir=image_dir,
            output_dir=output_dir,
            **kwargs,
        )
        self.reconstruction = Reconstruction()

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def run(self) -> Reconstruction:
        """Execute the full incremental SfM pipeline."""
        cfg = self.config
        recon = self.reconstruction

        t0 = time.time()
        print("=" * 60)
        print("  Incremental SfM Pipeline")
        print("=" * 60)

        # ── Step 1: Load images ─────────────────────────────────────
        images, exif_list = load_images(
            cfg.image_dir, recon,
            loader=cfg.loader,
            preprocessor=cfg.preprocessor,
        )

        # ── Step 2: Feature extraction ───────────────────────────────
        extract_features(images, recon, extractor=cfg.feature_extractor)

        # ── Step 3: Feature matching ─────────────────────────────────
        match_features(
            recon,
            pair_selector=cfg.pair_selector,
            matcher=cfg.matcher,
            min_matches=cfg.min_matches,
        )

        # ── Step 4: Geometric verification ───────────────────────────
        verify_matches(recon, verifier=cfg.verifier, min_inliers=cfg.min_inliers)

        if not recon.verified_matches:
            print("ERROR: No verified matches found. Aborting.")
            return recon

        # ── Step 5: Camera intrinsics ─────────────────────────────────
        estimate_intrinsics(
            images, exif_list, recon,
            estimator=cfg.intrinsics_estimator,
            shared_camera=cfg.shared_camera,
        )

        # ── Step 6: Initial two-view reconstruction ───────────────────
        ok = initialize_reconstruction(
            images, recon,
            seed_selector=cfg.seed_selector,
            pose_recoverer=cfg.pose_recoverer,
            max_reproj_error=cfg.max_reproj_init,
            min_triangulation_angle_deg=cfg.min_angle_init,
        )
        if not ok:
            print("ERROR: Initial reconstruction failed. Aborting.")
            return recon

        # Run BA on seed pair
        run_bundle_adjustment(recon, adjuster=cfg.bundle_adjuster)
        filter_outliers(recon, point_filter=cfg.point_filter)

        # ── Steps 7–10: Incremental growth ───────────────────────────
        n_total = len(images)
        n_registered = 2   # seed pair
        images_since_ba = 0

        print(f"\n{'─'*60}")
        print(f"  Incremental growth  ({n_total - 2} images remaining)")
        print(f"{'─'*60}")

        while n_registered < n_total:
            # Step 7: Register next image
            new_id = register_next_image(
                recon,
                view_selector=cfg.view_selector,
                pnp_solver=cfg.pnp_solver,
                min_inliers=cfg.min_pnp_inliers,
            )

            if new_id is None:
                print("No more images can be registered. Stopping.")
                break

            # Step 8: Triangulate new points
            triangulate_new_points(
                new_id, images, recon,
                triangulator=cfg.triangulator,
                max_reproj_error=cfg.max_reproj_tri,
                min_triangulation_angle_deg=cfg.min_angle_tri,
            )

            n_registered += 1
            images_since_ba += 1

            # Step 9: Bundle adjustment (periodic)
            if images_since_ba >= cfg.ba_every_n_images:
                run_bundle_adjustment(recon, adjuster=cfg.bundle_adjuster)
                filter_outliers(recon, point_filter=cfg.point_filter)
                images_since_ba = 0

            print(f"  Progress: {n_registered}/{n_total} registered  "
                  f"| {len(recon.points3d)} 3-D points")

        # ── Final BA + filter ─────────────────────────────────────────
        print(f"\n{'─'*60}")
        print("  Final bundle adjustment")
        print(f"{'─'*60}")
        run_bundle_adjustment(recon, adjuster=cfg.bundle_adjuster)
        filter_outliers(recon, point_filter=cfg.point_filter)

        # ── Step 11: Export ───────────────────────────────────────────
        export_reconstruction(recon, cfg.output_dir, exporter=cfg.exporter)

        elapsed = time.time() - t0
        print(f"\n{'='*60}")
        print(f"  DONE  {len(recon.registered_images)} cameras  "
              f"{len(recon.points3d)} points  [{elapsed:.1f}s]")
        print(f"{'='*60}")

        return recon


# ============================================================
#  QUICK-START EXAMPLE
# ============================================================

if __name__ == "__main__":
    import sys
    from step1_image_loader      import ResizePreprocessor, BasicImageLoader, NoOpPreprocessor
    from step2_feature_extraction import SIFTExtractor
    from step3_feature_matching   import SequentialPairSelector, BFMatcher, ExhaustivePairSelector
    from step4_geometric_verification import FundamentalRANSAC, HomographyRANSAC
    from step5_camera_intrinsics  import EXIFIntrinsicsEstimator,PrincipalPointEstimator,CalibrationFileEstimator
    from step6_initial_reconstruction import EssentialMatrixRecoverer, HomographyRatioSeedSelector, MaxInliersSeedSelector
    from step7_camera_registration import ScoreBasedSelector, EPnPRANSAC, MaxCorrespondencesSelector
    from step8_triangulation      import DLTTriangulator
    from step9_bundle_adjustment   import SciPyBundleAdjuster
    from step10_outlier_filtering  import ReprojectionFilter, TrackLengthFilter, TriangulationAngleFilter
    from step11_export            import COLMAPExporter

    image_dir  = sys.argv[1] if len(sys.argv) > 1 else "your_images_directory/"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "your_sparse_directory/"

    # ── Default run (all built-in defaults) ──────────────────────────
    pipeline = SfMPipeline(image_dir="your_images_directory/", 
                           output_dir="your_sparse_directory/",
                           preprocessor=None,
                           pair_selector=SequentialPairSelector(),
                           matcher=BFMatcher(),
                           verifier=FundamentalRANSAC(),
                           intrinsics_estimator=EXIFIntrinsicsEstimator(),
                           seed_selector=HomographyRatioSeedSelector(),
                           pose_recoverer=EssentialMatrixRecoverer(),
                           view_selector=MaxCorrespondencesSelector(),
                           pnp_solver=EPnPRANSAC(),
                           triangulator=DLTTriangulator(),
                           bundle_adjuster=SciPyBundleAdjuster(),
                           point_filter=ReprojectionFilter(),
                           exporter=None,)
    recon = pipeline.run()

    # ── Example: customise individual steps ──────────────────────────
    # from step2_feature_extraction import ORBExtractor
    # from step3_feature_matching   import SequentialPairSelector, BFMatcher
    # from step9_bundle_adjustment  import LocalBundleAdjuster
    # from step11_export            import COLMAPExporter
    #
    # pipeline = SfMPipeline(
    #     image_dir=image_dir,
    #     output_dir=output_dir,
    #     feature_extractor = ORBExtractor(n_features=6000),
    #     pair_selector     = SequentialPairSelector(window=5),
    #     bundle_adjuster   = LocalBundleAdjuster(window=5),
    #     exporter          = COLMAPExporter(),
    # )
    # recon = pipeline.run()

    
