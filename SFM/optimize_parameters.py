"""
Parameter Optimization Script for Incremental SfM Pipeline
==========================================================

This script finds optimal parameters for each step of the SfM pipeline
using the default pipeline (all strategies set to None, using built-in defaults).
"""

import os
import json
import numpy as np
from pathlib import Path
import time

from pipeline import SfMPipeline
from sfm_types import Reconstruction


class ParameterOptimizer:
    """Systematic parameter optimization for SfM pipeline."""

    def __init__(self, image_dir: str, output_base_dir: str):
        self.image_dir = image_dir
        self.output_base_dir = output_base_dir
        self.results = {}

    def evaluate_pipeline(self, config_name: str, output_dir: str, **config_kwargs):
        """
        Run pipeline with given configuration and collect metrics.
        
        Returns metrics dict with:
        - num_registered_cameras
        - num_points_3d
        - mean_reproj_error (estimated)
        - num_verified_matches
        - execution_time
        """
        try:
            start_time = time.time()
            pipeline = SfMPipeline(
                image_dir=self.image_dir,
                output_dir=output_dir,
                **config_kwargs
            )
            reconstruction = pipeline.run()
            elapsed = time.time() - start_time

            # Compute metrics
            metrics = {
                "num_registered_cameras": len(reconstruction.registered_images),
                "num_points_3d": len(reconstruction.points3d),
                "num_verified_matches": len(reconstruction.verified_matches),
                "mean_reproj_error": self._compute_mean_reproj_error(reconstruction),
                "execution_time_sec": elapsed,
                "config_name": config_name,
            }
            self.results[config_name] = metrics
            return metrics
        except Exception as e:
            print(f"ERROR in config '{config_name}': {e}")
            return None

    @staticmethod
    def _compute_mean_reproj_error(reconstruction: Reconstruction) -> float:
        """Compute mean reprojection error across all points."""
        import cv2
        
        total_error = 0.0
        total_obs = 0

        for p3d in reconstruction.points3d.values():
            for (img_id, kp_idx) in p3d.track:
                ri = reconstruction.registered_images.get(img_id)
                cam = reconstruction.cameras.get(img_id)
                kp = reconstruction.keypoints.get(img_id)

                if ri is None or cam is None or kp is None or kp_idx >= len(kp.points):
                    continue

                # Reproject 3D point
                rvec, _ = cv2.Rodrigues(ri.R)
                proj, _ = cv2.projectPoints(
                    p3d.xyz.reshape(1, 1, 3),
                    rvec.reshape(3, 1),
                    ri.t.reshape(3, 1),
                    cam.K,
                    cam.dist_coeffs,
                )
                reproj_error = np.linalg.norm(proj.reshape(2) - kp.points[kp_idx])
                total_error += reproj_error
                total_obs += 1

        return total_error / max(total_obs, 1)

    def print_results(self):
        """Print parameter optimization results."""
        print("\n" + "=" * 80)
        print("PARAMETER OPTIMIZATION RESULTS")
        print("=" * 80)

        if not self.results:
            print("No results to display.")
            return

        # Sort by number of 3D points (descending)
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]["num_points_3d"],
            reverse=True
        )

        print(f"\n{'Config Name':<40} | {'Cameras':>8} | {'Points 3D':>10} | {'Reproj Err':>11} | {'Time(s)':>8}")
        print("-" * 100)

        for config_name, metrics in sorted_results:
            if metrics is None:
                print(f"{config_name:<40} | FAILED")
            else:
                print(
                    f"{config_name:<40} | "
                    f"{metrics['num_registered_cameras']:>8} | "
                    f"{metrics['num_points_3d']:>10} | "
                    f"{metrics['mean_reproj_error']:>11.4f} | "
                    f"{metrics['execution_time_sec']:>8.2f}"
                )

    def save_results(self, output_file: str = "optimization_results.json"):
        """Save results to JSON file."""
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {output_file}")


# ============================================================================
# MAIN OPTIMIZATION ROUTINES
# ============================================================================

def optimize_feature_extraction_params():
    """
    Optimize Step 2: Feature Extraction Parameters

    Default extractor: SIFTExtractor with n_features=5000
    
    Parameters to optimize:
    - n_features: [2000, 3000, 5000, 8000, 10000]  (more features = more time)
    """
    print("\n" + "=" * 80)
    print("STEP 2: FEATURE EXTRACTION OPTIMIZATION")
    print("=" * 80)

    # Image and output directories
    image_dir = "your_image_directory_here"
    output_base = "your_output_directory"

    from step2_feature_extraction import SIFTExtractor

    # Test different feature counts
    n_features_list = [2000, 3000, 5000, 8000, 10000]

    optimizer = ParameterOptimizer(image_dir, output_base)

    for n_feat in n_features_list:
        config_name = f"Step2_SIFT_n_features_{n_feat}"
        output_dir = os.path.join(output_base, config_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nTesting: {config_name}")
        optimizer.evaluate_pipeline(
            config_name,
            output_dir,
            feature_extractor=SIFTExtractor(n_features=n_feat)
        )

    optimizer.print_results()
    optimizer.save_results(os.path.join(output_base, "step2_optimization.json"))


def optimize_feature_matching_params():
    """
    Optimize Step 3: Feature Matching Parameters

    Default: ExhaustivePairSelector + BFMatcher
    
    Parameters to optimize:
    - min_matches: [10, 15, 20, 30, 40]  (minimum matches per pair)
    - pair_selector window (if using sequential)
    """
    print("\n" + "=" * 80)
    print("STEP 3: FEATURE MATCHING OPTIMIZATION")
    print("=" * 80)

    # Image and output directories
    image_dir = "your_image_directory_here"  
    output_base = "your_output_directory"

    from step3_feature_matching import SequentialPairSelector

    # Test different minimum match thresholds
    min_matches_list = [10, 15, 20, 30, 40]

    optimizer = ParameterOptimizer(image_dir, output_base)

    for min_m in min_matches_list:
        config_name = f"Step3_MinMatches_{min_m}"
        output_dir = os.path.join(output_base, config_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nTesting: {config_name}")
        optimizer.evaluate_pipeline(
            config_name,
            output_dir,
            min_matches=min_m
        )

    optimizer.print_results()
    optimizer.save_results(os.path.join(output_base, "step3_optimization.json"))


def optimize_geometric_verification_params():
    """
    Optimize Step 4: Geometric Verification Parameters

    Default: FundamentalRANSAC
    
    Parameters to optimize:
    - min_inliers: [8, 12, 15, 20, 25]  (minimum inliers per match pair)
    - ransac_reproj_thresh: [1.0, 2.0, 3.0, 4.0, 5.0]
    """
    print("\n" + "=" * 80)
    print("STEP 4: GEOMETRIC VERIFICATION OPTIMIZATION")
    print("=" * 80)

    # Image and output directories
    image_dir = "your_image_directory_here"
    output_base = "your_output_directory"

    from step4_geometric_verification import FundamentalRANSAC

    # Test different RANSAC thresholds
    ransac_thresh_list = [1.0, 2.0, 3.0, 4.0, 5.0]

    optimizer = ParameterOptimizer(image_dir, output_base)

    for thresh in ransac_thresh_list:
        config_name = f"Step4_RANSAC_thresh_{thresh}"
        output_dir = os.path.join(output_base, config_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nTesting: {config_name}")
        optimizer.evaluate_pipeline(
            config_name,
            output_dir,
            verifier=FundamentalRANSAC(ransac_reproj_thresh=thresh),
            min_inliers=15
        )

    optimizer.print_results()
    optimizer.save_results(os.path.join(output_base, "step4_optimization.json"))


def optimize_triangulation_params():
    """
    Optimize Step 8: Triangulation Parameters

    Default: DLTTriangulator
    
    Parameters to optimize:
    - max_reproj_error: [2.0, 3.0, 4.0, 5.0, 6.0]  (max reprojection error in pixels)
    - min_triangulation_angle: [1.0, 2.0, 3.0, 4.0, 5.0]  (minimum angle in degrees)
    """
    print("\n" + "=" * 80)
    print("STEP 8: TRIANGULATION OPTIMIZATION")
    print("=" * 80)

    # Image and output directories
    image_dir = "your_image_directory_here"
    output_base = "your_output_directory"

    # Test different reprojection error and angle thresholds
    max_reproj_list = [2.0, 3.0, 4.0, 5.0, 6.0]

    optimizer = ParameterOptimizer(image_dir, output_base)

    for max_repr in max_reproj_list:
        config_name = f"Step8_MaxReproj_{max_repr}"
        output_dir = os.path.join(output_base, config_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nTesting: {config_name}")
        optimizer.evaluate_pipeline(
            config_name,
            output_dir,
            max_reproj_tri=max_repr,
            min_angle_tri=2.0
        )

    optimizer.print_results()
    optimizer.save_results(os.path.join(output_base, "step8_optimization.json"))


def optimize_bundle_adjustment_params():
    """
    Optimize Step 9: Bundle Adjustment Parameters

    Default: SciPyBundleAdjuster with periodic BA every N images
    
    Parameters to optimize:
    - ba_every_n_images: [3, 5, 8, 10]  (frequency of bundle adjustment)
    """
    print("\n" + "=" * 80)
    print("STEP 9: BUNDLE ADJUSTMENT OPTIMIZATION")
    print("=" * 80)

    # Image and output directories
    image_dir = "your_image_directory_here"
    output_base = "your_output_directory"

    # Test different BA frequencies
    ba_frequencies = [3, 5, 8, 10]

    optimizer = ParameterOptimizer(image_dir, output_base)

    for ba_freq in ba_frequencies:
        config_name = f"Step9_BA_every_{ba_freq}_images"
        output_dir = os.path.join(output_base, config_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nTesting: {config_name}")
        optimizer.evaluate_pipeline(
            config_name,
            output_dir,
            ba_every_n_images=ba_freq
        )

    optimizer.print_results()
    optimizer.save_results(os.path.join(output_base, "step9_optimization.json"))


def optimize_outlier_filtering_params():
    """
    Optimize Step 10: Outlier Filtering Parameters

    Default: ReprojectionFilter with threshold = 4.0 pixels
    
    Filtering strategies:
    - ReprojectionFilter: max_error [2.0, 4.0, 6.0, 8.0]
    - TrackLengthFilter: min_observations [2, 3, 4, 5]
    """
    print("\n" + "=" * 80)
    print("STEP 10: OUTLIER FILTERING OPTIMIZATION")
    print("=" * 80)

    # Image and output directories
    image_dir = "your_image_directory_here"
    output_base = "your_output_directory"

    from step10_outlier_filtering import ReprojectionFilter

    # Test different reprojection error thresholds
    reproj_thresholds = [2.0, 4.0, 6.0, 8.0]

    optimizer = ParameterOptimizer(image_dir, output_base)

    for threshold in reproj_thresholds:
        config_name = f"Step10_ReprrojFilter_{threshold}"
        output_dir = os.path.join(output_base, config_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nTesting: {config_name}")
        optimizer.evaluate_pipeline(
            config_name,
            output_dir,
            point_filter=ReprojectionFilter(max_error=threshold)
        )

    optimizer.print_results()
    optimizer.save_results(os.path.join(output_base, "step10_optimization.json"))


def run_full_optimization():
    """
    Run comprehensive parameter optimization across all steps.
    Select which steps to optimize:
    """
    print("\n" + "=" * 80)
    print("INCREMENTAL SFM PARAMETER OPTIMIZATION SUITE")
    print("=" * 80)

    # UNCOMMENT the steps you want to optimize:

    optimize_feature_extraction_params()
    optimize_feature_matching_params()
    optimize_geometric_verification_params()
    optimize_triangulation_params()
    optimize_bundle_adjustment_params()
    optimize_outlier_filtering_params()

    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review the optimization_results_*.json files")
    print("2. Identify the best-performing configurations")
    print("3. Create a custom pipeline with the optimal parameters")
    print("4. Run final pipeline with best config")


# ============================================================================
# EXAMPLE: RUN SPECIFIC OPTIMIZATION
# ============================================================================

if __name__ == "__main__":
    # START HERE: Set your directories
    IMAGE_DIR = "your_image_directory_here"        # e.g., "your_image_directory_here"
    OUTPUT_DIR = "your_output_directory"      # e.g., "your_output_directory"

    # Option 1: Test a single step
    # print("\n" + "=" * 80)
    # print("EXAMPLE: Single-Step Optimization")
    # print("=" * 80)

    # # Tests feature extraction parameter variation
    # image_dir = IMAGE_DIR
    # output_base = OUTPUT_DIR

    # optimizer = ParameterOptimizer(image_dir, output_base)

    # # Test with default (None) parameters
    # print("\nTesting: Default Pipeline (all parameters = None)")
    # output_dir = os.path.join(output_base, "Default_Pipeline")
    # os.makedirs(output_dir, exist_ok=True)
    # default_metrics = optimizer.evaluate_pipeline("Default_Pipeline", output_dir)

    # # Test with a custom configuration
    # #from step2_feature_extraction import SIFTExtractor
    # #print("\nTesting: Custom SIFT (n_features=8000)")
    # #output_dir = os.path.join(output_base, "Custom_SIFT_8000")
    # #os.makedirs(output_dir, exist_ok=True)
    # #custom_metrics = optimizer.evaluate_pipeline(
    # #    "Custom_SIFT_8000",
    # #    output_dir,
    # #    feature_extractor=SIFTExtractor(n_features=8000)
    # #)

    # optimizer.print_results()

    # Option 2: Run full optimization suite
    # Uncomment to run all optimization routines:
    run_full_optimization()

