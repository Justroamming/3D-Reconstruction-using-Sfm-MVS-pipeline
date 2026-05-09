"""
mvs_step4_point_cloud.py
-------------------------
Steps 4–7 — Point Cloud Generation, Filtering, Merging, Normal Estimation

Step 4: Back-project fused depth maps → raw 3-D point cloud per view
Step 5: Filter / clean the raw cloud (statistical, radius, density)
Step 6: Merge per-view clouds + voxel downsampling / deduplication
Step 7: Estimate / refine surface normals
"""

from __future__ import annotations
import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from mvs_types import ACCELERATOR, DepthMap, DensePointCloud, MVSCamera, MVSReconstruction


# ============================================================
#  ─── STEP 4: POINT CLOUD GENERATION ───────────────────────
# ============================================================

class PointGeneratorBase(ABC):
    @abstractmethod
    def generate(
        self,
        depth_map: DepthMap,
        camera: MVSCamera,
        color_image: Optional[np.ndarray] = None,
    ) -> DensePointCloud:
        """Back-project one depth map into a 3-D point cloud."""


class BackProjectionGenerator(PointGeneratorBase):
    """
    Standard back-projection: for every valid depth pixel compute
        X_world = R^T (K^{-1} [u,v,1]^T * d - t)
    Colour is sampled directly from the image.
    Normals are optionally estimated from depth gradients.
    """

    def __init__(self, compute_normals: bool = True):
        self.compute_normals = compute_normals

    def generate(self, depth_map, camera, color_image=None) -> DensePointCloud:
        depth  = depth_map.depth
        H, W   = depth.shape
        valid  = depth > 0

        K_inv = np.linalg.inv(camera.K)

        ys, xs = np.where(valid)
        d_vals = depth[ys, xs].astype(np.float64)

        pix_h = np.stack([xs, ys, np.ones_like(xs)], axis=0).astype(np.float64)
        rays  = K_inv @ pix_h                        # (3, N)
        pts_c = rays * d_vals                         # (3, N) camera space
        pts_w = (camera.R.T @ (pts_c - camera.t[:, None])).T  # (N, 3) world space

        # Colour
        if color_image is not None:
            img = color_image
            if img.shape[:2] != (H, W):
                img = cv2.resize(img, (W, H))
            bgr = img[ys, xs]
            colors = bgr[:, ::-1].copy()             # BGR → RGB
        else:
            colors = np.full((len(ys), 3), 128, dtype=np.uint8)

        conf = depth_map.confidence[ys, xs] if depth_map.confidence is not None else None

        # Surface normals from depth map gradients
        normals = None
        if self.compute_normals:
            normals = self._normals_from_depth(depth, K_inv, camera.R, valid, ys, xs)

        cloud = DensePointCloud(
            points=pts_w.astype(np.float64),
            colors=colors.astype(np.uint8),
            normals=normals,
            confidences=conf.astype(np.float32) if conf is not None else None,
        )
        return cloud

    @staticmethod
    def _normals_from_depth(
        depth: np.ndarray,
        K_inv: np.ndarray,
        R: np.ndarray,
        valid: np.ndarray,
        ys: np.ndarray,
        xs: np.ndarray,
    ) -> np.ndarray:
        """Estimate per-point normals from depth map gradients (Sobel)."""
        dzdx = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=5)
        dzdy = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=5)

        # 3-D gradient via chain rule
        fx = 1.0 / K_inv[0, 0]   # approximate
        fy = 1.0 / K_inv[1, 1]

        nx = -dzdx[ys, xs] * fx
        ny = -dzdy[ys, xs] * fy
        nz = depth[ys, xs].astype(np.float64)

        norms_cam = np.stack([nx, ny, nz], axis=1)          # camera space
        lengths = np.linalg.norm(norms_cam, axis=1, keepdims=True) + 1e-10
        norms_cam /= lengths
        norms_world = (R.T @ norms_cam.T).T                 # world space
        return norms_world.astype(np.float64)


class BackProjectionGeneratorGPU(PointGeneratorBase):
    """
    GPU-accelerated back-projection via PyTorch.
    Falls back to CPU version if PyTorch unavailable.
    """

    def __init__(self, compute_normals: bool = True):
        self.compute_normals = compute_normals

    def generate(self, depth_map, camera, color_image=None) -> DensePointCloud:
        try:
            import torch
        except ImportError:
            return BackProjectionGenerator(self.compute_normals).generate(
                depth_map, camera, color_image)

        import torch
        device = torch.device(ACCELERATOR.torch_device)
        depth = depth_map.depth
        H, W  = depth.shape

        d_t = torch.from_numpy(depth.astype(np.float32)).to(device)
        valid_t = d_t > 0
        ys_t, xs_t = torch.where(valid_t)

        K_inv = np.linalg.inv(camera.K)
        K_inv_t = torch.from_numpy(K_inv.astype(np.float32)).to(device)
        R_t     = torch.from_numpy(camera.R.astype(np.float32)).to(device)
        t_t     = torch.from_numpy(camera.t.astype(np.float32)).to(device)

        d_vals = d_t[ys_t, xs_t].float()
        pix_h  = torch.stack([xs_t.float(), ys_t.float(),
                               torch.ones_like(xs_t, dtype=torch.float32)], dim=0)
        rays   = K_inv_t @ pix_h                       # (3, N)
        pts_c  = rays * d_vals.unsqueeze(0)             # (3, N)
        pts_w  = (R_t.T @ (pts_c.T - t_t).T).T         # (N, 3)

        pts_np = pts_w.cpu().numpy().astype(np.float64)
        ys_np  = ys_t.cpu().numpy()
        xs_np  = xs_t.cpu().numpy()

        if color_image is not None:
            img = color_image
            if img.shape[:2] != (H, W):
                img = cv2.resize(img, (W, H))
            bgr = img[ys_np, xs_np]
            colors = bgr[:, ::-1].copy()
        else:
            colors = np.full((len(ys_np), 3), 128, dtype=np.uint8)

        conf = depth_map.confidence[ys_np, xs_np].astype(np.float32) \
               if depth_map.confidence is not None else None

        normals = None
        if self.compute_normals:
            normals = BackProjectionGenerator._normals_from_depth(
                depth, K_inv, camera.R, depth > 0, ys_np, xs_np)

        return DensePointCloud(
            points=pts_np,
            colors=colors.astype(np.uint8),
            normals=normals,
            confidences=conf,
        )


def generate_point_clouds(
    reconstruction: MVSReconstruction,
    generator: Optional[PointGeneratorBase] = None,
    use_fused: bool = True,
) -> Dict[int, DensePointCloud]:
    """
    Step 4 — Back-project depth maps into per-view point clouds.

    Parameters
    ----------
    reconstruction : MVSReconstruction
    generator      : PointGeneratorBase (default: GPU if available else CPU)
    use_fused      : use fused depth maps if available, else raw depth maps

    Returns
    -------
    per_view_clouds : dict image_id → DensePointCloud
    """
    if generator is None:
        if ACCELERATOR.has_cuda or ACCELERATOR.has_mps:
            generator = BackProjectionGeneratorGPU(compute_normals=True)
        else:
            generator = BackProjectionGenerator(compute_normals=True)

    depth_source = reconstruction.fused_depth if (
        use_fused and reconstruction.fused_depth
    ) else reconstruction.depth_maps

    per_view: Dict[int, DensePointCloud] = {}
    print(f"[Step 4] Point cloud generation ({generator.__class__.__name__}) "
          f"— {len(depth_source)} views")

    for iid, dm in depth_source.items():
        cam = reconstruction.cameras[iid]
        img = cv2.imread(cam.image_path)
        if img is not None and img.shape[:2] != (dm.height, dm.width):
            img = cv2.resize(img, (dm.width, dm.height))

        cloud = generator.generate(dm, cam, img)
        per_view[iid] = cloud
        print(f"  [img {iid:03d}] {len(cloud)} points generated")

    return per_view


# ============================================================
#  ─── STEP 5: POINT CLOUD FILTERING ────────────────────────
# ============================================================

class PointFilterBase(ABC):
    @abstractmethod
    def filter(self, cloud: DensePointCloud) -> DensePointCloud:
        """Return a filtered copy of the cloud."""


class StatisticalOutlierFilter(PointFilterBase):
    """
    Remove points whose mean distance to k nearest neighbours is more than
    `std_ratio` standard deviations from the global mean.
    CPU: scipy KDTree.   GPU: PyTorch pairwise (for smaller clouds).
    """

    def __init__(self, k: int = 20, std_ratio: float = 2.0, use_gpu: bool = False):
        self.k = k
        self.std_ratio = std_ratio
        self.use_gpu = use_gpu and (ACCELERATOR.has_cuda or ACCELERATOR.has_mps)

    def filter(self, cloud: DensePointCloud) -> DensePointCloud:
        pts = cloud.points
        if len(pts) < self.k + 1:
            return cloud

        if self.use_gpu:
            mean_dists = self._gpu_knn_mean(pts)
        else:
            mean_dists = self._cpu_knn_mean(pts)

        mu, sigma = mean_dists.mean(), mean_dists.std()
        thresh = mu + self.std_ratio * sigma
        # Use <= so sigma==0 keeps points; avoid wiping cloud on degenerate stats.
        keep = np.isfinite(mean_dists) & (mean_dists <= thresh)
        if not np.any(keep):
            return DensePointCloud.empty()
        return self._mask(cloud, keep)

    def _cpu_knn_mean(self, pts):
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        dists, _ = tree.query(pts, k=self.k + 1)
        return dists[:, 1:].mean(axis=1)

    def _gpu_knn_mean(self, pts):
        try:
            import torch
            device = torch.device(ACCELERATOR.torch_device)
            pts_t = torch.from_numpy(pts.astype(np.float32)).to(device)
            # Chunked pairwise to avoid OOM
            chunk = 4096
            mean_dists = []
            for i in range(0, len(pts_t), chunk):
                diff = pts_t[i:i+chunk].unsqueeze(1) - pts_t.unsqueeze(0)
                d = diff.norm(dim=-1)
                topk = torch.topk(d, self.k + 1, largest=False).values[:, 1:]
                mean_dists.append(topk.mean(dim=1).cpu().numpy())
            return np.concatenate(mean_dists)
        except Exception:
            return self._cpu_knn_mean(pts)

    @staticmethod
    def _mask(cloud: DensePointCloud, keep: np.ndarray) -> DensePointCloud:
        keep = np.asarray(keep, dtype=bool).reshape(-1)

        points = cloud.points
        colors = cloud.colors
        normals = cloud.normals
        confidences = cloud.confidences

        # Defensive fix: some upstream paths may accidentally provide points as (3, N).
        if points.ndim == 2 and points.shape[0] == 3 and points.shape[1] == keep.shape[0]:
            points = points.T
            if normals is not None and normals.ndim == 2 and normals.shape[0] == 3 and normals.shape[1] == keep.shape[0]:
                normals = normals.T

        return DensePointCloud(
            points=points[keep],
            colors=colors[keep],
            normals=normals[keep] if normals is not None else None,
            confidences=confidences[keep] if confidences is not None else None,
        )


class RadiusOutlierFilter(PointFilterBase):
    """Remove points that have fewer than `min_neighbours` within radius `r`."""

    def __init__(self, radius: float = 0.05, min_neighbours: int = 5):
        self.radius = radius
        self.min_n  = min_neighbours

    def filter(self, cloud: DensePointCloud) -> DensePointCloud:
        from scipy.spatial import cKDTree
        tree = cKDTree(cloud.points)
        counts = np.array([len(tree.query_ball_point(p, self.radius)) - 1
                           for p in cloud.points])
        keep = counts >= self.min_n
        return StatisticalOutlierFilter._mask(cloud, keep)


class ConfidenceFilter(PointFilterBase):
    """Keep only points with confidence >= threshold."""

    def __init__(self, min_confidence: float = 0.3):
        self.min_conf = min_confidence

    def filter(self, cloud: DensePointCloud) -> DensePointCloud:
        if cloud.confidences is None:
            return cloud
        keep = cloud.confidences >= self.min_conf
        if not np.any(keep):
            return DensePointCloud.empty()
        return StatisticalOutlierFilter._mask(cloud, keep)


class ChainPointFilter(PointFilterBase):
    """Apply multiple filters in sequence."""

    def __init__(self, filters: List[PointFilterBase]):
        self.filters = filters

    def filter(self, cloud: DensePointCloud) -> DensePointCloud:
        for f in self.filters:
            cloud = f.filter(cloud)
        return cloud


def filter_point_clouds(
    per_view_clouds: Dict[int, DensePointCloud],
    point_filter: Optional[PointFilterBase] = None,
) -> Dict[int, DensePointCloud]:
    """Step 5 — Filter per-view point clouds."""
    if point_filter is None:
        # Default to geometry-only filtering. Confidence maps after fusion can be
        # extremely low and would otherwise drop all points.
        point_filter = ChainPointFilter([
            StatisticalOutlierFilter(k=20, std_ratio=2.5,
                                     use_gpu=ACCELERATOR.has_cuda or ACCELERATOR.has_mps),
        ])

    print(f"[Step 5] Point filtering ({point_filter.__class__.__name__})")
    filtered = {}
    for iid, cloud in per_view_clouds.items():
        f_cloud = point_filter.filter(cloud)
        filtered[iid] = f_cloud
        print(f"  [img {iid:03d}] {len(cloud)} → {len(f_cloud)} points")
    return filtered


# ============================================================
#  ─── STEP 6: MERGE & DEDUPLICATE ───────────────────────────
# ============================================================

class CloudMergerBase(ABC):
    @abstractmethod
    def merge(self, per_view_clouds: Dict[int, DensePointCloud]) -> DensePointCloud:
        """Combine per-view clouds into one unified dense cloud."""


class VoxelGridMerger(CloudMergerBase):
    """
    Merge all per-view clouds and apply voxel grid downsampling.
    Each occupied voxel contributes one point (centroid of all points in voxel).
    Colour is averaged; normals are averaged and renormalised.
    """

    def __init__(self, voxel_size: float = 0.01):
        self.voxel_size = voxel_size

    def merge(self, per_view_clouds: Dict[int, DensePointCloud]) -> DensePointCloud:
        if not per_view_clouds:
            return DensePointCloud.empty()

        if sum(len(c) for c in per_view_clouds.values()) == 0:
            return DensePointCloud.empty()

        # Concatenate all clouds
        all_pts  = np.vstack([c.points for c in per_view_clouds.values()])
        all_col  = np.vstack([c.colors for c in per_view_clouds.values()])
        has_norm = all(c.normals is not None for c in per_view_clouds.values())
        all_norm = np.vstack([c.normals for c in per_view_clouds.values()]) if has_norm else None

        # Assign each point to a voxel
        origin = all_pts.min(axis=0)
        voxel_idx = np.floor((all_pts - origin) / self.voxel_size).astype(np.int64)

        # Use row-wise unique voxel coordinates to avoid integer overflow collisions.
        unique_vox, inverse = np.unique(voxel_idx, axis=0, return_inverse=True)
        voxel_counts = np.bincount(inverse)

        pts_out  = np.zeros((len(unique_vox), 3), dtype=np.float64)
        col_out  = np.zeros((len(unique_vox), 3), dtype=np.float64)
        norm_out = np.zeros((len(unique_vox), 3), dtype=np.float64) if has_norm else None

        np.add.at(pts_out, inverse, all_pts)
        np.add.at(col_out, inverse, all_col.astype(np.float64))

        if has_norm and norm_out is not None:
            np.add.at(norm_out, inverse, all_norm)

        denom = voxel_counts[:, None].astype(np.float64)
        pts_out /= np.maximum(denom, 1e-10)
        col_out /= np.maximum(denom, 1e-10)

        if has_norm and norm_out is not None:
            norm_out /= np.maximum(denom, 1e-10)
            if has_norm and norm_out is not None:
                lengths = np.linalg.norm(norm_out, axis=1, keepdims=True)
                norm_out /= lengths + 1e-10

        return DensePointCloud(
            points=pts_out,
            colors=np.clip(col_out, 0, 255).astype(np.uint8),
            normals=norm_out,
        )


class SimpleMerger(CloudMergerBase):
    """Concatenate all per-view clouds without any deduplication."""

    def merge(self, per_view_clouds: Dict[int, DensePointCloud]) -> DensePointCloud:
        if not per_view_clouds:
            return DensePointCloud.empty()
        result = DensePointCloud.empty()
        for cloud in per_view_clouds.values():
            result = result.append(cloud)
        return result


def merge_point_clouds(
    per_view_clouds: Dict[int, DensePointCloud],
    reconstruction: MVSReconstruction,
    merger: Optional[CloudMergerBase] = None,
) -> DensePointCloud:
    """Step 6 — Merge per-view clouds into one dense cloud."""
    if merger is None:
        merger = VoxelGridMerger(voxel_size=0.005)

    total_before = sum(len(c) for c in per_view_clouds.values())
    print(f"[Step 6] Merging {len(per_view_clouds)} clouds "
          f"({total_before} total points) via {merger.__class__.__name__}")

    if total_before == 0:
        reconstruction.dense_cloud = DensePointCloud.empty()
        print("  -> no points to merge (all per-view clouds are empty)")
        return reconstruction.dense_cloud

    merged = merger.merge(per_view_clouds)
    reconstruction.dense_cloud = merged
    print(f"  → {len(merged)} points after merge/downsample")
    return merged


# ============================================================
#  ─── STEP 7: NORMAL ESTIMATION ─────────────────────────────
# ============================================================

class NormalEstimatorBase(ABC):
    @abstractmethod
    def estimate(self, cloud: DensePointCloud) -> DensePointCloud:
        """Add or refine normals in the cloud."""


class PCLNormalEstimator(NormalEstimatorBase):
    """
    PCA-based normal estimation (CPU, scipy KDTree).
    For each point, fits a plane to its k nearest neighbours via SVD.
    Orients normals consistently using a propagation heuristic.
    """

    def __init__(
        self,
        k: int = 20,
        orient_towards_origin: bool = True,
        query_chunk_size: int = 50000,
    ):
        self.k = k
        self.orient = orient_towards_origin
        self.query_chunk_size = max(1, int(query_chunk_size))

    def estimate(self, cloud: DensePointCloud) -> DensePointCloud:
        from scipy.spatial import cKDTree
        pts = cloud.points
        n_points = len(pts)
        k_neighbors = min(self.k, n_points - 1)
        if k_neighbors < 3:
            return cloud

        tree = cKDTree(pts)
        normals = np.zeros((n_points, 3), dtype=np.float64)
        query_k = k_neighbors + 1

        for start in range(0, n_points, self.query_chunk_size):
            end = min(start + self.query_chunk_size, n_points)
            _, nn_idx = tree.query(pts[start:end], k=query_k)
            neighbours = pts[nn_idx[:, 1:]]                      # (C, k, 3)
            centred = neighbours - neighbours.mean(axis=1, keepdims=True)

            # Batched 3x3 covariance eigendecomposition is much faster than per-point SVD.
            cov = np.matmul(centred.transpose(0, 2, 1), centred)
            eigvals, eigvecs = np.linalg.eigh(cov)
            normals[start:end] = eigvecs[:, :, 0]                # smallest eigenvector

        # Orient normals towards camera/origin
        if self.orient:
            dot = np.sum(normals * (-pts), axis=1)       # towards origin
            normals[dot < 0] *= -1

        return DensePointCloud(
            points=cloud.points,
            colors=cloud.colors,
            normals=normals.astype(np.float64),
            confidences=cloud.confidences,
        )


class PCLNormalEstimatorGPU(NormalEstimatorBase):
    """
    GPU-accelerated PCA normal estimation.
    Uses PyTorch for batched SVD. Falls back to CPU if unavailable.
    """

    def __init__(
        self,
        k: int = 20,
        orient_towards_origin: bool = True,
        chunk_size: int = 2048,
        query_chunk_size: int = 20000,
    ):
        self.k = k
        self.orient = orient_towards_origin
        self.chunk = chunk_size
        self.query_chunk_size = max(1, int(query_chunk_size))

    def estimate(self, cloud: DensePointCloud) -> DensePointCloud:
        try:
            import torch
        except ImportError:
            return PCLNormalEstimator(self.k, self.orient).estimate(cloud)

        import torch
        device = torch.device(ACCELERATOR.torch_device)
        pts = cloud.points.astype(np.float32)
        N   = len(pts)
        k_neighbors = min(self.k, N - 1)
        if k_neighbors < 3:
            return cloud

        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        query_k = k_neighbors + 1
        normals_np = np.zeros((N, 3), dtype=np.float64)

        # Query neighbors on CPU in bounded chunks; run PCA on GPU per mini-batch.
        for q_start in range(0, N, self.query_chunk_size):
            q_end = min(q_start + self.query_chunk_size, N)
            _, nn_idx = tree.query(pts[q_start:q_end], k=query_k)
            nn_idx = nn_idx[:, 1:]
            neighbours = pts[nn_idx]                           # (Q, k, 3)

            for b_start in range(0, len(neighbours), self.chunk):
                b_end = min(b_start + self.chunk, len(neighbours))
                nb_t = torch.from_numpy(neighbours[b_start:b_end]).to(device)
                centred = nb_t - nb_t.mean(dim=1, keepdim=True)
                try:
                    _, _, Vt = torch.linalg.svd(centred, full_matrices=False)
                    batch_normals = Vt[:, -1, :]              # smallest singular vector
                except Exception:
                    # Robust fallback: CPU covariance eigendecomposition.
                    c_np = centred.detach().cpu().numpy().astype(np.float64)
                    cov = np.matmul(c_np.transpose(0, 2, 1), c_np)
                    _, eigvecs = np.linalg.eigh(cov)
                    batch_normals = torch.from_numpy(eigvecs[:, :, 0].astype(np.float32)).to(device)

                out_start = q_start + b_start
                out_end = q_start + b_end
                normals_np[out_start:out_end] = batch_normals.cpu().numpy().astype(np.float64)

        if self.orient:
            dot = np.sum(normals_np * (-cloud.points), axis=1)
            normals_np[dot < 0] *= -1

        return DensePointCloud(
            points=cloud.points,
            colors=cloud.colors,
            normals=normals_np,
            confidences=cloud.confidences,
        )


class DepthGradientNormalEstimator(NormalEstimatorBase):
    """
    If normals are already estimated from depth maps (Step 4), this refiner
    smooths them with a neighbourhood average.
    Useful when normals from back-projection are noisy.
    """

    def __init__(self, smooth_iters: int = 2, k: int = 10, query_chunk_size: int = 50000):
        self.iters = smooth_iters
        self.k = k
        self.query_chunk_size = max(1, int(query_chunk_size))

    def estimate(self, cloud: DensePointCloud) -> DensePointCloud:
        if cloud.normals is None:
            return PCLNormalEstimator(self.k).estimate(cloud)

        from scipy.spatial import cKDTree
        n_points = len(cloud.points)
        k_neighbors = min(self.k, n_points - 1)
        if k_neighbors < 1:
            return cloud

        normals = cloud.normals.copy()
        tree    = cKDTree(cloud.points)
        query_k = k_neighbors + 1

        for _ in range(self.iters):
            new_normals = np.zeros_like(normals)
            for start in range(0, n_points, self.query_chunk_size):
                end = min(start + self.query_chunk_size, n_points)
                _, nn_idx = tree.query(cloud.points[start:end], k=query_k)
                new_normals[start:end] = normals[nn_idx[:, 1:]].mean(axis=1)
            lengths = np.linalg.norm(new_normals, axis=1, keepdims=True)
            normals = new_normals / (lengths + 1e-10)

        # Re-orient
        dot = np.sum(normals * (-cloud.points), axis=1)
        normals[dot < 0] *= -1

        return DensePointCloud(
            points=cloud.points,
            colors=cloud.colors,
            normals=normals.astype(np.float64),
            confidences=cloud.confidences,
        )


def estimate_normals(
    reconstruction: MVSReconstruction,
    estimator: Optional[NormalEstimatorBase] = None,
) -> DensePointCloud:
    """
    Step 7 — Estimate or refine surface normals on the dense cloud.

    Parameters
    ----------
    reconstruction : MVSReconstruction (dense_cloud modified in-place)
    estimator      : NormalEstimatorBase (default: GPU if available else CPU PCA)

    Returns
    -------
    DensePointCloud with normals
    """
    if estimator is None:
        if ACCELERATOR.has_cuda or ACCELERATOR.has_mps:
            estimator = PCLNormalEstimatorGPU(k=20)
        else:
            estimator = PCLNormalEstimator(k=20)

    cloud = reconstruction.dense_cloud
    print(f"[Step 7] Normal estimation ({estimator.__class__.__name__}) "
          f"— {len(cloud)} points")

    cloud_with_normals = estimator.estimate(cloud)
    reconstruction.dense_cloud = cloud_with_normals

    has_n = cloud_with_normals.normals is not None
    print(f"  Normals computed: {has_n}")
    return cloud_with_normals
