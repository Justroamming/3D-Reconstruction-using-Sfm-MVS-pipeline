"""
mvs_step3_depth_fusion.py
--------------------------
Step 3 — Depth Map Fusion (Multi-View Consistency Check)

Merge depth maps from multiple views into a fused, geometrically consistent set.

Swappable strategies
  CPU:
    - GeometricConsistencyFusion  — COLMAP-style reprojection consistency check
    - TSDFFusion                  — Truncated Signed Distance Function volumetric fusion
    - VotingFusion                — per-pixel depth voting across views
  GPU:
    - GeometricConsistencyFusionGPU — batch reprojection on GPU via PyTorch
"""

from __future__ import annotations
import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from mvs_types import ACCELERATOR, DepthMap, MVSCamera, MVSReconstruction


# ============================================================
#  HELPERS
# ============================================================

def _backproject(
    depth: np.ndarray,
    K_inv: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """
    Back-project all valid pixels in `depth` to world-space 3-D points.
    Returns (N, 3) float64.
    """
    H, W = depth.shape
    ys, xs = np.where(depth > 0)
    depths = depth[ys, xs].astype(np.float64)

    pix_h = np.stack([xs, ys, np.ones_like(xs)], axis=0).astype(np.float64)
    rays   = K_inv @ pix_h                    # (3, N)  camera-space
    pts_c  = rays * depths                    # (3, N)  scaled by depth
    pts_w  = R.T @ (pts_c.T - t).T           # (3, N)  world-space
    return pts_w.T, ys, xs                    # (N,3), ys, xs


# ============================================================
#  INTERFACE
# ============================================================

class DepthFusionBase(ABC):
    @abstractmethod
    def fuse(
        self,
        reconstruction: MVSReconstruction,
    ) -> Dict[int, DepthMap]:
        """
        Fuse all depth maps and return a filtered set.
        Stores results in reconstruction.fused_depth.
        """


# ============================================================
#  CPU IMPLEMENTATIONS
# ============================================================

class GeometricConsistencyFusion(DepthFusionBase):
    """
    COLMAP-style geometric consistency fusion (CPU).

    For every pixel in a reference depth map:
      1. Back-project to world space
      2. Project into each neighbouring depth map
      3. Check that the reprojected depth agrees within `rel_depth_thresh`
         AND the reprojected pixel is within `pixel_thresh` of the original
      4. Keep the pixel only if it passes for at least `min_consistent_views`

    Consistent depths are averaged for robustness.
    """

    def __init__(
        self,
        min_consistent_views: int = 2,
        pixel_thresh: float = 2.0,
        rel_depth_thresh: float = 0.05,
    ):
        self.min_views      = min_consistent_views
        self.pix_thresh     = pixel_thresh
        self.rel_d_thresh   = rel_depth_thresh

    def fuse(self, reconstruction: MVSReconstruction) -> Dict[int, DepthMap]:
        cameras    = reconstruction.cameras
        depth_maps = reconstruction.depth_maps
        ids        = list(depth_maps.keys())
        fused: Dict[int, DepthMap] = {}

        print(f"[Step 3] GeometricConsistencyFusion — {len(ids)} depth maps")

        for ref_id in ids:
            ref_dm  = depth_maps[ref_id]
            ref_cam = cameras[ref_id]
            H, W    = ref_dm.height, ref_dm.width

            K_ref     = ref_cam.K
            K_ref_inv = np.linalg.inv(K_ref)

            depth_ref  = ref_dm.depth.astype(np.float64)
            consistent = np.zeros((H, W), dtype=np.int32)
            depth_sum  = np.zeros((H, W), dtype=np.float64)

            src_ids = [i for i in ids if i != ref_id]

            for src_id in src_ids:
                src_dm  = depth_maps[src_id]
                src_cam = cameras[src_id]
                K_src   = src_cam.K

                # All valid reference pixels
                ys, xs = np.where(depth_ref > 0)
                if len(ys) == 0:
                    continue

                d_ref = depth_ref[ys, xs]
                pix_h = np.stack([xs, ys, np.ones_like(xs)], axis=0).astype(np.float64)
                rays  = K_ref_inv @ pix_h                 # (3, N) camera-space
                pts_c = rays * d_ref                      # scaled
                # Keep world points as (N, 3) for downstream camera projections.
                pts_w = (ref_cam.R.T @ (pts_c - ref_cam.t[:, None])).T

                # Project into source camera
                pts_sc = (src_cam.R @ pts_w.T).T + src_cam.t    # source camera space
                valid_fwd = pts_sc[:, 2] > 0
                px_src = np.full((len(ys), 2), -1.0)
                proj = (K_src @ pts_sc[valid_fwd].T).T
                px_src[valid_fwd, 0] = proj[:, 0] / proj[:, 2]
                px_src[valid_fwd, 1] = proj[:, 1] / proj[:, 2]

                # Sample depth from source map
                src_depth = src_dm.depth
                xi = np.round(px_src[:, 0]).astype(int)
                yi = np.round(px_src[:, 1]).astype(int)
                in_bounds = (xi >= 0) & (xi < src_dm.width) & \
                            (yi >= 0) & (yi < src_dm.height) & valid_fwd
                d_src_sampled = np.zeros(len(ys))
                d_src_sampled[in_bounds] = src_depth[yi[in_bounds], xi[in_bounds]]

                # Depth agreement check
                rel_diff = np.abs(d_src_sampled - pts_sc[:, 2]) / (pts_sc[:, 2] + 1e-10)
                depth_ok = (d_src_sampled > 0) & (rel_diff < self.rel_d_thresh)

                # Back-project source depth → world → reference image
                fwd_ok_idx = np.where(in_bounds & depth_ok)[0]
                if len(fwd_ok_idx) == 0:
                    continue

                xi_ok = xi[fwd_ok_idx]; yi_ok = yi[fwd_ok_idx]
                d_src_ok = d_src_sampled[fwd_ok_idx]
                pix_src_h = np.stack([xi_ok, yi_ok, np.ones_like(xi_ok)], axis=0).astype(np.float64)
                K_src_inv = np.linalg.inv(K_src)
                rays_src = K_src_inv @ pix_src_h
                pts_sc2  = rays_src * d_src_ok
                # Keep world points as (N, 3) for reprojection into reference view.
                pts_w2   = (src_cam.R.T @ (pts_sc2 - src_cam.t[:, None])).T

                # Project back into reference
                pts_rc2 = (ref_cam.R @ pts_w2.T).T + ref_cam.t
                valid_bk = pts_rc2[:, 2] > 0
                proj_bk  = (K_ref @ pts_rc2[valid_bk].T).T
                px_bk    = proj_bk[:, :2] / proj_bk[:, 2:3]

                orig_x = xs[fwd_ok_idx][valid_bk]
                orig_y = ys[fwd_ok_idx][valid_bk]
                pixel_err = np.linalg.norm(
                    px_bk - np.stack([orig_x, orig_y], axis=1), axis=1
                )
                pixel_ok = pixel_err < self.pix_thresh

                final_ys = orig_y[pixel_ok]
                final_xs = orig_x[pixel_ok]
                final_d  = pts_sc[fwd_ok_idx][valid_bk][pixel_ok, 2]

                consistent[final_ys, final_xs] += 1
                depth_sum[final_ys, final_xs]  += final_d

            # Build fused depth map
            enough = consistent >= self.min_views
            depth_fused = np.zeros((H, W), dtype=np.float32)
            depth_fused[enough] = (depth_sum[enough] / consistent[enough]).astype(np.float32)

            conf_fused = (consistent / max(len(src_ids), 1)).astype(np.float32)
            conf_fused = np.clip(conf_fused, 0, 1)
            conf_fused[~enough] = 0.0

            dm_fused = DepthMap(
                image_id=ref_id,
                depth=depth_fused,
                confidence=conf_fused,
            )
            fused[ref_id] = dm_fused
            reconstruction.fused_depth[ref_id] = dm_fused

            n_before = ref_dm.valid_count()
            n_after  = dm_fused.valid_count()
            print(f"  [img {ref_id:03d}] {n_before} → {n_after} valid pixels "
                  f"({100*n_after/max(n_before,1):.1f}% kept)")

        return fused


class TSDFFusion(DepthFusionBase):
    """
    Truncated Signed Distance Function (TSDF) volumetric fusion.
    Fuses all depth maps into a 3-D voxel grid, then extracts
    a consistent depth map per camera by ray-marching.

    Memory: voxel_resolution^3 × 8 bytes.
    Use lower resolution for large scenes.
    """

    def __init__(
        self,
        voxel_resolution: int = 256,
        truncation_factor: float = 5.0,     # truncation = factor × voxel_size
        min_weight: float = 2.0,
    ):
        self.resolution = voxel_resolution
        self.trunc_factor = truncation_factor
        self.min_weight = min_weight

    def fuse(self, reconstruction: MVSReconstruction) -> Dict[int, DepthMap]:
        cameras    = reconstruction.cameras
        depth_maps = reconstruction.depth_maps
        ids        = list(depth_maps.keys())
        print(f"[Step 3] TSDFFusion (res={self.resolution}³) — {len(ids)} depth maps")

        # Compute scene bounding box from sparse cloud or depth maps
        if reconstruction.sparse_points is not None and len(reconstruction.sparse_points) > 0:
            pts = reconstruction.sparse_points
        else:
            pts = []
            for iid in ids:
                cam = cameras[iid]
                dm  = depth_maps[iid]
                K_inv = np.linalg.inv(cam.K)
                p, _, _ = _backproject(dm.depth, K_inv, cam.R, cam.t)
                if len(p):
                    pts.append(p)
            pts = np.vstack(pts) if pts else np.zeros((1, 3))

        scene_min = pts.min(axis=0) - 0.5
        scene_max = pts.max(axis=0) + 0.5
        scene_size = scene_max - scene_min
        voxel_size = float(scene_size.max() / self.resolution)
        trunc = self.trunc_factor * voxel_size

        R = self.resolution
        tsdf   = np.ones((R, R, R), dtype=np.float32)
        weight = np.zeros((R, R, R), dtype=np.float32)

        # Voxel centres
        xs = np.linspace(scene_min[0], scene_max[0], R)
        ys = np.linspace(scene_min[1], scene_max[1], R)
        zs = np.linspace(scene_min[2], scene_max[2], R)
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
        pts_w = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)   # (R³, 3)

        for iid in ids:
            cam = cameras[iid]
            dm  = depth_maps[iid]
            H, W = dm.height, dm.width
            depth_img = dm.depth

            # Project all voxels into this camera
            pts_c = (cam.R @ pts_w.T).T + cam.t
            valid_z = pts_c[:, 2] > 0
            proj = (cam.K @ pts_c[valid_z].T).T
            px   = proj[:, :2] / proj[:, 2:3]

            xi = np.round(px[:, 0]).astype(int)
            yi = np.round(px[:, 1]).astype(int)
            in_bounds = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            measured_d = np.zeros(valid_z.sum())
            measured_d[in_bounds] = depth_img[yi[in_bounds], xi[in_bounds]]

            sdf = measured_d - pts_c[valid_z, 2]
            valid_obs = (measured_d > 0) & (sdf > -trunc)
            sdf_trunc = np.clip(sdf / trunc, -1, 1)

            idx = np.where(valid_z)[0][valid_obs]
            if dm.confidence is not None:
                # Build one fusion weight per valid TSDF update (aligned with idx/sdf_trunc[valid_obs]).
                w_upd = np.ones(valid_obs.sum(), dtype=np.float32)
                obs_in_bounds = in_bounds[valid_obs]
                if np.any(obs_in_bounds):
                    yi_obs = yi[valid_obs][obs_in_bounds]
                    xi_obs = xi[valid_obs][obs_in_bounds]
                    lin = np.clip(yi_obs * W + xi_obs, 0, H * W - 1)
                    w_upd[obs_in_bounds] = dm.confidence.ravel()[lin].astype(np.float32)
            else:
                w_upd = np.ones(valid_obs.sum(), dtype=np.float32)

            tsdf_flat   = tsdf.ravel()
            weight_flat = weight.ravel()
            w_old = weight_flat[idx]
            w_new = w_upd.astype(np.float32)
            tsdf_flat[idx]   = (w_old * tsdf_flat[idx] + w_new * sdf_trunc[valid_obs]) / (w_old + w_new)
            weight_flat[idx] += w_new
            tsdf[:]   = tsdf_flat.reshape(R, R, R)
            weight[:] = weight_flat.reshape(R, R, R)

        # Extract consistent depth per camera by TSDF ray marching
        fused: Dict[int, DepthMap] = {}
        for iid in ids:
            cam = cameras[iid]
            dm  = depth_maps[iid]
            H, W = dm.height, dm.width
            K_inv = np.linalg.inv(cam.K)

            ys_px, xs_px = np.mgrid[0:H, 0:W]
            pix_h = np.stack([xs_px.ravel(), ys_px.ravel(), np.ones(H*W)], axis=0).astype(np.float64)
            rays  = (K_inv @ pix_h).T  # (H*W, 3)

            depth_out = np.zeros(H * W, dtype=np.float32)
            n_steps   = int(scene_size.max() / voxel_size) * 2

            # Simple front-to-back ray marching
            for si in range(n_steps):
                t_ray = 0.1 + si * voxel_size
                pts = rays * t_ray  # camera space
                pts_w_ray = (cam.R.T @ (pts.T - cam.t[:, None])).T  # world
                # Voxel index
                vi = ((pts_w_ray - scene_min) / voxel_size).astype(int)
                in_vol = np.all((vi >= 0) & (vi < R), axis=1)
                active = in_vol & (depth_out == 0)
                if not active.any():
                    break
                idx3 = vi[active]
                tsdf_vals   = tsdf[idx3[:, 0], idx3[:, 1], idx3[:, 2]]
                w_vals      = weight[idx3[:, 0], idx3[:, 1], idx3[:, 2]]
                surface_hit = (tsdf_vals <= 0) & (w_vals >= self.min_weight)
                hit_idx     = np.where(active)[0][surface_hit]
                depth_out[hit_idx] = t_ray

            depth_out = depth_out.reshape(H, W)
            dm_fused = DepthMap(image_id=iid, depth=depth_out, confidence=None)
            fused[iid] = dm_fused
            reconstruction.fused_depth[iid] = dm_fused

            print(f"  [img {iid:03d}] TSDF fused: {dm_fused.valid_count()} valid pixels")

        return fused


class VotingFusion(DepthFusionBase):
    """
    Per-pixel depth voting across views.
    For each reference pixel, collects depth hypotheses from all consistent
    source views and picks the median (or mode in a histogram sense).
    Lightweight and fast.
    """

    def __init__(
        self,
        min_consistent_views: int = 2,
        rel_depth_thresh: float = 0.05,
        pixel_thresh: float = 3.0,
    ):
        self.min_views    = min_consistent_views
        self.rel_thresh   = rel_depth_thresh
        self.pix_thresh   = pixel_thresh

    def fuse(self, reconstruction: MVSReconstruction) -> Dict[int, DepthMap]:
        cameras    = reconstruction.cameras
        depth_maps = reconstruction.depth_maps
        ids        = list(depth_maps.keys())
        fused: Dict[int, DepthMap] = {}
        print(f"[Step 3] VotingFusion — {len(ids)} depth maps")

        for ref_id in ids:
            ref_dm  = depth_maps[ref_id]
            ref_cam = cameras[ref_id]
            H, W    = ref_dm.height, ref_dm.width
            K_ref_inv = np.linalg.inv(ref_cam.K)
            K_ref = ref_cam.K

            # Collect depth votes per pixel
            vote_depths = [[[] for _ in range(W)] for _ in range(H)]

            for src_id in ids:
                if src_id == ref_id:
                    continue
                src_dm  = depth_maps[src_id]
                src_cam = cameras[src_id]
                K_src_inv = np.linalg.inv(src_cam.K)

                ys, xs = np.where(ref_dm.depth > 0)
                if len(ys) == 0:
                    continue
                d_ref = ref_dm.depth[ys, xs].astype(np.float64)

                pix_h = np.stack([xs, ys, np.ones_like(xs)], axis=0).astype(np.float64)
                pts_c = (K_ref_inv @ pix_h) * d_ref
                # Keep world points as (N, 3) for downstream camera projections.
                pts_w = (ref_cam.R.T @ (pts_c - ref_cam.t[:, None])).T

                pts_sc = (src_cam.R @ pts_w.T).T + src_cam.t
                valid  = pts_sc[:, 2] > 0
                proj   = (src_cam.K @ pts_sc[valid].T).T
                px_s   = proj[:, :2] / proj[:, 2:3]

                xi = np.round(px_s[:, 0]).astype(int)
                yi = np.round(px_s[:, 1]).astype(int)
                in_b = (xi >= 0) & (xi < src_dm.width) & (yi >= 0) & (yi < src_dm.height)

                valid_idx = np.where(valid)[0]
                for k, (vk, ib) in enumerate(zip(valid_idx, in_b)):
                    if not ib:
                        continue
                    d_src = float(src_dm.depth[yi[k], xi[k]])
                    d_exp = float(pts_sc[vk, 2])
                    if d_src <= 0:
                        continue
                    if abs(d_src - d_exp) / (d_exp + 1e-10) < self.rel_thresh:
                        # Back-project source depth and verify reprojection lands near original pixel.
                        pix_src_h = np.array([xi[k], yi[k], 1.0], dtype=np.float64)
                        pt_sc2 = (K_src_inv @ pix_src_h) * d_src
                        pt_w2 = src_cam.R.T @ (pt_sc2 - src_cam.t)
                        pt_rc2 = ref_cam.R @ pt_w2 + ref_cam.t
                        if pt_rc2[2] <= 0:
                            continue
                        px_bk = K_ref @ pt_rc2
                        px_bk = px_bk[:2] / px_bk[2]
                        pix_err = float(np.linalg.norm(px_bk - np.array([xs[vk], ys[vk]], dtype=np.float64)))
                        if pix_err >= self.pix_thresh:
                            continue

                        oy, ox = int(ys[vk]), int(xs[vk])
                        if 0 <= oy < H and 0 <= ox < W:
                            vote_depths[oy][ox].append(d_src)

            # For each pixel: median of votes + original depth
            depth_fused = np.zeros((H, W), dtype=np.float32)
            conf_fused  = np.zeros((H, W), dtype=np.float32)
            n_src_total = max(len(ids) - 1, 1)
            for y in range(H):
                for x in range(W):
                    votes = vote_depths[y][x]
                    if len(votes) >= self.min_views:
                        depth_fused[y, x] = float(np.median(votes + [float(ref_dm.depth[y, x])]))
                        conf_fused[y, x]  = min(len(votes) / n_src_total, 1.0)

            dm_fused = DepthMap(image_id=ref_id, depth=depth_fused, confidence=conf_fused)
            fused[ref_id] = dm_fused
            reconstruction.fused_depth[ref_id] = dm_fused
            print(f"  [img {ref_id:03d}] {ref_dm.valid_count()} → {dm_fused.valid_count()} valid")

        return fused


# ============================================================
#  GPU IMPLEMENTATION
# ============================================================

class GeometricConsistencyFusionGPU(DepthFusionBase):
    """
    GPU-accelerated geometric consistency fusion via PyTorch.
    Batches all neighbour projections as tensor operations.
    Falls back to GeometricConsistencyFusion if PyTorch unavailable.
    """

    def __init__(
        self,
        min_consistent_views: int = 2,
        pixel_thresh: float = 2.0,
        rel_depth_thresh: float = 0.05,
    ):
        self.min_views    = min_consistent_views
        self.pix_thresh   = pixel_thresh
        self.rel_d_thresh = rel_depth_thresh

    def fuse(self, reconstruction: MVSReconstruction) -> Dict[int, DepthMap]:
        try:
            import torch
        except ImportError:
            print("  [GeometricConsistencyFusionGPU] falling back to CPU")
            return GeometricConsistencyFusion(
                self.min_views, self.pix_thresh, self.rel_d_thresh
            ).fuse(reconstruction)

        import torch
        device = torch.device(ACCELERATOR.torch_device)
        print(f"[Step 3] GeometricConsistencyFusionGPU (device={device}) — "
              f"{len(reconstruction.depth_maps)} depth maps")

        cameras    = reconstruction.cameras
        depth_maps = reconstruction.depth_maps
        ids        = list(depth_maps.keys())
        fused: Dict[int, DepthMap] = {}

        for ref_id in ids:
            ref_dm  = depth_maps[ref_id]
            ref_cam = cameras[ref_id]
            H, W    = ref_dm.height, ref_dm.width

            K_ref_inv = np.linalg.inv(ref_cam.K)
            K_ref_t = torch.from_numpy(ref_cam.K.astype(np.float32)).to(device)
            depth_ref = ref_dm.depth.astype(np.float32)

            # Move ref depth to GPU
            d_ref_t = torch.from_numpy(depth_ref).to(device)
            ys_t, xs_t = torch.where(d_ref_t > 0)
            if len(ys_t) == 0:
                fused[ref_id] = ref_dm
                reconstruction.fused_depth[ref_id] = ref_dm
                continue

            d_vals = d_ref_t[ys_t, xs_t].float()   # (N,)
            K_ref_inv_t = torch.from_numpy(K_ref_inv.astype(np.float32)).to(device)
            R_ref_t     = torch.from_numpy(ref_cam.R.astype(np.float32)).to(device)
            t_ref_t     = torch.from_numpy(ref_cam.t.astype(np.float32)).to(device)

            pix_h = torch.stack([xs_t.float(), ys_t.float(),
                                  torch.ones_like(xs_t, dtype=torch.float32)], dim=0)  # (3,N)
            pts_c = (K_ref_inv_t @ pix_h) * d_vals.unsqueeze(0)    # (3,N) camera-space
            pts_w = (R_ref_t.T @ (pts_c.T - t_ref_t).T).T          # (N,3) world-space

            consistent_count = torch.zeros(H, W, dtype=torch.int32, device=device)
            depth_sum        = torch.zeros(H, W, dtype=torch.float32, device=device)

            for src_id in ids:
                if src_id == ref_id:
                    continue
                src_dm  = depth_maps[src_id]
                src_cam = cameras[src_id]

                K_src_t = torch.from_numpy(src_cam.K.astype(np.float32)).to(device)
                R_src_t = torch.from_numpy(src_cam.R.astype(np.float32)).to(device)
                t_src_t = torch.from_numpy(src_cam.t.astype(np.float32)).to(device)
                d_src_t = torch.from_numpy(src_dm.depth.astype(np.float32)).to(device)

                pts_sc = (R_src_t @ pts_w.T).T + t_src_t          # (N,3)
                valid  = pts_sc[:, 2] > 0
                proj   = (K_src_t @ pts_sc[valid].T).T
                px_s   = proj[:, :2] / proj[:, 2:3]

                xi = px_s[:, 0].round().long()
                yi = px_s[:, 1].round().long()
                in_b = (xi >= 0) & (xi < src_dm.width) & (yi >= 0) & (yi < src_dm.height)

                d_sampled = torch.zeros(valid.sum(), device=device)
                d_sampled[in_b] = d_src_t[yi[in_b], xi[in_b]]
                d_exp = pts_sc[valid, 2]

                rel_diff = (d_sampled - d_exp).abs() / (d_exp.abs() + 1e-10)
                depth_ok = (d_sampled > 0) & (rel_diff < self.rel_d_thresh)

                if depth_ok.any():
                    K_src_inv_t = torch.inverse(K_src_t)
                    xi_ok = xi[depth_ok].float()
                    yi_ok = yi[depth_ok].float()
                    d_src_ok = d_sampled[depth_ok]

                    pix_src_h = torch.stack([
                        xi_ok,
                        yi_ok,
                        torch.ones_like(xi_ok),
                    ], dim=0)
                    pts_sc2 = (K_src_inv_t @ pix_src_h) * d_src_ok.unsqueeze(0)
                    pts_w2 = (R_src_t.T @ (pts_sc2 - t_src_t.unsqueeze(1))).T

                    pts_rc2 = (R_ref_t @ pts_w2.T).T + t_ref_t
                    valid_bk = pts_rc2[:, 2] > 0
                    pixel_ok = torch.zeros_like(depth_ok)

                    if valid_bk.any():
                        proj_bk = (K_ref_t @ pts_rc2[valid_bk].T).T
                        px_bk = proj_bk[:, :2] / proj_bk[:, 2:3]

                        orig_x = xs_t[valid][depth_ok][valid_bk].float()
                        orig_y = ys_t[valid][depth_ok][valid_bk].float()
                        pixel_err = torch.linalg.norm(
                            px_bk - torch.stack([orig_x, orig_y], dim=1),
                            dim=1,
                        )
                        pixel_ok_vals = pixel_err < self.pix_thresh

                        ok_indices = torch.where(depth_ok)[0][valid_bk][pixel_ok_vals]
                        pixel_ok[ok_indices] = True

                    depth_ok = depth_ok & pixel_ok

                orig_ys = ys_t[valid][depth_ok]
                orig_xs = xs_t[valid][depth_ok]
                d_agree = d_exp[depth_ok]

                # Scatter-add
                idx_flat = orig_ys * W + orig_xs
                consistent_count.view(-1).scatter_add_(0, idx_flat, torch.ones_like(idx_flat, dtype=torch.int32))
                depth_sum.view(-1).scatter_add_(0, idx_flat, d_agree)

            enough = consistent_count >= self.min_views
            depth_fused = torch.where(
                enough,
                depth_sum / consistent_count.float().clamp(1),
                torch.zeros_like(depth_sum),
            ).cpu().numpy().astype(np.float32)

            conf_fused = (consistent_count.float() / max(len(ids) - 1, 1)).clamp(0, 1)
            conf_fused = torch.where(enough, conf_fused, torch.zeros_like(conf_fused))
            conf_fused = conf_fused.cpu().numpy().astype(np.float32)

            dm_fused = DepthMap(image_id=ref_id, depth=depth_fused, confidence=conf_fused)
            fused[ref_id] = dm_fused
            reconstruction.fused_depth[ref_id] = dm_fused

            print(f"  [img {ref_id:03d}] {ref_dm.valid_count()} → {dm_fused.valid_count()} "
                  f"valid pixels")

        return fused


# ============================================================
#  PIPELINE STEP FUNCTION
# ============================================================

def fuse_depth_maps(
    reconstruction: MVSReconstruction,
    fusion: Optional[DepthFusionBase] = None,
) -> Dict[int, DepthMap]:
    """
    Step 3 — Fuse depth maps via multi-view geometric consistency.

    Parameters
    ----------
    reconstruction : MVSReconstruction (fused_depth populated in-place)
    fusion         : DepthFusionBase (default: GPU if available, else CPU)

    Returns
    -------
    dict image_id → fused DepthMap
    """
    if fusion is None:
        if ACCELERATOR.has_cuda or ACCELERATOR.has_mps:
            fusion = GeometricConsistencyFusionGPU()
        else:
            fusion = GeometricConsistencyFusion()

    return fusion.fuse(reconstruction)
