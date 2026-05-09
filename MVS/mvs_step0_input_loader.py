"""
mvs_step0_input_loader.py
--------------------------
Step 0 — Load SfM Output into MVS Reconstruction

Responsibilities:
  - Read camera poses and intrinsics from SfM output
  - Load source images
  - Populate MVSReconstruction.cameras

Swappable strategies
  - COLMAPLoader       — read from COLMAP text/binary sparse directory
  - SfMPipelineLoader  — load directly from our sfm_types.Reconstruction object
  - NVMLoader          — read VisualSFM NVM file
  - ManualLoader       — build from raw numpy arrays (testing / custom pipelines)
"""

from __future__ import annotations
import os
import struct
import numpy as np
import cv2
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from mvs_types import MVSCamera, MVSReconstruction


# ============================================================
#  INTERFACE
# ============================================================

class SfMLoaderBase(ABC):
    @abstractmethod
    def load(self, source, reconstruction: MVSReconstruction) -> List[MVSCamera]:
        """
        Populate reconstruction.cameras and return the camera list.

        Parameters
        ----------
        source         : path string, Reconstruction object, etc. (depends on impl.)
        reconstruction : MVSReconstruction to populate
        """


# ============================================================
#  HELPERS
# ============================================================

def _quat_to_R(qw, qx, qy, qz) -> np.ndarray:
    """Unit quaternion → 3×3 rotation matrix."""
    n = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    if n < 1e-12:
        raise ValueError("Invalid quaternion with near-zero norm")
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float64)


def _load_image_shape(path: str) -> Tuple[int, int]:
    img = cv2.imread(path)
    if img is None:
        return 0, 0
    return img.shape[1], img.shape[0]  # width, height


def _find_colmap_image_root(sparse_dir: str, image_name: str) -> Optional[str]:
    """Best-effort image root inference for COLMAP image names."""
    if os.path.isabs(image_name):
        return os.path.dirname(image_name) if os.path.exists(image_name) else None

    sparse_dir_abs = os.path.abspath(sparse_dir)
    parent = os.path.dirname(sparse_dir_abs)
    grandparent = os.path.dirname(parent)

    # Typical COLMAP layouts plus this project's common image folder names.
    candidates = [
        sparse_dir_abs,
        parent,
        grandparent,
        os.path.join(parent, "images"),
        os.path.join(parent, "image"),
        os.path.join(grandparent, "images"),
        os.path.join(grandparent, "image"),
        os.path.join(grandparent, "House3d"),
    ]

    # Also try one-level sibling directories under the project root.
    try:
        for entry in os.listdir(grandparent):
            p = os.path.join(grandparent, entry)
            if os.path.isdir(p):
                candidates.append(p)
    except OSError:
        pass

    seen = set()
    for root in candidates:
        if not root or root in seen:
            continue
        seen.add(root)
        if os.path.exists(os.path.join(root, image_name)):
            return root
    return None


# ============================================================
#  CONCRETE LOADERS
# ============================================================

class COLMAPLoader(SfMLoaderBase):
    """
    Load from a COLMAP sparse reconstruction (text format).
    Expects a directory containing:
      cameras.txt, images.txt, points3D.txt
    """

    def load(self, source: str, reconstruction: MVSReconstruction) -> List[MVSCamera]:
        sparse_dir = source
        cam_intrinsics = self._read_cameras(os.path.join(sparse_dir, "cameras.txt"))
        cameras = self._read_images(
            os.path.join(sparse_dir, "images.txt"),
            cam_intrinsics,
            reconstruction,
            sparse_dir,
        )
        pts, colors = self._read_points3d(os.path.join(sparse_dir, "points3D.txt"))
        reconstruction.sparse_points = pts
        reconstruction.sparse_colors = colors

        print(f"[Step 0] COLMAP loader: {len(cameras)} cameras, "
              f"{len(pts) if pts is not None else 0} sparse points")
        return cameras

    def _read_cameras(self, path: str) -> dict:
        intrinsics = {}
        with open(path) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                cam_id = int(parts[0])
                model  = parts[1]
                w, h   = int(parts[2]), int(parts[3])
                params = list(map(float, parts[4:]))

                if model in ("PINHOLE", "OPENCV"):
                    fx, fy, cx, cy = params[:4]
                    k1 = params[4] if len(params) > 4 else 0.0
                    k2 = params[5] if len(params) > 5 else 0.0
                elif model in ("SIMPLE_RADIAL", "RADIAL", "SIMPLE_PINHOLE"):
                    f = params[0]; cx, cy = params[1], params[2]
                    fx = fy = f
                    k1 = params[3] if len(params) > 3 else 0.0
                    k2 = 0.0
                else:
                    fx = fy = params[0]
                    cx = w / 2; cy = h / 2
                    k1 = k2 = 0.0

                K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
                dist = np.array([k1, k2, 0, 0, 0], dtype=np.float64)
                intrinsics[cam_id] = (K, dist, w, h)
        return intrinsics

    def _read_images(self, path, intrinsics, reconstruction, sparse_dir: str) -> List[MVSCamera]:
        cameras = []
        with open(path) as f:
            lines = [l.rstrip("\n") for l in f]

        inferred_image_root = None
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith("#"):
                i += 1
                continue

            parts = line.split()
            if len(parts) < 10:
                raise ValueError(f"Malformed COLMAP images.txt header at line {i + 1}: {lines[i]}")

            img_id  = int(parts[0])
            qw,qx,qy,qz = map(float, parts[1:5])
            tx,ty,tz     = map(float, parts[5:8])
            cam_id  = int(parts[8])
            name    = parts[9]
            i += 1

            # COLMAP stores one points2D line after each header; it can be empty.
            if i < len(lines):
                i += 1

            if os.path.isabs(name):
                image_path = name
            else:
                if inferred_image_root is None:
                    inferred_image_root = _find_colmap_image_root(sparse_dir, name)
                base_dir = inferred_image_root if inferred_image_root else sparse_dir
                image_path = os.path.join(base_dir, name)

            R = _quat_to_R(qw, qx, qy, qz)
            t = np.array([tx, ty, tz], dtype=np.float64)
            K, dist, w, h = intrinsics.get(cam_id,
                (np.eye(3), np.zeros(5), 0, 0))

            cam = MVSCamera(
                image_id=img_id, image_path=image_path,
                width=w, height=h,
                K=K, dist_coeffs=dist, R=R, t=t,
            )
            cameras.append(cam)
            reconstruction.cameras[img_id] = cam
        return cameras

    def _read_points3d(self, path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        pts, cols = [], []
        try:
            with open(path) as f:
                for line in f:
                    if line.startswith("#") or not line.strip():
                        continue
                    p = line.split()
                    pts.append([float(p[1]), float(p[2]), float(p[3])])
                    cols.append([int(p[4]), int(p[5]), int(p[6])])
        except FileNotFoundError:
            return None, None
        return np.array(pts, dtype=np.float64), np.array(cols, dtype=np.uint8)


class SfMPipelineLoader(SfMLoaderBase):
    """
    Load directly from our sfm_types.Reconstruction object.
    Zero disk I/O — pass the in-memory SfM result straight to MVS.
    """

    def __init__(self, image_dir: str = ""):
        self.image_dir = image_dir

    def load(self, source, reconstruction: MVSReconstruction) -> List[MVSCamera]:
        """source = sfm_types.Reconstruction instance"""
        sfm_recon = source
        cameras = []

        for img_id, ri in sfm_recon.registered_images.items():
            sfm_cam = sfm_recon.cameras[img_id]
            img_path = sfm_recon.image_paths[img_id]
            if self.image_dir:
                img_path = os.path.join(self.image_dir, os.path.basename(img_path))

            w = sfm_cam.width or 0
            h = sfm_cam.height or 0
            if w == 0:
                w, h = _load_image_shape(img_path)

            cam = MVSCamera(
                image_id=img_id,
                image_path=img_path,
                width=w, height=h,
                K=sfm_cam.K.astype(np.float64),
                dist_coeffs=sfm_cam.dist_coeffs.astype(np.float64),
                R=ri.R.astype(np.float64),
                t=ri.t.astype(np.float64),
            )
            cameras.append(cam)
            reconstruction.cameras[img_id] = cam

        # Sparse point cloud
        if sfm_recon.points3d:
            pts = np.array([p.xyz  for p in sfm_recon.points3d.values()], dtype=np.float64)
            col = np.array([p.color for p in sfm_recon.points3d.values()], dtype=np.uint8)
            reconstruction.sparse_points = pts
            reconstruction.sparse_colors = col

        print(f"[Step 0] SfMPipelineLoader: {len(cameras)} cameras, "
              f"{len(reconstruction.sparse_points) if reconstruction.sparse_points is not None else 0} sparse points")
        return cameras


class NVMLoader(SfMLoaderBase):
    """Load from a VisualSFM .nvm file."""

    def __init__(self, image_dir: str = ""):
        self.image_dir = image_dir

    def load(self, source: str, reconstruction: MVSReconstruction) -> List[MVSCamera]:
        cameras = []
        with open(source) as f:
            lines = [l.strip() for l in f]

        idx = 0
        while idx < len(lines) and not lines[idx]:
            idx += 1
        assert lines[idx].startswith("NVM_V3")
        idx += 1

        while idx < len(lines) and not lines[idx]:
            idx += 1

        n_cams = int(lines[idx]); idx += 1
        for i in range(n_cams):
            p = lines[idx].split(); idx += 1
            name = p[0]; f = float(p[1])
            qw,qx,qy,qz = map(float, p[2:6])
            tx,ty,tz     = map(float, p[6:9])

            img_path = name if os.path.isabs(name) else os.path.join(self.image_dir, name)
            w, h = _load_image_shape(img_path)
            K = np.array([[f,0,w/2],[0,f,h/2],[0,0,1]], dtype=np.float64)
            R = _quat_to_R(qw, qx, qy, qz)
            t = np.array([tx, ty, tz], dtype=np.float64)

            cam = MVSCamera(
                image_id=i, image_path=img_path,
                width=w, height=h,
                K=K, dist_coeffs=np.zeros(5), R=R, t=t,
            )
            cameras.append(cam)
            reconstruction.cameras[i] = cam

        while idx < len(lines) and not lines[idx]:
            idx += 1
        n_pts = int(lines[idx]); idx += 1
        pts, cols = [], []
        for _ in range(n_pts):
            while idx < len(lines) and not lines[idx]:
                idx += 1
            p = lines[idx].split(); idx += 1
            pts.append([float(p[0]), float(p[1]), float(p[2])])
            cols.append([int(p[3]), int(p[4]), int(p[5])])

        reconstruction.sparse_points = np.array(pts, dtype=np.float64)
        reconstruction.sparse_colors = np.array(cols, dtype=np.uint8)

        print(f"[Step 0] NVMLoader: {len(cameras)} cameras, {len(pts)} sparse points")
        return cameras


class ManualLoader(SfMLoaderBase):
    """
    Build reconstruction from raw numpy arrays.
    Convenient for unit tests or custom integrations.
    """

    def __init__(
        self,
        image_paths: List[str],
        Ks: np.ndarray,           # (N,3,3)
        Rs: np.ndarray,           # (N,3,3)
        ts: np.ndarray,           # (N,3)
        dist_coeffs: Optional[np.ndarray] = None,  # (N,5) or None
        sparse_pts: Optional[np.ndarray] = None,
        sparse_colors: Optional[np.ndarray] = None,
    ):
        self.image_paths = image_paths
        self.Ks = Ks; self.Rs = Rs; self.ts = ts
        self.dist_coeffs = dist_coeffs
        self.sparse_pts = sparse_pts
        self.sparse_colors = sparse_colors

    def load(self, source=None, reconstruction: MVSReconstruction = None) -> List[MVSCamera]:
        if reconstruction is None:
            raise ValueError("ManualLoader.load requires a valid reconstruction object")

        cameras = []
        for i, path in enumerate(self.image_paths):
            w, h = _load_image_shape(path)
            dist = self.dist_coeffs[i] if self.dist_coeffs is not None else np.zeros(5)
            cam = MVSCamera(
                image_id=i, image_path=path,
                width=w, height=h,
                K=self.Ks[i], dist_coeffs=dist,
                R=self.Rs[i], t=self.ts[i],
            )
            cameras.append(cam)
            reconstruction.cameras[i] = cam
        reconstruction.sparse_points = self.sparse_pts
        reconstruction.sparse_colors = self.sparse_colors
        print(f"[Step 0] ManualLoader: {len(cameras)} cameras")
        return cameras


# ============================================================
#  PIPELINE STEP FUNCTION
# ============================================================

def load_sfm_result(
    source,
    reconstruction: MVSReconstruction,
    loader: Optional[SfMLoaderBase] = None,
) -> List[MVSCamera]:
    """
    Step 0 — Load SfM output into the MVS reconstruction.

    Parameters
    ----------
    source         : input source (path string or sfm_types.Reconstruction)
    reconstruction : MVSReconstruction to populate
    loader         : SfMLoaderBase implementation
                     (default: COLMAPLoader if source is a str path,
                               SfMPipelineLoader if source is a Reconstruction)

    Returns
    -------
    cameras : list of MVSCamera
    """
    if loader is None:
        if isinstance(source, str):
            loader = COLMAPLoader()
        else:
            loader = SfMPipelineLoader()

    return loader.load(source, reconstruction)
