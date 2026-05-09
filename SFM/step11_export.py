"""
step11_export.py
----------------
Step 11 — Sparse Point Cloud Export

Responsibilities:
  - Write the final sparse point cloud and camera poses to disk
  - Support multiple output formats

Swappable strategies / formats:
  - PLYExporter      — Stanford PLY with per-point RGB colour  (MeshLab, CloudCompare)
  - COLMAPExporter   — COLMAP text format (cameras.txt, images.txt, points3D.txt)
  - NVMExporter      — VisualSFM NVM format
  - BundlerExporter  — Bundler .out format (input to PMVS/CMVS)
"""

import os
import struct
import numpy as np
import cv2
from abc import ABC, abstractmethod
from typing import Optional

from sfm_types import Reconstruction


# ============================================================
#  INTERFACE
# ============================================================

class ExporterBase(ABC):
    @abstractmethod
    def export(self, reconstruction: Reconstruction, output_dir: str) -> str:
        """
        Write the reconstruction to `output_dir`.

        Returns
        -------
        path : str — path to the primary output file or directory
        """


# ============================================================
#  HELPERS
# ============================================================

def _rot_to_rvec(R):
    rvec, _ = cv2.Rodrigues(R)
    return rvec.ravel()


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ============================================================
#  CONCRETE EXPORTERS
# ============================================================

class PLYExporter(ExporterBase):
    """
    Export sparse point cloud as a binary PLY file.
    Each point carries X Y Z R G B.
    Camera centres are optionally written as a second PLY.
    """

    def __init__(self, write_cameras: bool = True):
        self.write_cameras = write_cameras

    def export(self, reconstruction: Reconstruction, output_dir: str) -> str:
        _ensure_dir(output_dir)
        ply_path = os.path.join(output_dir, "sparse_cloud.ply")

        points = list(reconstruction.points3d.values())
        n = len(points)

        try:
            with open(ply_path, "wb") as f:
                # Header
                header = (
                    "ply\n"
                    "format binary_little_endian 1.0\n"
                    f"element vertex {n}\n"
                    "property float x\n"
                    "property float y\n"
                    "property float z\n"
                    "property uchar red\n"
                    "property uchar green\n"
                    "property uchar blue\n"
                    "end_header\n"
                )
                f.write(header.encode("ascii"))

                for p in points:
                    x, y, z = p.xyz.astype(np.float32)
                    r, g, b  = p.color[:3].astype(np.uint8)
                    f.write(struct.pack("<fffBBB", x, y, z, r, g, b))
            print(f"[Step 11] PLY  → {ply_path}  ({n} points)")
        except IOError as e:
            print(f"[Step 11] ERROR writing PLY file {ply_path}: {e}")
            raise

        if self.write_cameras:
            cam_path = os.path.join(output_dir, "cameras.ply")
            self._write_camera_ply(reconstruction, cam_path)

        return ply_path

    def _write_camera_ply(self, reconstruction: Reconstruction, path: str):
        centres = []
        for ri in reconstruction.registered_images.values():
            C = -ri.R.T @ ri.t
            centres.append(C)
        n = len(centres)

        try:
            with open(path, "wb") as f:
                header = (
                    "ply\n"
                    "format binary_little_endian 1.0\n"
                    f"element vertex {n}\n"
                    "property float x\n"
                    "property float y\n"
                    "property float z\n"
                    "property uchar red\n"
                    "property uchar green\n"
                    "property uchar blue\n"
                    "end_header\n"
                )
                f.write(header.encode("ascii"))
                for C in centres:
                    x, y, z = C.astype(np.float32)
                    f.write(struct.pack("<fffBBB", x, y, z, 255, 165, 0))  # orange
            print(f"           cameras → {path}  ({n} cameras)")
        except IOError as e:
            print(f"[Step 11] ERROR writing camera PLY file {path}: {e}")
            raise


class COLMAPExporter(ExporterBase):
    """
    Export in COLMAP text format.
    Produces:
      sparse/
        cameras.txt   — intrinsics
        images.txt    — poses + feature observations
        points3D.txt  — 3-D points + tracks
    """

    def export(self, reconstruction: Reconstruction, output_dir: str) -> str:
        sparse_dir = os.path.join(output_dir, "sparse")
        _ensure_dir(sparse_dir)

        self._write_cameras(reconstruction, sparse_dir)
        self._write_images(reconstruction, sparse_dir)
        self._write_points3d(reconstruction, sparse_dir)

        print(f"[Step 11] COLMAP text format → {sparse_dir}")
        return sparse_dir

    def _write_cameras(self, reconstruction, sparse_dir):
        path = os.path.join(sparse_dir, "cameras.txt")
        try:
            with open(path, "w") as f:
                f.write("# Camera list with one line of data per camera:\n")
                f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
                seen = set()
                for img_id, cam in reconstruction.cameras.items():
                    if cam.camera_id in seen:
                        continue
                    seen.add(cam.camera_id)
                    fx = cam.K[0, 0]; fy = cam.K[1, 1]
                    cx = cam.K[0, 2]; cy = cam.K[1, 2]
                    k1 = cam.dist_coeffs[0] if len(cam.dist_coeffs) > 0 else 0.0
                    k2 = cam.dist_coeffs[1] if len(cam.dist_coeffs) > 1 else 0.0
                    f.write(f"{cam.camera_id} OPENCV {cam.width} {cam.height} "
                            f"{fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f} {k1:.6f} {k2:.6f}\n")
        except IOError as e:
            print(f"[Step 11] ERROR writing cameras.txt: {e}")
            raise

    def _write_images(self, reconstruction, sparse_dir):
        path = os.path.join(sparse_dir, "images.txt")
        try:
            with open(path, "w") as f:
                f.write("# Image list with two lines of data per image:\n")
                f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
                f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

                # Build observation map: (img_id, kp_idx) → point3d_id
                obs_map = {}
                for pid, p3d in reconstruction.points3d.items():
                    for (obs_id, kp_idx) in p3d.track:
                        obs_map[(obs_id, kp_idx)] = pid

                for img_id, ri in reconstruction.registered_images.items():
                    R = ri.R
                    t = ri.t
                    # Rotation matrix → quaternion
                    qw, qx, qy, qz = _R_to_quat(R)
                    name = os.path.basename(reconstruction.image_paths[img_id])
                    # Validate camera exists for this image
                    if img_id not in reconstruction.cameras:
                        print(f"[Step 11] WARNING: image_id {img_id} has no camera intrinsics, skipping")
                        continue
                    cam_id = reconstruction.cameras[img_id].camera_id
                    f.write(f"{img_id} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} "
                            f"{t[0]:.8f} {t[1]:.8f} {t[2]:.8f} {cam_id} {name}\n")

                    kp = reconstruction.keypoints.get(img_id)
                    obs_line = []
                    if kp is not None:
                        for kp_idx in range(len(kp.points)):
                            pid = obs_map.get((img_id, kp_idx), -1)
                            x, y = kp.points[kp_idx]
                            obs_line.append(f"{x:.2f} {y:.2f} {pid}")
                    f.write(" ".join(obs_line) + "\n")
        except IOError as e:
            print(f"[Step 11] ERROR writing images.txt: {e}")
            raise

    def _write_points3d(self, reconstruction, sparse_dir):
        path = os.path.join(sparse_dir, "points3D.txt")
        try:
            with open(path, "w") as f:
                f.write("# 3D point list with one line of data per point:\n")
                f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
                for pid, p3d in reconstruction.points3d.items():
                    x, y, z = p3d.xyz
                    r, g, b = p3d.color[:3]
                    track_str = " ".join(f"{obs_id} {kp_idx}" for obs_id, kp_idx in p3d.track)
                    f.write(f"{pid} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} "
                            f"{p3d.error:.4f} {track_str}\n")
        except IOError as e:
            print(f"[Step 11] ERROR writing points3D.txt: {e}")
            raise


class NVMExporter(ExporterBase):
    """
    Export in VisualSFM NVM format (input to PMVS/CMVS for MVS).
    """

    def export(self, reconstruction: Reconstruction, output_dir: str) -> str:
        _ensure_dir(output_dir)
        nvm_path = os.path.join(output_dir, "reconstruction.nvm")

        try:
            with open(nvm_path, "w") as f:
                f.write("NVM_V3\n\n")

                # Cameras
                reg = reconstruction.registered_images
                f.write(f"{len(reg)}\n")
                for img_id, ri in reg.items():
                    # Validate camera exists for this image
                    if img_id not in reconstruction.cameras:
                        print(f"[Step 11] WARNING: image_id {img_id} has no camera intrinsics, skipping")
                        continue
                    cam = reconstruction.cameras[img_id]
                    name = os.path.basename(reconstruction.image_paths[img_id])
                    fx = cam.K[0, 0]
                    qw, qx, qy, qz = _R_to_quat(ri.R)
                    tx, ty, tz = ri.t
                    # NVM radial distortion (k1 only)
                    k1 = float(cam.dist_coeffs[0]) if len(cam.dist_coeffs) > 0 else 0.0
                    f.write(f"{name} {fx:.4f} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} "
                            f"{tx:.8f} {ty:.8f} {tz:.8f} {k1:.8f} 0\n")

                f.write("\n")

                # Points
                pts = list(reconstruction.points3d.values())
                f.write(f"{len(pts)}\n")
                for p3d in pts:
                    x, y, z = p3d.xyz
                    r, g, b = p3d.color[:3]
                    # Track
                    track_entries = []
                    for (obs_id, kp_idx) in p3d.track:
                        kp = reconstruction.keypoints.get(obs_id)
                        if kp is None or kp_idx >= len(kp.points):
                            continue
                        px, py = kp.points[kp_idx]
                        track_entries.append(f"{obs_id} {kp_idx} {px:.2f} {py:.2f}")
                    track_str = " ".join(track_entries)
                    f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b} "
                            f"{len(track_entries)} {track_str}\n")
            print(f"[Step 11] NVM → {nvm_path}")
        except IOError as e:
            print(f"[Step 11] ERROR writing NVM file {nvm_path}: {e}")
            raise
        return nvm_path


def _R_to_quat(R: np.ndarray):
    """Convert 3×3 rotation matrix to quaternion (w, x, y, z)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return w, x, y, z


# ============================================================
#  PIPELINE STEP FUNCTION
# ============================================================

def export_reconstruction(
    reconstruction: Reconstruction,
    output_dir: str,
    exporter: Optional[ExporterBase] = None,
) -> str:
    """
    Step 11 — Export the sparse point cloud and camera poses.

    Parameters
    ----------
    reconstruction : final SfM state
    output_dir     : directory to write outputs
    exporter       : ExporterBase implementation (default: PLYExporter)

    Returns
    -------
    output_path : str — path to primary output file/directory
    """
    if exporter is None:
        exporter = PLYExporter()

    # Validate reconstruction state
    if not reconstruction.registered_images:
        raise ValueError("No registered images in reconstruction")
    if not reconstruction.points3d:
        raise ValueError("No 3D points in reconstruction")

    n_cams = len(reconstruction.registered_images)
    n_pts  = len(reconstruction.points3d)
    print(f"[Step 11] Exporting: {n_cams} cameras, {n_pts} 3-D points  "
          f"(format: {exporter.__class__.__name__})")

    return exporter.export(reconstruction, output_dir)
