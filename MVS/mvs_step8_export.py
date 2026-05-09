"""
mvs_step8_export.py
--------------------
Step 8 — Dense Point Cloud Export

Swappable strategies
  - PLYExporter     — Stanford PLY (binary or ASCII), with normals + colours
  - LASExporter     — ASPRS LAS / LAZ (requires laspy)
  - PCDExporter     — PCL PCD format (ASCII or binary)
  - E57Exporter     — ASTM E57 placeholder (requires pye57)
  - MultiExporter   — write multiple formats in one call
"""

from __future__ import annotations
import os
import struct
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional

from mvs_types import DensePointCloud, MVSReconstruction


# ============================================================
#  INTERFACE
# ============================================================

class DenseExporterBase(ABC):
    @abstractmethod
    def export(self, cloud: DensePointCloud, output_dir: str) -> str:
        """Write cloud to output_dir. Returns path of primary output file."""


# ============================================================
#  HELPERS
# ============================================================

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ============================================================
#  CONCRETE EXPORTERS
# ============================================================

class PLYExporter(DenseExporterBase):
    """
    Export as Stanford PLY.
    Supports per-point: XYZ, RGB colour, surface normals (NX NY NZ), confidence.
    Binary little-endian by default (much smaller and faster than ASCII).
    """

    def __init__(self, binary: bool = True, include_normals: bool = True,
                 include_confidence: bool = True):
        self.binary = binary
        self.include_normals = include_normals
        self.include_confidence = include_confidence

    def export(self, cloud: DensePointCloud, output_dir: str) -> str:
        _ensure_dir(output_dir)
        path = os.path.join(output_dir, "dense_cloud.ply")

        pts  = cloud.points
        cols = cloud.colors
        has_normals = self.include_normals and cloud.normals is not None
        has_conf    = self.include_confidence and cloud.confidences is not None
        n = len(pts)

        # Build header
        props = [
            "property float x\n", "property float y\n", "property float z\n",
            "property uchar red\n", "property uchar green\n", "property uchar blue\n",
        ]
        if has_normals:
            props += ["property float nx\n", "property float ny\n", "property float nz\n"]
        if has_conf:
            props += ["property float confidence\n"]

        fmt = "binary_little_endian" if self.binary else "ascii"
        header = (
            f"ply\nformat {fmt} 1.0\n"
            f"element vertex {n}\n"
            + "".join(props) +
            "end_header\n"
        )

        with open(path, "wb") as f:
            f.write(header.encode("ascii"))

            if self.binary:
                pack_fmt = "<fffBBB"
                pack_extra = ""
                if has_normals:
                    pack_fmt += "fff"
                if has_conf:
                    pack_fmt += "f"

                for i in range(n):
                    x, y, z = pts[i].astype(np.float32)
                    r, g, b  = cols[i].astype(np.uint8)
                    row = [x, y, z, r, g, b]
                    if has_normals:
                        row.extend(cloud.normals[i].astype(np.float32).tolist())
                    if has_conf:
                        row.append(float(cloud.confidences[i]))
                    f.write(struct.pack(pack_fmt, *row))
            else:
                for i in range(n):
                    x, y, z = pts[i]
                    r, g, b  = cols[i]
                    line = f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}"
                    if has_normals:
                        nx, ny, nz = cloud.normals[i]
                        line += f" {nx:.6f} {ny:.6f} {nz:.6f}"
                    if has_conf:
                        line += f" {cloud.confidences[i]:.4f}"
                    f.write((line + "\n").encode("ascii"))

        print(f"[Step 8] PLY → {path}  ({n} points, binary={self.binary})")
        return path


class PCDExporter(DenseExporterBase):
    """
    Export as PCL PCD format.
    Supports XYZ + RGB (packed as float) + normals.
    ASCII mode for readability, binary for performance.
    """

    def __init__(self, binary: bool = False, include_normals: bool = True):
        self.binary = binary
        self.include_normals = include_normals

    def export(self, cloud: DensePointCloud, output_dir: str) -> str:
        _ensure_dir(output_dir)
        path = os.path.join(output_dir, "dense_cloud.pcd")
        n    = len(cloud.points)
        has_n = self.include_normals and cloud.normals is not None

        fields = "x y z rgb"
        sizes  = "4 4 4 4"
        types  = "F F F F"
        count  = "1 1 1 1"
        if has_n:
            fields += " normal_x normal_y normal_z"
            sizes  += " 4 4 4"
            types  += " F F F"
            count  += " 1 1 1"

        header_lines = [
            "# .PCD v0.7 - Point Cloud Data file format",
            f"VERSION 0.7",
            f"FIELDS {fields}",
            f"SIZE {sizes}",
            f"TYPE {types}",
            f"COUNT {count}",
            f"WIDTH {n}",
            "HEIGHT 1",
            "VIEWPOINT 0 0 0 1 0 0 0",
            f"POINTS {n}",
            "DATA " + ("binary" if self.binary else "ascii"),
        ]
        header = "\n".join(header_lines) + "\n"

        def pack_rgb(r, g, b):
            """Pack RGB as a float32 (PCL convention)."""
            val = (int(r) << 16) | (int(g) << 8) | int(b)
            return struct.unpack("<f", struct.pack("<I", val))[0]

        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            for i in range(n):
                x, y, z = cloud.points[i].astype(np.float32)
                rgb = pack_rgb(*cloud.colors[i])
                if self.binary:
                    row = struct.pack("<ffff", x, y, z, rgb)
                    if has_n:
                        nx, ny, nz = cloud.normals[i].astype(np.float32)
                        row += struct.pack("<fff", nx, ny, nz)
                    f.write(row)
                else:
                    line = f"{x:.6f} {y:.6f} {z:.6f} {rgb}"
                    if has_n:
                        nx, ny, nz = cloud.normals[i]
                        line += f" {nx:.6f} {ny:.6f} {nz:.6f}"
                    f.write((line + "\n").encode("ascii"))

        print(f"[Step 8] PCD → {path}  ({n} points, binary={self.binary})")
        return path


class LASExporter(DenseExporterBase):
    """
    Export as ASPRS LAS 1.4 point cloud.
    Requires: pip install laspy
    """

    def __init__(self, include_color: bool = True):
        self.include_color = include_color

    def export(self, cloud: DensePointCloud, output_dir: str) -> str:
        _ensure_dir(output_dir)
        path = os.path.join(output_dir, "dense_cloud.las")
        try:
            import laspy
        except ImportError:
            print("[LASExporter] laspy not installed. Falling back to PLY.")
            return PLYExporter().export(cloud, output_dir)

        header = laspy.LasHeader(point_format=2, version="1.4")
        las = laspy.LasData(header=header)

        pts = cloud.points
        # Scale and offset to preserve float64 precision in int32 storage
        scale  = 0.0001
        offset = pts.mean(axis=0)
        las.header.offsets = offset
        las.header.scales  = np.array([scale, scale, scale])

        las.x = pts[:, 0]
        las.y = pts[:, 1]
        las.z = pts[:, 2]

        if self.include_color:
            # LAS stores colours as uint16
            las.red   = (cloud.colors[:, 0].astype(np.uint16)) * 256
            las.green = (cloud.colors[:, 1].astype(np.uint16)) * 256
            las.blue  = (cloud.colors[:, 2].astype(np.uint16)) * 256

        las.write(path)
        print(f"[Step 8] LAS → {path}  ({len(pts)} points)")
        return path


class E57Exporter(DenseExporterBase):
    """
    Export as ASTM E57 format.
    Requires: pip install pye57
    """

    def export(self, cloud: DensePointCloud, output_dir: str) -> str:
        _ensure_dir(output_dir)
        path = os.path.join(output_dir, "dense_cloud.e57")
        try:
            import pye57
        except ImportError:
            print("[E57Exporter] pye57 not installed. Falling back to PLY.")
            return PLYExporter().export(cloud, output_dir)

        e57 = pye57.E57(path, mode="w")
        data = {
            "cartesianX": cloud.points[:, 0].astype(np.float64),
            "cartesianY": cloud.points[:, 1].astype(np.float64),
            "cartesianZ": cloud.points[:, 2].astype(np.float64),
        }
        if cloud.colors is not None:
            data["colorRed"]   = cloud.colors[:, 0]
            data["colorGreen"] = cloud.colors[:, 1]
            data["colorBlue"]  = cloud.colors[:, 2]
        if cloud.normals is not None:
            data["normalX"] = cloud.normals[:, 0].astype(np.float32)
            data["normalY"] = cloud.normals[:, 1].astype(np.float32)
            data["normalZ"] = cloud.normals[:, 2].astype(np.float32)
        e57.write_scan_raw(data)
        print(f"[Step 8] E57 → {path}  ({len(cloud)} points)")
        return path


class MultiExporter(DenseExporterBase):
    """Write multiple formats in one call."""

    def __init__(self, exporters: List[DenseExporterBase]):
        self.exporters = exporters

    def export(self, cloud: DensePointCloud, output_dir: str) -> str:
        paths = []
        for exp in self.exporters:
            paths.append(exp.export(cloud, output_dir))
        return paths[0] if paths else output_dir


# ============================================================
#  PIPELINE STEP FUNCTION
# ============================================================

def export_dense_cloud(
    reconstruction: MVSReconstruction,
    output_dir: str,
    exporter: Optional[DenseExporterBase] = None,
) -> str:
    """
    Step 8 — Export the dense point cloud to disk.

    Parameters
    ----------
    reconstruction : MVSReconstruction
    output_dir     : directory to write output files
    exporter       : DenseExporterBase (default: PLYExporter)

    Returns
    -------
    path : str — primary output file path
    """
    if exporter is None:
        exporter = PLYExporter(binary=True, include_normals=True)

    cloud = reconstruction.dense_cloud
    n = len(cloud)
    has_n = cloud.normals is not None
    print(f"[Step 8] Exporting dense cloud: {n} points, normals={has_n}  "
          f"({exporter.__class__.__name__})")

    return exporter.export(cloud, output_dir)
