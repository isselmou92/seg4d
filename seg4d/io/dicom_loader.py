"""DICOM parsing utilities.

These helpers do *not* mutate any DICOM files; they only read tags and pixel
arrays. All geometry-aware reads delegate to SimpleITK so the resulting NIfTI
volumes have correct origin / direction / spacing.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Sequence

import numpy as np
import pydicom
import SimpleITK as sitk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _list_dicom_files(dicom_dir: str) -> list[str]:
    """Return file names inside ``dicom_dir`` that look like DICOM files.

    Skips compressed archives that the source data sometimes ships alongside
    the unpacked DICOMs (``*.7z``, ``*.zip``).
    """
    return [
        f for f in os.listdir(dicom_dir)
        if not f.endswith((".7z", ".zip"))
        and os.path.isfile(os.path.join(dicom_dir, f))
    ]


# ---------------------------------------------------------------------------
# Geometry-correct conversion
# ---------------------------------------------------------------------------

def convert_dicom_series_with_sitk(dicom_file_paths: Sequence[str]) -> sitk.Image:
    """Read a list of DICOM slices into a single ``SimpleITK.Image``.

    Slices are first sorted along the slice-normal direction (the cross
    product of the row and column orientation vectors from
    ``ImageOrientationPatient``) before being handed to SimpleITK's series
    reader. This guarantees a geometrically correct stacking even when the
    files are not pre-sorted by ``InstanceNumber``.

    Parameters
    ----------
    dicom_file_paths
        Absolute paths to the DICOM files belonging to a single 3D volume.

    Returns
    -------
    SimpleITK.Image
        The volume with proper origin, direction and spacing.
    """
    dicom_file_paths = list(dicom_file_paths)
    if len(dicom_file_paths) >= 2:
        ds0 = pydicom.dcmread(dicom_file_paths[0], stop_before_pixels=True)
        iop = np.array(ds0.ImageOrientationPatient, dtype=np.float64)
        normal = np.cross(iop[:3], iop[3:])

        projections = []
        for fpath in dicom_file_paths:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True)
            ipp = np.array(ds.ImagePositionPatient, dtype=np.float64)
            projections.append((float(np.dot(ipp, normal)), fpath))
        projections.sort()
        dicom_file_paths = [p[1] for p in projections]

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(dicom_file_paths)
    return reader.Execute()


# ---------------------------------------------------------------------------
# Volumetric (3D + time) sequences
# ---------------------------------------------------------------------------

def parse_volumetric_dicoms(dicom_dir: str) -> tuple[dict[int, list[tuple[float, str, "pydicom.Dataset"]]], int]:
    """Group volumetric DICOM files by timepoint.

    Files are read with ``stop_before_pixels=True`` for memory efficiency.
    Each timepoint's slice list is sorted by ``SliceLocation``.

    Parameters
    ----------
    dicom_dir
        Directory containing the DICOM files for one volumetric 4D series.

    Returns
    -------
    groups : dict[int, list[tuple[float, str, pydicom.Dataset]]]
        ``{TemporalPositionIdentifier: [(slice_location, file_path, ds), ...]}``
    num_slices : int
        Number of unique slice locations per timepoint (assumed constant).
    """
    print(f"  Parsing volumetric DICOMs from: {dicom_dir}")
    groups: dict[int, list[tuple[float, str, pydicom.Dataset]]] = defaultdict(list)

    for fname in _list_dicom_files(dicom_dir):
        fpath = os.path.join(dicom_dir, fname)
        ds = pydicom.dcmread(fpath, stop_before_pixels=True)
        tp = int(getattr(ds, "TemporalPositionIdentifier", 0))
        sl = float(getattr(ds, "SliceLocation", 0))
        groups[tp].append((sl, fpath, ds))

    for tp in groups:
        groups[tp].sort(key=lambda x: x[0])

    first_key = next(iter(sorted(groups.keys())))
    num_slices = len(groups[first_key])
    num_timepoints = len(groups)
    print(f"  Found {num_timepoints} timepoints x {num_slices} slices "
          f"= {num_timepoints * num_slices} files")
    return dict(groups), num_slices


# ---------------------------------------------------------------------------
# 2D dynamic (single-slice + time) sequences
# ---------------------------------------------------------------------------

def parse_2d_dynamic_dicoms(dicom_dir: str) -> tuple[list[tuple[int, "pydicom.Dataset"]], dict]:
    """Read all frames of a 2D dynamic series into memory and order by time.

    Pixel data *is* loaded here (the dynamic sequences are small enough that
    keeping the frames in memory simplifies the NIfTI/JPEG export step).

    Parameters
    ----------
    dicom_dir
        Directory containing the DICOM files for one 2D dynamic series.

    Returns
    -------
    frames : list[tuple[int, pydicom.Dataset]]
        ``[(InstanceNumber, ds), ...]`` sorted by ``InstanceNumber``.
    geometry : dict
        Cross-section geometry of the first frame, used by the reslicer to
        derive seed masks from a 3D volumetric segmentation.
    """
    print(f"  Parsing 2D dynamic DICOMs from: {dicom_dir}")
    frames: list[tuple[int, pydicom.Dataset]] = []

    for fname in _list_dicom_files(dicom_dir):
        ds = pydicom.dcmread(os.path.join(dicom_dir, fname))
        inst = int(getattr(ds, "InstanceNumber", 0))
        frames.append((inst, ds))

    frames.sort(key=lambda x: x[0])

    ds0 = frames[0][1]
    geometry = {
        "ImagePositionPatient": np.array(ds0.ImagePositionPatient, dtype=np.float64).tolist(),
        "ImageOrientationPatient": np.array(ds0.ImageOrientationPatient, dtype=np.float64).tolist(),
        "PixelSpacing": np.array(ds0.PixelSpacing, dtype=np.float64).tolist(),
        "SliceThickness": float(getattr(ds0, "SliceThickness", 1.0)),
        "Rows": int(ds0.Rows),
        "Columns": int(ds0.Columns),
        "SeriesDescription": str(getattr(ds0, "SeriesDescription", "")),
    }
    print(f"  Found {len(frames)} frames, {geometry['Rows']}x{geometry['Columns']}")
    return frames, geometry
