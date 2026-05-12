"""NIfTI and JPEG writers used by the conversion step.

Volumetric per-timepoint NIfTIs are written through SimpleITK so origin,
direction and spacing are preserved exactly. 2D dynamic sequences are stacked
along a synthetic time axis with 1mm spacing; a directory of per-frame JPEGs
is written alongside, ready to be consumed as a "video" by the SAM3 / MedSAM2
predictors.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import SimpleITK as sitk
from PIL import Image

from .dicom_loader import convert_dicom_series_with_sitk


def convert_volumetric_to_nifti(
    groups: dict[int, list[tuple[float, str, "object"]]],
    num_slices: int,
    output_dir: str,
) -> list[str]:
    """Write one NIfTI per timepoint of a volumetric 4D acquisition.

    Parameters
    ----------
    groups
        Output of :func:`seg4d.io.dicom_loader.parse_volumetric_dicoms`.
    num_slices
        Expected number of slices per timepoint (asserted as a sanity check).
    output_dir
        Destination directory; created if missing.

    Returns
    -------
    list[str]
        Paths to the written NIfTI files in ascending timepoint order.
    """
    os.makedirs(output_dir, exist_ok=True)
    nifti_paths: list[str] = []

    for tp in sorted(groups.keys()):
        slices = groups[tp]
        assert len(slices) == num_slices, (
            f"Timepoint {tp}: expected {num_slices} slices, got {len(slices)}"
        )

        dcm_file_paths = [fpath for (_sl, fpath, _ds) in slices]
        sitk_img = convert_dicom_series_with_sitk(dcm_file_paths)

        out_path = os.path.join(output_dir, f"timepoint_{tp:02d}.nii.gz")
        sitk.WriteImage(sitk_img, out_path)
        nifti_paths.append(out_path)

    print(f"  Saved {len(nifti_paths)} volumetric NIfTIs to {output_dir}")
    return nifti_paths


def _normalize_to_uint8(raw: np.ndarray) -> np.ndarray:
    """Min-max normalise a single 2D frame to ``uint8`` for JPEG export."""
    pmin, pmax = float(raw.min()), float(raw.max())
    if pmax > pmin:
        return ((raw - pmin) / (pmax - pmin) * 255).astype(np.uint8)
    return np.zeros_like(raw, dtype=np.uint8)


def convert_2d_dynamic_to_nifti(
    frames: Sequence[tuple[int, "object"]],
    output_dir: str,
    seq_name: str,
) -> tuple[str, str]:
    """Convert a 2D dynamic acquisition to a 3D NIfTI plus a JPEG frame dir.

    The resulting NIfTI has shape ``(rows, cols, num_frames)``. Origin,
    in-plane direction and pixel spacing are taken from the first DICOM frame
    via SimpleITK; the time axis is mapped onto the slice-normal direction
    with 1mm spacing so the volume opens at the correct anatomical position
    in 3D Slicer or RayStation.

    Parameters
    ----------
    frames
        Output of :func:`seg4d.io.dicom_loader.parse_2d_dynamic_dicoms`.
    output_dir
        Destination directory for the NIfTI and the ``<seq>_frames_jpg`` dir.
    seq_name
        Sequence name used as a filename prefix.

    Returns
    -------
    nifti_path : str
        Path to the written ``<seq>_frames.nii.gz``.
    jpeg_dir : str
        Path to the directory containing one ``00000.jpg``, ``00001.jpg``, ...
        per frame.
    """
    os.makedirs(output_dir, exist_ok=True)
    jpeg_dir = os.path.join(output_dir, f"{seq_name}_frames_jpg")
    os.makedirs(jpeg_dir, exist_ok=True)

    ds0 = frames[0][1]
    rows, cols = int(ds0.Rows), int(ds0.Columns)
    num_frames = len(frames)
    rescale_slope = float(getattr(ds0, "RescaleSlope", 1.0))
    rescale_intercept = float(getattr(ds0, "RescaleIntercept", 0.0))

    ref_sitk = sitk.ReadImage(ds0.filename)
    sitk_origin = ref_sitk.GetOrigin()
    sitk_spacing = ref_sitk.GetSpacing()
    sitk_direction = ref_sitk.GetDirection()

    volume_zyx = np.zeros((num_frames, rows, cols), dtype=np.float32)
    for i, (_inst, ds) in enumerate(frames):
        raw = ds.pixel_array.astype(np.float32)
        volume_zyx[i] = raw * rescale_slope + rescale_intercept

        normalized = _normalize_to_uint8(raw)
        Image.fromarray(normalized, mode="L").convert("RGB").save(
            os.path.join(jpeg_dir, f"{i:05d}.jpg"), quality=95
        )

    sitk_vol = sitk.GetImageFromArray(volume_zyx)
    sitk_vol.SetOrigin(sitk_origin)
    sitk_vol.SetSpacing((sitk_spacing[0], sitk_spacing[1], 1.0))
    sitk_vol.SetDirection(sitk_direction)

    nifti_path = os.path.join(output_dir, f"{seq_name}_frames.nii.gz")
    sitk.WriteImage(sitk_vol, nifti_path)

    print(f"  Saved {seq_name}: {num_frames} frames NIfTI + JPEG dir")
    return nifti_path, jpeg_dir
