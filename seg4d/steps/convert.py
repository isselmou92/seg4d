"""Step 1 - DICOM to NIfTI / JPEG conversion."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ..constants import DYNAMIC_SEQUENCES, VOLUMETRIC_SEQUENCE
from ..io import (
    convert_2d_dynamic_to_nifti,
    convert_volumetric_to_nifti,
    parse_2d_dynamic_dicoms,
    parse_volumetric_dicoms,
    save_metadata,
)


def run_convert(data_dir: str) -> dict[str, Any]:
    """Parse all configured DICOM sequences and convert them to NIfTI.

    Side effects
    ------------
    * Writes per-timepoint NIfTIs for the volumetric sequence under
      ``<data_dir>/nifti/<seq>/timepoint_XX.nii.gz``.
    * Writes a 3D NIfTI (rows x cols x time) and a JPEG-frames directory
      for every dynamic sequence under
      ``<data_dir>/nifti/<seq>/<seq>_frames.nii.gz`` and
      ``.../<seq>_frames_jpg/``.
    * Writes ``<data_dir>/nifti/conversion_metadata.json``.
    """
    print("\n" + "=" * 60)
    print("STEP 1: DICOM to NIfTI Conversion")
    print("=" * 60)

    dicom_base = os.path.join(data_dir, "dicom")
    nifti_base = os.path.join(data_dir, "nifti")
    metadata: dict[str, Any] = {}

    # --- Volumetric ---
    vol_dicom_dir = os.path.join(dicom_base, VOLUMETRIC_SEQUENCE)
    if os.path.isdir(vol_dicom_dir):
        print(f"\n[Volumetric] Processing {VOLUMETRIC_SEQUENCE}...")
        groups, num_slices = parse_volumetric_dicoms(vol_dicom_dir)
        vol_nifti_dir = os.path.join(nifti_base, VOLUMETRIC_SEQUENCE)
        nifti_paths = convert_volumetric_to_nifti(groups, num_slices, vol_nifti_dir)
        metadata["volumetric"] = {
            "sequence": VOLUMETRIC_SEQUENCE,
            "num_timepoints": len(nifti_paths),
            "num_slices": num_slices,
            "nifti_dir": vol_nifti_dir,
            "nifti_paths": nifti_paths,
        }

        sample_ds = groups[sorted(groups.keys())[0]][0][2]
        first_tp = sorted(groups.keys())[0]
        metadata["volumetric"]["geometry"] = {
            "ImagePositionPatient": np.array(sample_ds.ImagePositionPatient, dtype=np.float64).tolist(),
            "ImageOrientationPatient": np.array(sample_ds.ImageOrientationPatient, dtype=np.float64).tolist(),
            "PixelSpacing": np.array(sample_ds.PixelSpacing, dtype=np.float64).tolist(),
            "SliceThickness": float(getattr(sample_ds, "SliceThickness", 1.0)),
            "SpacingBetweenSlices": float(getattr(sample_ds, "SpacingBetweenSlices", 1.0)),
            "Rows": int(sample_ds.Rows),
            "Columns": int(sample_ds.Columns),
            "slice_positions": [
                np.array(ds.ImagePositionPatient, dtype=np.float64).tolist()
                for (_sl, _fpath, ds) in groups[first_tp]
            ],
        }
    else:
        print(f"  WARNING: Volumetric sequence {VOLUMETRIC_SEQUENCE} not found at {vol_dicom_dir}")

    # --- Dynamic ---
    metadata["dynamic"] = {}
    for seq_name in DYNAMIC_SEQUENCES:
        dyn_dicom_dir = os.path.join(dicom_base, seq_name)
        if not os.path.isdir(dyn_dicom_dir):
            print(f"\n  WARNING: Dynamic sequence {seq_name} not found, skipping")
            continue

        print(f"\n[Dynamic] Processing {seq_name}...")
        frames, geometry = parse_2d_dynamic_dicoms(dyn_dicom_dir)
        dyn_nifti_dir = os.path.join(nifti_base, seq_name)
        nifti_path, jpeg_dir = convert_2d_dynamic_to_nifti(frames, dyn_nifti_dir, seq_name)
        metadata["dynamic"][seq_name] = {
            "num_frames": len(frames),
            "nifti_path": nifti_path,
            "jpeg_dir": jpeg_dir,
            "geometry": geometry,
        }

    meta_path = save_metadata(data_dir, metadata)
    print(f"\n  Metadata saved to {meta_path}")
    return metadata
