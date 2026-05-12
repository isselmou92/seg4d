"""Step 4 - reassemble per-timepoint outputs into 4D volumes + summary stats."""

from __future__ import annotations

import json
import os
from typing import Any

import nibabel as nib
import numpy as np

from ..constants import (
    DYNAMIC_SEQUENCES,
    ORGAN_OBJECT_IDS,
    TOTALSEG_LABEL_MAP,
    VOLUMETRIC_SEQUENCE,
)
from ..io import load_metadata


def run_reassemble(data_dir: str, results_dir: str = "results") -> dict[str, Any]:
    """Stack per-timepoint segs into 4D volumes and compute simple statistics.

    Parameters
    ----------
    data_dir
        Patient data directory.
    results_dir
        Directory that holds the dynamic outputs, relative to ``data_dir``
        or absolute. Must match the ``results_dir`` used in the corresponding
        :func:`seg4d.steps.dynamic.run_dynamic` call. Defaults to
        ``"results"``.

    Outputs
    -------
    * ``<results_dir>/<vol_seq>/segmentation_4d.nii.gz`` - 4D label map.
    * ``<results_dir>/<vol_seq>/<organ>_4d.nii.gz``      - per-organ 4D mask.
    * ``<results_dir>/summary.json``                     - voxel counts and volumes
      for the volumetric series, plus mean / coverage statistics for each
      dynamic sequence.

    Returns
    -------
    dict
        The summary dict that was written to ``summary.json``.
    """
    print("\n" + "=" * 60)
    print("STEP 4: Reassemble 4D Segmentation + Summary")
    print("=" * 60)

    metadata = load_metadata(data_dir)
    results_base = (
        results_dir if os.path.isabs(results_dir)
        else os.path.join(data_dir, results_dir)
    )
    summary: dict[str, Any] = {"volumetric": {}, "dynamic": {}}

    if "volumetric" in metadata and "seg_paths" in metadata["volumetric"]:
        seg_paths = metadata["volumetric"]["seg_paths"]
        print(f"\n  Assembling volumetric 4D from {len(seg_paths)} timepoints...")

        first_seg = nib.load(seg_paths[0])
        first_data = np.asanyarray(first_seg.dataobj)
        shape_3d = first_data.shape
        affine = first_seg.affine

        seg_4d = np.zeros((*shape_3d, len(seg_paths)), dtype=np.uint8)
        vol_stats: list[dict[str, Any]] = []
        voxel_vol_mm3 = abs(np.linalg.det(affine[:3, :3]))

        for i, sp in enumerate(seg_paths):
            seg_data = np.asanyarray(nib.load(sp).dataobj).astype(np.uint8)
            seg_4d[:, :, :, i] = seg_data

            tp_stats: dict[str, Any] = {"timepoint": i + 1}
            for organ_name, label_id in TOTALSEG_LABEL_MAP.items():
                voxel_count = int((seg_data == label_id).sum())
                volume_ml = voxel_count * voxel_vol_mm3 / 1000.0
                tp_stats[f"{organ_name}_voxels"] = voxel_count
                tp_stats[f"{organ_name}_volume_ml"] = round(volume_ml, 2)
            vol_stats.append(tp_stats)

        seg_4d_path = os.path.join(results_base, VOLUMETRIC_SEQUENCE, "segmentation_4d.nii.gz")
        os.makedirs(os.path.dirname(seg_4d_path), exist_ok=True)
        nib.save(nib.Nifti1Image(seg_4d, affine), seg_4d_path)
        print(f"  Saved 4D segmentation: {seg_4d_path} (shape: {seg_4d.shape})")

        for organ_name, label_id in TOTALSEG_LABEL_MAP.items():
            organ_4d = (seg_4d == label_id).astype(np.uint8)
            organ_path = os.path.join(results_base, VOLUMETRIC_SEQUENCE, f"{organ_name}_4d.nii.gz")
            nib.save(nib.Nifti1Image(organ_4d, affine), organ_path)
            print(f"  Saved {organ_name} 4D mask: {organ_path}")

        summary["volumetric"] = {
            "seg_4d_path": seg_4d_path,
            "shape": list(seg_4d.shape),
            "per_timepoint": vol_stats,
        }

    for seq_name in DYNAMIC_SEQUENCES:
        seg_path = os.path.join(results_base, seq_name, "segmentation_2d.nii.gz")
        if not os.path.exists(seg_path):
            continue

        seg_data = np.asanyarray(nib.load(seg_path).dataobj)
        dyn_stats: dict[str, Any] = {
            "sequence": seq_name,
            "num_frames": int(seg_data.shape[2]),
        }

        for organ_name, obj_id in ORGAN_OBJECT_IDS.items():
            areas = [int((seg_data[:, :, t] == obj_id).sum()) for t in range(seg_data.shape[2])]
            dyn_stats[f"{organ_name}_mean_area_px"] = round(float(np.mean(areas)), 1)
            dyn_stats[f"{organ_name}_frames_with_organ"] = int(np.sum(np.array(areas) > 0))

        summary["dynamic"][seq_name] = dyn_stats
        print(f"  Dynamic {seq_name}: summarized {seg_data.shape[2]} frames")

    summary_path = os.path.join(results_base, "summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {summary_path}")
    return summary
