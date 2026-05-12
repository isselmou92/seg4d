"""Step 2 - TotalSegmentator on the volumetric (3D + time) sequence."""

from __future__ import annotations

import os

from ..constants import VOLUMETRIC_SEQUENCE
from ..io import load_metadata, save_metadata
from ..models.totalsegmentator import TotalSegmentatorBackend


def run_volumetric(data_dir: str, device: str = "gpu") -> list[str]:
    """Run TotalSegmentator on every volumetric timepoint.

    Parameters
    ----------
    data_dir
        Patient data directory.
    device
        TotalSegmentator device string (``"gpu"``, ``"cpu"``, ``"gpu:0"``).

    Returns
    -------
    list[str]
        Paths of the per-timepoint segmentation NIfTIs.
    """
    print("\n" + "=" * 60)
    print("STEP 2: TotalSegmentator on Volumetric Data")
    print("=" * 60)

    metadata = load_metadata(data_dir)
    if "volumetric" not in metadata:
        print("  No volumetric sequence in metadata; skipping.")
        return []

    nifti_paths = metadata["volumetric"]["nifti_paths"]
    results_base = os.path.join(data_dir, "results", VOLUMETRIC_SEQUENCE)

    backend = TotalSegmentatorBackend(device=device)
    seg_paths = backend.segment_timepoints(nifti_paths, results_base)

    metadata["volumetric"]["seg_paths"] = seg_paths
    save_metadata(data_dir, metadata)
    return seg_paths
