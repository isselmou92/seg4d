"""Validation checks for the per-phase DICOM export."""

from __future__ import annotations

import os
from typing import Any

import pydicom


def validate_output(output_dir: str, phase_summaries: list[dict[str, Any]]) -> bool:
    """Re-read the exported phase folders and run sanity checks.

    Checks:
      * StudyInstanceUID is identical across all phases.
      * FrameOfReferenceUID is identical across all phases.
      * Each phase has a unique SeriesInstanceUID.
      * SOP UIDs are unique across the whole 4D dataset.

    Returns ``True`` when all checks pass.
    """
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    all_studies = set()
    all_frames = set()
    all_series = set()
    all_sops: set[str] = set()
    sop_collisions: list[str] = []

    for s in phase_summaries:
        phase_dir = s["output_dir"]
        files = sorted(
            os.path.join(phase_dir, f) for f in os.listdir(phase_dir) if f.endswith(".dcm")
        )
        for fpath in files:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True)
            all_studies.add(str(ds.StudyInstanceUID))
            all_frames.add(str(getattr(ds, "FrameOfReferenceUID", "MISSING")))
            all_series.add(str(ds.SeriesInstanceUID))
            sop = str(ds.SOPInstanceUID)
            if sop in all_sops:
                sop_collisions.append(sop)
            all_sops.add(sop)

    ok = True
    print(f"  Phases written:          {len(phase_summaries)}")
    print(f"  Total slices:            {len(all_sops)}")
    print(f"  Distinct StudyUIDs:      {len(all_studies)} (must be 1)")
    print(f"  Distinct FrameOfRefUIDs: {len(all_frames)} (must be 1)")
    print(f"  Distinct SeriesUIDs:     {len(all_series)} (must equal phases)")
    print(f"  SOP UID collisions:      {len(sop_collisions)} (must be 0)")

    if len(all_studies) != 1:
        print("  FAIL: StudyInstanceUID is not identical across phases")
        ok = False
    if len(all_frames) != 1:
        print("  FAIL: FrameOfReferenceUID is not identical across phases")
        ok = False
    if len(all_series) != len(phase_summaries):
        print("  FAIL: Each phase must have its own SeriesInstanceUID")
        ok = False
    if sop_collisions:
        print(f"  FAIL: Duplicate SOPInstanceUIDs detected: {sop_collisions[:5]}...")
        ok = False

    print("=" * 60)
    print("VALIDATION RESULT:", "PASS" if ok else "FAIL")
    print("=" * 60)
    return ok
