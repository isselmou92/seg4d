"""Reorganize a multi-temporal 4D MR series into per-phase DICOM series.

The output layout mirrors the Philips Big Bore 4D-CT pattern accepted by
RayStation:

  * each phase is a separate series (own ``SeriesInstanceUID``);
  * phase encoded in ``SeriesDescription`` and ``ImageComments`` (e.g.
    ``"..., 25.0%A"``);
  * ``StudyInstanceUID`` and ``FrameOfReferenceUID`` shared across phases;
  * ``InstanceNumber`` resets to 1..N within each series;
  * ``TemporalPositionIdentifier`` / ``NumberOfTemporalPositions`` removed.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any

import pydicom
from pydicom.uid import generate_uid


VOLUMETRIC_SEQUENCE_DEFAULT = "2_SAG_b"
PHASE_LABELS_DEFAULT: list[str] = ["0.0%A", "25.0%A", "50.0%A", "75.0%A"]


# ---------------------------------------------------------------------------
# DICOM grouping
# ---------------------------------------------------------------------------

def parse_volumetric_dicoms(dicom_dir: str) -> tuple[dict[int, list[tuple[float, str]]], int]:
    """Group volumetric DICOM files by ``TemporalPositionIdentifier``.

    Returns
    -------
    groups
        ``{tp: [(slice_location, file_path), ...]}`` sorted by slice location
        within each timepoint.
    num_slices
        Number of slices per timepoint.
    """
    groups: dict[int, list[tuple[float, str]]] = defaultdict(list)

    files = [
        f for f in os.listdir(dicom_dir)
        if not f.endswith((".7z", ".zip"))
        and os.path.isfile(os.path.join(dicom_dir, f))
    ]

    for fname in files:
        fpath = os.path.join(dicom_dir, fname)
        ds = pydicom.dcmread(fpath, stop_before_pixels=True)
        tp = int(getattr(ds, "TemporalPositionIdentifier", 0))
        sl = float(getattr(ds, "SliceLocation", 0))
        groups[tp].append((sl, fpath))

    for tp in groups:
        groups[tp].sort(key=lambda x: x[0])

    first_key = sorted(groups.keys())[0]
    num_slices = len(groups[first_key])
    return dict(groups), num_slices


# ---------------------------------------------------------------------------
# Phase map builders
# ---------------------------------------------------------------------------

def build_phase_map_manual(phase_map_arg: str) -> dict[int, str]:
    """Parse a manual phase map from a JSON string or file path.

    Accepts either a literal JSON string ``'{"1":"0.0%A","13":"25.0%A",...}'``
    or a path to a ``.json`` file with the same structure.
    """
    if os.path.isfile(phase_map_arg):
        with open(phase_map_arg, "r") as f:
            raw = json.load(f)
    else:
        raw = json.loads(phase_map_arg)
    return {int(k): str(v) for k, v in raw.items()}


def build_phase_map_evenly_spaced(
    groups: dict[int, list[tuple[float, str]]],
    labels: list[str] | None = None,
) -> dict[int, str]:
    """Pick ``len(labels)`` evenly-spaced timepoints from ``groups``."""
    if labels is None:
        labels = PHASE_LABELS_DEFAULT
    sorted_tps = sorted(groups.keys())
    n = len(sorted_tps)
    num_phases = len(labels)
    indices = [int(round(i * (n - 1) / (num_phases - 1))) for i in range(num_phases)]
    selected = [sorted_tps[i] for i in indices]
    return dict(zip(selected, labels))


# ---------------------------------------------------------------------------
# DICOM rewriter
# ---------------------------------------------------------------------------

def rewrite_phase_dicoms(
    groups: dict[int, list[tuple[float, str]]],
    phase_map: dict[int, str],
    base_series_desc: str,
    output_dir: str,
    base_series_number: int = 1501,
) -> list[dict[str, Any]]:
    """Write per-phase DICOM series to ``output_dir``.

    Returns a list of summary dicts (one per phase) used for validation
    and reporting.
    """
    os.makedirs(output_dir, exist_ok=True)

    now = datetime.now()
    creation_date = now.strftime("%Y%m%d")
    creation_time = now.strftime("%H%M%S.%f")

    phase_summaries: list[dict[str, Any]] = []

    for phase_idx, (tp, phase_label) in enumerate(
        sorted(phase_map.items(), key=lambda x: x[0])
    ):
        if tp not in groups:
            print(f"  WARNING: timepoint {tp} not found in data, skipping")
            continue

        series_uid = generate_uid()
        series_number = base_series_number + phase_idx
        series_desc = f"{base_series_desc}, {phase_label}"

        phase_dir = os.path.join(output_dir, f"phase_{phase_label.replace('%', 'pct')}")
        os.makedirs(phase_dir, exist_ok=True)

        slices_in_phase = groups[tp]
        sop_uids: list[str] = []
        slice_locations: list[float] = []

        print(f"  Phase {phase_label} (TP={tp}): {len(slices_in_phase)} slices "
              f"-> SeriesNumber={series_number}")

        for inst_num, (sl, fpath) in enumerate(slices_in_phase, start=1):
            ds = pydicom.dcmread(fpath)
            new_sop_uid = generate_uid()

            ds.SeriesInstanceUID = series_uid
            ds.SOPInstanceUID = new_sop_uid
            ds.SeriesNumber = series_number
            ds.SeriesDescription = series_desc
            ds.InstanceNumber = inst_num
            ds.InstanceCreationDate = creation_date
            ds.InstanceCreationTime = creation_time

            if hasattr(ds, "ImageComments"):
                ds.ImageComments = series_desc
            else:
                ds.add_new(0x00204000, "LT", series_desc)

            for tag in [(0x0020, 0x0100), (0x0020, 0x0105)]:
                if tag in ds:
                    del ds[tag]

            ds.file_meta.MediaStorageSOPInstanceUID = new_sop_uid

            out_name = f"MR_{phase_label.replace('%', 'pct')}_{inst_num:04d}.dcm"
            out_path = os.path.join(phase_dir, out_name)
            ds.save_as(out_path)

            sop_uids.append(new_sop_uid)
            slice_locations.append(sl)

        phase_summaries.append({
            "phase_label": phase_label,
            "timepoint": tp,
            "series_uid": series_uid,
            "series_number": series_number,
            "series_description": series_desc,
            "num_files": len(slices_in_phase),
            "sop_uids": sop_uids,
            "slice_locations": slice_locations,
            "output_dir": phase_dir,
        })

    return phase_summaries


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary_table(phase_summaries: list[dict[str, Any]]) -> None:
    """Pretty-print the per-phase summary table."""
    print("\n" + "=" * 60)
    print("4D MR REORGANIZATION SUMMARY")
    print("=" * 60)
    print(f"{'Phase':<12} {'TP':>4} {'SerNum':>8} {'Files':>6} "
          f"{'SL range':>20}  SeriesDescription")
    print("-" * 90)
    for s in phase_summaries:
        sl = s["slice_locations"]
        sl_range = f"{min(sl):.1f} .. {max(sl):.1f}" if sl else "N/A"
        print(f"{s['phase_label']:<12} {s['timepoint']:>4} {s['series_number']:>8} "
              f"{s['num_files']:>6} {sl_range:>20}  {s['series_description']}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def reorganize_for_tps(
    data_dir: str,
    *,
    phase_map: str | None = None,
    evenly_spaced: bool = False,
    output_dir: str | None = None,
    base_series_number: int = 1501,
    sequence: str = VOLUMETRIC_SEQUENCE_DEFAULT,
) -> bool:
    """Run the full reorganization and validation pipeline.

    Returns ``True`` if all validation checks passed.
    """
    from .validate import validate_output  # local import avoids cycles

    dicom_dir = os.path.join(data_dir, "dicom", sequence)
    if not os.path.isdir(dicom_dir):
        print(f"ERROR: DICOM directory not found: {dicom_dir}")
        return False

    output_dir = output_dir or os.path.join(data_dir, "dicom_4d")

    print("Parsing volumetric DICOMs ...")
    groups, num_slices = parse_volumetric_dicoms(dicom_dir)
    sorted_tps = sorted(groups.keys())
    print(f"  {len(sorted_tps)} timepoints x {num_slices} slices")

    if evenly_spaced:
        phase_map_dict = build_phase_map_evenly_spaced(groups)
        print(f"\nAuto-selected evenly-spaced timepoints:")
    elif phase_map is not None:
        phase_map_dict = build_phase_map_manual(phase_map)
        print(f"\nManual phase mapping:")
    else:
        raise ValueError("Either phase_map or evenly_spaced must be supplied.")

    for tp, label in sorted(phase_map_dict.items()):
        print(f"  TP {tp:>3} -> {label}")

    missing = [tp for tp in phase_map_dict if tp not in groups]
    if missing:
        print(f"\nERROR: timepoints not found in data: {missing}")
        print(f"  Available: {sorted_tps}")
        return False

    first_tp = sorted(phase_map_dict.keys())[0]
    first_file = groups[first_tp][0][1]
    ds_ref = pydicom.dcmread(first_file, stop_before_pixels=True)
    base_series_desc = str(getattr(ds_ref, "SeriesDescription", "4D_MR"))

    print(f"\nBase SeriesDescription: '{base_series_desc}'")
    print(f"Output directory: {output_dir}\n")

    print("Writing reorganized DICOM files ...")
    phase_summaries = rewrite_phase_dicoms(
        groups, phase_map_dict, base_series_desc, output_dir,
        base_series_number=base_series_number,
    )

    print_summary_table(phase_summaries)
    ok = validate_output(output_dir, phase_summaries)

    meta_path = os.path.join(output_dir, "reorganization_metadata.json")
    meta = {
        "source_dir": dicom_dir,
        "output_dir": output_dir,
        "base_series_description": base_series_desc,
        "phase_map": {str(k): v for k, v in phase_map_dict.items()},
        "phases": [
            {
                "phase_label": s["phase_label"],
                "timepoint": s["timepoint"],
                "series_instance_uid": s["series_uid"],
                "series_number": s["series_number"],
                "series_description": s["series_description"],
                "num_files": s["num_files"],
                "output_dir": s["output_dir"],
            }
            for s in phase_summaries
        ],
        "validation_passed": ok,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata written to: {meta_path}")

    if ok:
        print("\nDone. Import all phase folders into RayStation as a single 4D dataset.")
    return ok
