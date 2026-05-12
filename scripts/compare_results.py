"""Compare benchmark results across all models.

Usage
-----
    python scripts/compare_results.py --data_dir <patient_folder> [--output_dir <dir>]

Reads every ``results_*/summary.json`` (plus ``results/summary.json`` for the
TotalSegmentator baseline) under ``--data_dir``, prints a formatted comparison
table to the terminal, and writes:

    <output_dir>/comparison_table.csv   — machine-readable summary
    <output_dir>/comparison_chart.png   — bar charts per organ / metric
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Well-known result folder names and display labels
# ---------------------------------------------------------------------------

_KNOWN_ORDER = [
    ("results",                "TotalSegmentator"),
    ("results_sam3",           "SAM3"),
    ("results_medsam2_boxes",  "MedSAM2 (boxes)"),
    ("results_medsam2_reslice","MedSAM2 (reslice)"),
]

_ORGANS = ["liver", "kidney_right", "kidney_left"]


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _discover_runs(data_dir: Path) -> list[tuple[str, Path]]:
    """Return [(label, summary_path)] ordered by _KNOWN_ORDER, then alphabetically."""
    found: list[tuple[str, Path]] = []
    seen: set[str] = set()

    for folder, label in _KNOWN_ORDER:
        p = data_dir / folder / "summary.json"
        if p.exists():
            found.append((label, p))
            seen.add(folder)

    for p in sorted(data_dir.glob("results*/summary.json")):
        folder = p.parent.name
        if folder not in seen:
            found.append((folder, p))

    return found


def _load_summary(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def _build_volumetric_rows(
    runs: list[tuple[str, dict[str, Any]]],
) -> tuple[list[str], list[list[Any]]]:
    """Return (header, rows) for volumetric statistics."""
    header = ["Model", "Timepoints"]
    for org in _ORGANS:
        header += [f"{org}_vol_mL (mean)", f"{org}_vol_mL (min)", f"{org}_vol_mL (max)"]

    rows: list[list[Any]] = []
    for label, summary in runs:
        vol = summary.get("volumetric", {})
        per_tp = vol.get("per_timepoint", [])
        if not per_tp:
            continue

        row: list[Any] = [label, len(per_tp)]
        for org in _ORGANS:
            key = f"{org}_volume_ml"
            vals = [tp.get(key, 0.0) for tp in per_tp]
            if vals:
                row += [
                    round(sum(vals) / len(vals), 1),
                    round(min(vals), 1),
                    round(max(vals), 1),
                ]
            else:
                row += ["—", "—", "—"]
        rows.append(row)
    return header, rows


def _build_dynamic_rows(
    runs: list[tuple[str, dict[str, Any]]],
) -> dict[str, tuple[list[str], list[list[Any]]]]:
    """Return {seq_name: (header, rows)} for dynamic statistics."""
    all_seqs: list[str] = []
    for _label, summary in runs:
        for seq in summary.get("dynamic", {}):
            if seq not in all_seqs:
                all_seqs.append(seq)

    tables: dict[str, tuple[list[str], list[list[Any]]]] = {}
    for seq in all_seqs:
        header = ["Model", "Frames"]
        for org in _ORGANS:
            header += [f"{org}_area_px (mean)", f"{org}_frames_covered"]

        rows: list[list[Any]] = []
        for label, summary in runs:
            dyn = summary.get("dynamic", {}).get(seq)
            if dyn is None:
                continue
            row: list[Any] = [label, dyn.get("num_frames", "?")]
            for org in _ORGANS:
                row.append(dyn.get(f"{org}_mean_area_px", 0.0))
                row.append(dyn.get(f"{org}_frames_with_organ", 0))
            rows.append(row)
        tables[seq] = (header, rows)
    return tables


# ---------------------------------------------------------------------------
# Terminal pretty-print
# ---------------------------------------------------------------------------

def _col_widths(header: list[str], rows: list[list[Any]]) -> list[int]:
    widths = [len(str(h)) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    return widths


def _print_table(title: str, header: list[str], rows: list[list[Any]]) -> None:
    if not rows:
        print(f"\n  [{title}]  — no data\n")
        return
    widths = _col_widths(header, rows)
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in widths) + " |"

    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(sep)
    print(fmt.format(*header))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(c) for c in row]))
    print(sep)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def _write_csv(path: Path, sections: list[tuple[str, list[str], list[list[Any]]]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for section_title, header, rows in sections:
            writer.writerow([f"# {section_title}"])
            writer.writerow(header)
            for row in rows:
                writer.writerow(row)
            writer.writerow([])
    print(f"\n  CSV saved  → {path}")


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def _draw_chart(
    path: Path,
    vol_header: list[str],
    vol_rows: list[list[Any]],
    dyn_tables: dict[str, tuple[list[str], list[list[Any]]]],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib not available — skipping chart generation.")
        return

    n_vol_organs = len(_ORGANS)
    n_dyn_seqs = len(dyn_tables)
    total_panels = (1 if vol_rows else 0) + n_dyn_seqs * 2
    if total_panels == 0:
        return

    fig, axes = plt.subplots(
        total_panels, 1,
        figsize=(max(10, len(vol_rows) * 2 + 4), total_panels * 4 + 1),
        squeeze=False,
    )
    ax_idx = 0
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    # --- Volumetric volumes ---
    if vol_rows:
        ax = axes[ax_idx][0]
        ax_idx += 1
        labels = [r[0] for r in vol_rows]
        x = np.arange(n_vol_organs)
        bar_width = 0.8 / max(len(labels), 1)
        for i, (row, color) in enumerate(zip(vol_rows, colors)):
            means = [row[2 + j * 3] for j in range(n_vol_organs)]
            mins  = [row[3 + j * 3] for j in range(n_vol_organs)]
            maxs  = [row[4 + j * 3] for j in range(n_vol_organs)]
            try:
                errs_lo = [float(m) - float(lo) for m, lo in zip(means, mins)]
                errs_hi = [float(hi) - float(m) for m, hi in zip(maxs, means)]
                errs = [errs_lo, errs_hi]
                means_f = [float(v) for v in means]
            except (ValueError, TypeError):
                errs = None
                means_f = [0.0] * n_vol_organs
            offset = (i - len(labels) / 2 + 0.5) * bar_width
            ax.bar(
                x + offset, means_f,
                width=bar_width, label=row[0], color=color,
                yerr=errs, capsize=4, error_kw={"elinewidth": 1},
            )
        ax.set_xticks(x)
        ax.set_xticklabels([o.replace("_", " ") for o in _ORGANS])
        ax.set_ylabel("Volume (mL)")
        ax.set_title("Volumetric — mean organ volume across timepoints")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    # --- Dynamic: mean area and frame coverage ---
    for seq, (dyn_header, dyn_rows) in dyn_tables.items():
        if not dyn_rows:
            ax_idx += 2
            continue

        # mean area
        ax = axes[ax_idx][0]
        ax_idx += 1
        labels = [r[0] for r in dyn_rows]
        x = np.arange(n_vol_organs)
        bar_width = 0.8 / max(len(labels), 1)
        for i, (row, color) in enumerate(zip(dyn_rows, colors[1:])):
            # area columns: [Model, Frames, org0_area, org0_frames, org1_area, ...]
            areas = [row[2 + j * 2] for j in range(n_vol_organs)]
            offset = (i - len(labels) / 2 + 0.5) * bar_width
            ax.bar(x + offset, [float(a) for a in areas],
                   width=bar_width, label=row[0], color=color)
        ax.set_xticks(x)
        ax.set_xticklabels([o.replace("_", " ") for o in _ORGANS])
        ax.set_ylabel("Mean area (px)")
        ax.set_title(f"{seq} — mean segmentation area per frame")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        # frame coverage
        ax = axes[ax_idx][0]
        ax_idx += 1
        n_frames_per_model = {r[0]: r[1] for r in dyn_rows}
        for i, (row, color) in enumerate(zip(dyn_rows, colors[1:])):
            n_frames = n_frames_per_model.get(row[0], 1) or 1
            coverage = [
                round(float(row[3 + j * 2]) / n_frames * 100, 1)
                for j in range(n_vol_organs)
            ]
            offset = (i - len(labels) / 2 + 0.5) * bar_width
            ax.bar(x + offset, coverage, width=bar_width, label=row[0], color=color)
        ax.set_xticks(x)
        ax.set_xticklabels([o.replace("_", " ") for o in _ORGANS])
        ax.set_ylabel("Frame coverage (%)")
        ax.set_ylim(0, 110)
        ax.set_title(f"{seq} — % of frames with each organ detected")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout(pad=2.0)
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    print(f"  Chart saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare seg4d benchmark results across all models.",
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Patient data directory (the one passed to --data_dir in seg4d run).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Where to write comparison_table.csv and comparison_chart.png. "
             "Defaults to --data_dir.",
    )
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        sys.exit(f"ERROR: --data_dir '{data_dir}' does not exist.")

    output_dir = Path(args.output_dir) if args.output_dir else data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----- discover runs -----
    run_paths = _discover_runs(data_dir)
    if not run_paths:
        sys.exit(
            f"No summary.json files found under {data_dir}.\n"
            "Make sure you have run at least one model with 'seg4d run ... --steps reassemble'."
        )

    print(f"\nFound {len(run_paths)} result set(s):")
    for label, path in run_paths:
        print(f"  [{label}]  {path}")

    runs = [(label, _load_summary(path)) for label, path in run_paths]

    # ----- build tables -----
    vol_header, vol_rows = _build_volumetric_rows(runs)
    dyn_tables = _build_dynamic_rows(runs)

    # ----- print to terminal -----
    _print_table("Volumetric — organ volume (mL, mean / min / max over timepoints)", vol_header, vol_rows)
    for seq, (dyn_header, dyn_rows) in dyn_tables.items():
        _print_table(f"Dynamic [{seq}] — mean area (px) + frame coverage", dyn_header, dyn_rows)

    # ----- write CSV -----
    csv_sections: list[tuple[str, list[str], list[list[Any]]]] = [
        ("Volumetric", vol_header, vol_rows),
        *[(seq, h, r) for seq, (h, r) in dyn_tables.items()],
    ]
    _write_csv(output_dir / "comparison_table.csv", csv_sections)

    # ----- draw chart -----
    _draw_chart(
        output_dir / "comparison_chart.png",
        vol_header, vol_rows,
        dyn_tables,
    )

    print("\nDone.\n")


if __name__ == "__main__":
    main()
