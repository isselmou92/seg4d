#!/usr/bin/env python3
"""Standalone interactive seed-point picker for seg4d.

No seg4d package installation required.
Dependencies: matplotlib  Pillow  numpy  (all pip-installable in < 1 min)

    pip install matplotlib Pillow numpy

────────────────────────────────────────────────────────────────────────────
USAGE
────────────────────────────────────────────────────────────────────────────

A) You already ran `seg4d run --steps convert` on this machine:

    python tools/seed_picker.py --data_dir /path/to/patient

   The script reads `nifti/conversion_metadata.json` to find JPEG frames.

B) You have a folder of JPEG sub-directories (e.g. copied from the cluster):

    python tools/seed_picker.py --jpegs_dir /path/to/jpegs_root

   Every sub-folder that contains *.jpg files is treated as one sequence.

C) Explicit named sequence directories:

    python tools/seed_picker.py \\
        --sequences 60_COR_b=/data/jpegs/60_COR_b \\
                    60_SAG_b=/data/jpegs/60_SAG_b

   Each argument is   name=path_to_jpeg_folder.

OUTPUT

    seed_boxes.json is written to --output (default: <data_dir>/seed_boxes.json,
    or ./seed_boxes.json when using --jpegs_dir / --sequences).

CONTROLS IN THE GUI

    Left-click   add positive point for the active organ
    Right-click  add negative point for the active organ
    1 / 2 / 3    switch active organ (liver / kidney_right / kidney_left)
    Z            undo last point
    N            save current sequence and move to the next one
    Q            save current sequence and quit

    The border colour of the image indicates the active organ:
      green = liver    blue = kidney_right    orange = kidney_left

WORKFLOW ON CLUSTER

    1.  Run  seg4d run --steps convert  locally  (or copy JPEG frames from
        the cluster).
    2.  Run this script on your laptop to place seed points.
    3.  Copy the resulting  seed_boxes.json  to the cluster data directory.
    4.  Submit  slurm/submit_benchmark.sh  — no --interactive flag needed.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ORGAN_LIST = ("liver", "kidney_right", "kidney_left")
_ORGAN_COLORS = {
    "liver":         "#00C800",
    "kidney_right":  "#0064FF",
    "kidney_left":   "#FFA500",
}
_ORGAN_KEYS = {"1": "liver", "2": "kidney_right", "3": "kidney_left"}

# When only the 60-series is prompted, copy to the matching FB series.
_PAIR_MAP = {
    "60_COR_b": "FB_COR_b",
    "60_SAG_b": "FB_SAG_b",
}


# ---------------------------------------------------------------------------
# Sequence discovery
# ---------------------------------------------------------------------------

def _discover_from_metadata(data_dir: Path) -> dict[str, Path]:
    """Read conversion_metadata.json and return {seq_name: jpeg_dir}."""
    meta_path = data_dir / "nifti" / "conversion_metadata.json"
    if not meta_path.exists():
        return {}
    with open(meta_path) as f:
        meta = json.load(f)
    seqs: dict[str, Path] = {}
    for seq_name, seq_meta in meta.get("dynamic", {}).items():
        jdir = Path(seq_meta["jpeg_dir"])
        if jdir.is_dir():
            seqs[seq_name] = jdir
    return seqs


def _discover_from_nifti(data_dir: Path) -> dict[str, Path]:
    """Scan data_dir/nifti/ for sub-directories that contain JPEG files."""
    seqs: dict[str, Path] = {}
    nifti_root = data_dir / "nifti"
    if not nifti_root.is_dir():
        return seqs
    for candidate in sorted(nifti_root.iterdir()):
        if not candidate.is_dir():
            continue
        # look for a direct *.jpg match or a frames_jpg sub-folder
        if list(candidate.glob("*.jpg")) or list(candidate.glob("*.jpeg")):
            seqs[candidate.name] = candidate
        for sub in candidate.iterdir():
            if sub.is_dir() and (list(sub.glob("*.jpg")) or list(sub.glob("*.jpeg"))):
                seqs[candidate.name] = sub
                break
    return seqs


def _discover_from_jpegs_dir(jpegs_dir: Path) -> dict[str, Path]:
    """Every sub-directory that contains JPEGs is treated as one sequence."""
    seqs: dict[str, Path] = {}
    for sub in sorted(jpegs_dir.iterdir()):
        if sub.is_dir() and (list(sub.glob("*.jpg")) or list(sub.glob("*.jpeg"))):
            seqs[sub.name] = sub
    # Also accept a flat directory (single sequence named after the folder)
    if not seqs:
        if list(jpegs_dir.glob("*.jpg")) or list(jpegs_dir.glob("*.jpeg")):
            seqs[jpegs_dir.name] = jpegs_dir
    return seqs


def _load_first_frame(jpeg_dir: Path):
    """Return the first JPEG as a numpy RGB array."""
    import numpy as np
    from PIL import Image

    paths = sorted(jpeg_dir.glob("*.jpg"))
    if not paths:
        paths = sorted(jpeg_dir.glob("*.jpeg"))
    if not paths:
        return None
    return np.array(Image.open(paths[0]).convert("RGB"))


# ---------------------------------------------------------------------------
# Single-sequence picker
# ---------------------------------------------------------------------------

def _pick_one_sequence(
    seq_name: str,
    jpeg_dir: Path,
    existing: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Open the matplotlib GUI for one sequence. Returns the seed dict or None on skip."""
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit(
            "matplotlib is not installed.\n"
            "Run:  pip install matplotlib"
        )

    frame0 = _load_first_frame(jpeg_dir)
    if frame0 is None:
        print(f"  [skip] No JPEG frames found in {jpeg_dir}")
        return None

    # Pre-fill with any existing points so the user can amend them.
    points: dict[str, dict[str, list[list[int]]]] = {}
    for organ in _ORGAN_LIST:
        prev = (existing or {}).get(organ) or {}
        if isinstance(prev, dict):
            points[organ] = {
                "positive": list(prev.get("positive") or []),
                "negative": list(prev.get("negative") or []),
            }
        else:
            points[organ] = {"positive": [], "negative": []}

    state = {"organ": "liver", "history": [], "quit": False}

    fig, ax = plt.subplots(figsize=(9, 9))
    fig.canvas.manager.set_window_title(f"Seed Picker — {seq_name}")

    def _redraw() -> None:
        ax.clear()
        ax.imshow(frame0)
        organ_now = state["organ"]
        ax.set_title(
            f"Sequence: {seq_name}   |   Active organ: {organ_now}\n"
            "Left=positive  Right=negative  |  1=liver  2=kidney_right  3=kidney_left  "
            "Z=undo  N=next  Q=quit",
            fontsize=9,
        )
        for organ in _ORGAN_LIST:
            col = _ORGAN_COLORS[organ]
            for pt in points[organ]["positive"]:
                ax.plot(pt[0], pt[1], "o", color=col, markersize=11,
                        markeredgecolor="white", markeredgewidth=1.5)
                ax.annotate(f"+{organ}", (pt[0] + 9, pt[1]),
                            color=col, fontsize=7, fontweight="bold")
            for pt in points[organ]["negative"]:
                ax.plot(pt[0], pt[1], "x", color="red", markersize=13, markeredgewidth=2.5)
                ax.plot(pt[0], pt[1], "o", color="red", markersize=11,
                        markeredgecolor="white", markeredgewidth=1.0, fillstyle="none")
                ax.annotate(f"-{organ}", (pt[0] + 9, pt[1]),
                            color="#FF5050", fontsize=7)

        border_color = _ORGAN_COLORS[organ_now]
        for side in ("top", "bottom", "left", "right"):
            ax.spines[side].set_color(border_color)
            ax.spines[side].set_linewidth(4)
        fig.canvas.draw_idle()

    def _on_click(event) -> None:
        if event.inaxes != ax or event.xdata is None:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        organ = state["organ"]
        if event.button == 1:
            points[organ]["positive"].append([x, y])
            state["history"].append((organ, "positive", len(points[organ]["positive"]) - 1))
            print(f"    [{seq_name}] + {organ}  positive: ({x}, {y})")
        elif event.button == 3:
            points[organ]["negative"].append([x, y])
            state["history"].append((organ, "negative", len(points[organ]["negative"]) - 1))
            print(f"    [{seq_name}] - {organ}  negative: ({x}, {y})")
        _redraw()

    def _on_key(event) -> None:
        if event.key in _ORGAN_KEYS:
            state["organ"] = _ORGAN_KEYS[event.key]
            print(f"    Switched to: {state['organ']}")
            _redraw()
        elif event.key == "z":
            if state["history"]:
                organ, ptype, idx = state["history"].pop()
                removed = points[organ][ptype].pop(idx)
                print(f"    Undo {ptype} {organ}: {removed}")
                _redraw()
        elif event.key == "q":
            state["quit"] = True
            plt.close(fig)
        elif event.key == "n":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", _on_click)
    fig.canvas.mpl_connect("key_press_event", _on_key)
    _redraw()
    plt.show()

    # Build output dict (only include organs that have at least one point)
    seed: dict[str, Any] = {}
    for organ in _ORGAN_LIST:
        pos = points[organ]["positive"]
        neg = points[organ]["negative"]
        seed[organ] = {"positive": pos, "negative": neg} if (pos or neg) else None

    return seed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_sequences_arg(values: list[str]) -> dict[str, Path]:
    """Parse  name=/path  pairs from --sequences arguments."""
    result: dict[str, Path] = {}
    for item in values:
        if "=" not in item:
            sys.exit(f"--sequences expects  name=/path  pairs, got: {item!r}")
        name, path = item.split("=", 1)
        result[name.strip()] = Path(path.strip())
    return result


def run(
    sequences: dict[str, Path],
    output_path: Path,
    all_seq_names: list[str] | None = None,
) -> None:
    """Run the picker for all sequences and write seed_boxes.json."""
    # Decide which sequences to prompt (prefer 60-series over FB-series)
    prompt_seqs = [s for s in sequences if s in _PAIR_MAP]
    if not prompt_seqs:
        prompt_seqs = list(sequences.keys())

    # Load existing seed file so the user can amend previous picks
    existing_data: dict[str, Any] = {}
    if output_path.exists():
        with open(output_path) as f:
            existing_data = json.load(f)
        print(f"  Loaded existing seeds from {output_path}")

    seed_data: dict[str, Any] = dict(existing_data)

    print(f"\nSequences to prompt: {prompt_seqs}")
    print("─" * 60)

    for seq_name in prompt_seqs:
        jpeg_dir = sequences[seq_name]
        print(f"\n[{seq_name}]  frames: {jpeg_dir}")
        result = _pick_one_sequence(seq_name, jpeg_dir, existing_data.get(seq_name))

        if result is None:
            print(f"  Skipped {seq_name} (no JPEG frames).")
            continue

        seed_data[seq_name] = result
        print(f"  Saved points for: {seq_name}")

        # Auto-copy to paired FB sequence
        all_names = all_seq_names or list(sequences.keys())
        if seq_name in _PAIR_MAP:
            paired = _PAIR_MAP[seq_name]
            if paired in all_names or paired in sequences:
                seed_data[paired] = dict(result)
                print(f"  Copied points to paired sequence: {paired}")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(seed_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Seed file written to: {output_path}")
    print(f"{'='*60}")
    print("\nNext step: copy this file to the cluster data directory, then submit:")
    print("  bash slurm/submit_benchmark.sh")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="seed_picker",
        description="Interactive seed-point picker for seg4d. "
                    "Run on your local machine; upload seed_boxes.json to the cluster.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--data_dir",
        metavar="DIR",
        help="Patient data directory (same as --data_dir in seg4d run). "
             "Must contain nifti/ sub-folder produced by 'seg4d run --steps convert'.",
    )
    src.add_argument(
        "--jpegs_dir",
        metavar="DIR",
        help="Directory whose sub-folders each contain JPEG frames for one sequence.",
    )
    src.add_argument(
        "--sequences",
        nargs="+",
        metavar="name=/path",
        help="Explicit sequence directories as  name=/path  pairs.",
    )

    parser.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help="Output JSON path (default: <data_dir>/seed_boxes.json or ./seed_boxes.json).",
    )

    args = parser.parse_args(argv)

    # ---- resolve sequences dict ----
    all_seq_names: list[str] | None = None

    if args.data_dir:
        data_dir = Path(args.data_dir)
        if not data_dir.is_dir():
            sys.exit(f"ERROR: --data_dir '{data_dir}' does not exist.")
        seqs = _discover_from_metadata(data_dir)
        if not seqs:
            print("  conversion_metadata.json not found — scanning nifti/ sub-folders...")
            seqs = _discover_from_nifti(data_dir)
        all_seq_names = list(seqs.keys())
        default_output = data_dir / "seed_boxes.json"

    elif args.jpegs_dir:
        jpegs_dir = Path(args.jpegs_dir)
        if not jpegs_dir.is_dir():
            sys.exit(f"ERROR: --jpegs_dir '{jpegs_dir}' does not exist.")
        seqs = _discover_from_jpegs_dir(jpegs_dir)
        default_output = Path("seed_boxes.json")

    else:  # --sequences
        seqs = _parse_sequences_arg(args.sequences)
        default_output = Path("seed_boxes.json")

    if not seqs:
        sys.exit(
            "No JPEG sequences found. "
            "Run 'seg4d run --steps convert' first to generate JPEG frames, "
            "or point --jpegs_dir at a folder of JPEG sub-directories."
        )

    output_path = Path(args.output) if args.output else default_output

    print("=" * 60)
    print("  seg4d — Standalone Seed Picker")
    print("=" * 60)
    for name, path in seqs.items():
        print(f"  {name:20s}  →  {path}")
    print(f"\n  Output file: {output_path}")

    run(seqs, output_path, all_seq_names)


if __name__ == "__main__":
    main()
