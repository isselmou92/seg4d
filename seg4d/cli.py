"""Command-line interface for seg4d.

Subcommands:
  * ``run``            - run one or more pipeline steps end-to-end
  * ``generate-seeds`` - render seed_template references / starter JSON
  * ``reorganize-tps`` - export per-phase DICOMs for RayStation
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from .models import list_segmenters
from .pipeline import run_pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_seed_frames(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_gpus(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Sub-parsers
# ---------------------------------------------------------------------------

def _add_run_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("run", help="Run pipeline steps (convert / volumetric / dynamic / reassemble).")
    p.add_argument("--data_dir", required=True, help="Patient data directory (must contain dicom/).")
    p.add_argument(
        "--steps",
        default="all",
        help="Comma-separated steps or 'all'. Choices: convert, volumetric, dynamic, reassemble.",
    )
    p.add_argument(
        "--model",
        default="sam3",
        choices=list_segmenters(),
        help="Backend used by the dynamic step.",
    )
    p.add_argument("--device", default="gpu", help="Device for TotalSegmentator (gpu/cpu).")
    p.add_argument(
        "--seed_mode",
        default="boxes",
        choices=["boxes", "reslice"],
        help="boxes: prompts from seed_boxes.json (or interactive). reslice: prompts derived from volumetric seg (MedSAM2 only).",
    )
    p.add_argument("--seed_file", default=None, help="Path to seed_boxes.json (default: <data_dir>/seed_boxes.json).")
    p.add_argument(
        "--seed_frames",
        default="0",
        help="Comma-separated frame indices to apply prompts on (default: 0).",
    )
    p.add_argument(
        "--interactive",
        action="store_true",
        help="Open the matplotlib point picker before running the dynamic step.",
    )
    p.add_argument(
        "--results_dir",
        default="results",
        help=(
            "Output folder for dynamic + reassemble results, relative to --data_dir "
            "or absolute. Defaults to 'results'. Use a unique name per model when "
            "benchmarking, e.g. results_sam3, results_medsam2_boxes."
        ),
    )

    medsam2_group = p.add_argument_group("MedSAM2 backend options")
    medsam2_group.add_argument(
        "--medsam2_checkpoint",
        default="MedSAM2/checkpoints/MedSAM2_MRI_LiverLesion.pt",
        help=(
            "Path to MedSAM2 .pt checkpoint. "
            "Available checkpoints (downloaded to MedSAM2/checkpoints/ by setup_all.bat): "
            "MedSAM2_MRI_LiverLesion.pt (default, best for liver/kidney MRI), "
            "MedSAM2_latest.pt (general-purpose)."
        ),
    )
    medsam2_group.add_argument(
        "--medsam2_cfg",
        default="configs/sam2.1_hiera_t512.yaml",
        help="MedSAM2 model config name (resolved by the SAM2 hydra search path).",
    )

    sam3_group = p.add_argument_group("SAM3 backend options")
    sam3_group.add_argument(
        "--sam3_model_id", default="facebook/sam3", help="HuggingFace model ID for SAM3."
    )
    sam3_group.add_argument(
        "--gpus",
        default="0",
        help="Comma-separated GPU IDs (only the first is used by SAM3).",
    )


def _add_generate_seeds_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "generate-seeds",
        help="Render seed-template reference images and create a starter seed_boxes.json.",
    )
    p.add_argument("--data_dir", required=True, help="Patient data directory (must have run 'convert').")


def _add_reorganize_tps_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "reorganize-tps",
        help="Reorganize 4D MR DICOMs into per-phase series for RayStation TPS.",
    )
    p.add_argument("--data_dir", required=True, help="Patient data directory containing dicom/.")
    p.add_argument("--sequence", default="2_SAG_b", help="Volumetric DICOM sub-folder name.")
    p.add_argument(
        "--phase_map",
        default=None,
        help='JSON map of TP -> phase label, e.g. \'{"1":"0.0%%A","13":"25.0%%A","25":"50.0%%A","37":"75.0%%A"}\' or path to a JSON file.',
    )
    p.add_argument(
        "--evenly_spaced",
        action="store_true",
        help="Auto-pick evenly-spaced timepoints (overrides --phase_map).",
    )
    p.add_argument("--output_dir", default=None, help="Output directory (default: <data_dir>/dicom_4d).")
    p.add_argument("--base_series_number", type=int, default=1501, help="Starting SeriesNumber.")


# ---------------------------------------------------------------------------
# Dispatchers
# ---------------------------------------------------------------------------

def _run_dispatch(args: argparse.Namespace) -> None:
    steps = ["all"] if args.steps == "all" else [s.strip() for s in args.steps.split(",")]
    seed_frames = _parse_seed_frames(args.seed_frames)

    backend_kwargs: dict[str, Any] = {}
    if args.model == "medsam2":
        backend_kwargs["checkpoint"] = args.medsam2_checkpoint
        backend_kwargs["config"] = args.medsam2_cfg
    elif args.model == "sam3":
        backend_kwargs["hf_model_id"] = args.sam3_model_id
        backend_kwargs["gpus"] = _parse_gpus(args.gpus)

    if args.seed_mode == "reslice" and args.model != "medsam2":
        raise SystemExit(
            f"--seed_mode reslice is only supported with --model medsam2 "
            f"(requested: {args.model})."
        )
    if args.interactive and args.seed_mode != "boxes":
        raise SystemExit("--interactive requires --seed_mode boxes.")

    run_pipeline(
        data_dir=args.data_dir,
        steps=steps,
        model=args.model,
        device=args.device,
        seed_mode=args.seed_mode,
        seed_file=args.seed_file,
        seed_frames=seed_frames,
        interactive=args.interactive,
        backend_kwargs=backend_kwargs,
        results_dir=args.results_dir,
    )


def _generate_seeds_dispatch(args: argparse.Namespace) -> None:
    from .seeds.template import generate_seed_template
    generate_seed_template(args.data_dir)


def _reorganize_tps_dispatch(args: argparse.Namespace) -> None:
    from .reorganize import reorganize_for_tps

    if args.phase_map is None and not args.evenly_spaced:
        raise SystemExit("Either --phase_map or --evenly_spaced must be provided.")

    ok = reorganize_for_tps(
        data_dir=args.data_dir,
        phase_map=args.phase_map,
        evenly_spaced=args.evenly_spaced,
        output_dir=args.output_dir,
        base_series_number=args.base_series_number,
        sequence=args.sequence,
    )
    if not ok:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seg4d",
        description="4D medical image segmentation pipeline (MedSAM2 / SAM3 + TotalSegmentator).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _add_run_parser(sub)
    _add_generate_seeds_parser(sub)
    _add_reorganize_tps_parser(sub)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        _run_dispatch(args)
    elif args.command == "generate-seeds":
        _generate_seeds_dispatch(args)
    elif args.command == "reorganize-tps":
        _reorganize_tps_dispatch(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
