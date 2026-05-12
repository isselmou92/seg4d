"""Step 3 - 2D dynamic segmentation, model-agnostic.

The driver:

  1. Loads ``conversion_metadata.json``.
  2. Resolves the seed source (interactive picker, JSON file, or reslice).
  3. Builds a per-sequence :class:`~seg4d.models.base.SeedSpec`.
  4. Hands the work to the chosen :class:`~seg4d.models.base.VideoSegmenter`.
  5. Saves per-sequence 2D NIfTI outputs.

Backends (MedSAM2, SAM3) are loaded via the registry in
:mod:`seg4d.models`. The choice of backend determines which seed sources are
admissible: ``reslice`` requires a backend with ``supports_mask_prompts``,
boxes require ``supports_box_prompts``, etc.
"""

from __future__ import annotations

import os
from typing import Any

import nibabel as nib
import numpy as np

from ..constants import ORGAN_OBJECT_IDS
from ..geometry import reslice_volume_at_dynamic_plane
from ..io import load_metadata, save_metadata
from ..models import SeedSpec, VideoSegments, build_segmenter
from ..seeds import (
    interactive_point_picker,
    load_seed_file,
    parse_organ_seed,
    save_seed_file,
)


def _pick_reference_segmentation(seg_paths: list[str]) -> str:
    """Choose the volumetric segmentation with the most labelled voxels."""
    best_path, best_count = seg_paths[0], 0
    for sp in seg_paths:
        if os.path.exists(sp):
            seg_data = np.asanyarray(nib.load(sp).dataobj)
            count = int(np.count_nonzero(seg_data))
            if count > best_count:
                best_count = count
                best_path = sp
    print(f"  Reference seg: {os.path.basename(best_path)} ({best_count} voxels)")
    return best_path


def _build_seed_spec(
    seq_name: str,
    seq_meta: dict[str, Any],
    seed_data: dict[str, Any],
    seed_mode: str,
    seed_frames: list[int],
    ref_seg_path: str | None,
) -> SeedSpec:
    """Resolve seeds for one sequence into a :class:`SeedSpec`."""
    if seed_mode == "reslice":
        if not ref_seg_path:
            return SeedSpec(seed_frames=seed_frames)
        masks = reslice_volume_at_dynamic_plane(ref_seg_path, seq_meta["geometry"])
        return SeedSpec(organ_masks=masks, seed_frames=seed_frames)

    organ_points: dict[str, tuple[list[list[int]], list[list[int]]]] = {}
    organ_boxes: dict[str, list[int]] = {}
    seq_seed = seed_data.get(seq_name) or {}
    for organ_name, organ_cfg in seq_seed.items():
        seed = parse_organ_seed(organ_cfg)
        if seed.is_empty:
            continue
        organ_points[organ_name] = (
            list(seed.positive_points or []),
            list(seed.negative_points or []),
        )
        if seed.box:
            organ_boxes[organ_name] = list(seed.box)

    return SeedSpec(
        organ_points=organ_points,
        organ_boxes=organ_boxes or None,
        seed_frames=seed_frames,
    )


def _save_video_segments_as_nifti(
    seq_name: str,
    seq_meta: dict[str, Any],
    video_segments: VideoSegments,
    results_base: str,
) -> tuple[str, dict[str, str]]:
    """Persist a backend's output as per-organ NIfTI volumes.

    Returns the combined-segmentation path and a mapping from organ name to
    its individual mask path. Mirrors the original pipeline's layout so
    downstream tools keep working.
    """
    from PIL import Image as PILImage

    source_img = nib.load(seq_meta["nifti_path"])
    source_affine = source_img.affine
    target_rows = int(seq_meta["geometry"]["Rows"])
    target_cols = int(seq_meta["geometry"]["Columns"])
    num_frames = int(seq_meta["num_frames"])

    seg_volume = np.zeros(source_img.shape, dtype=np.uint8)
    for frame_idx in range(num_frames):
        if frame_idx not in video_segments:
            continue
        for obj_id, mask in video_segments[frame_idx].items():
            if mask.shape != (target_rows, target_cols):
                resized = np.array(
                    PILImage.fromarray(mask.astype(np.uint8) * 255).resize(
                        (target_cols, target_rows), PILImage.NEAREST
                    )
                ) > 127
            else:
                resized = mask
            seg_volume[:, :, frame_idx][resized.T] = obj_id

    seq_results_dir = os.path.join(results_base, seq_name)
    os.makedirs(seq_results_dir, exist_ok=True)

    seg_path = os.path.join(seq_results_dir, "segmentation_2d.nii.gz")
    nib.save(nib.Nifti1Image(seg_volume, source_affine), seg_path)

    organ_paths: dict[str, str] = {}
    for organ_name, obj_id in ORGAN_OBJECT_IDS.items():
        organ_volume = (seg_volume == obj_id).astype(np.uint8)
        if organ_volume.any():
            organ_path = os.path.join(seq_results_dir, f"{organ_name}.nii.gz")
            nib.save(nib.Nifti1Image(organ_volume, source_affine), organ_path)
            organ_paths[organ_name] = organ_path
            print(f"      {organ_name}: {int(organ_volume.sum())} pixels total")

    return seg_path, organ_paths


def run_dynamic(
    data_dir: str,
    model: str,
    seed_mode: str = "boxes",
    seed_file: str | None = None,
    seed_frames: list[int] | None = None,
    interactive: bool = False,
    backend_kwargs: dict[str, Any] | None = None,
    results_dir: str = "results",
) -> dict[str, Any]:
    """Run the chosen video-segmentation backend on every dynamic sequence.

    Parameters
    ----------
    data_dir
        Patient data directory.
    model
        Backend name (``"medsam2"`` / ``"sam3"``); see
        :func:`seg4d.models.list_segmenters`.
    seed_mode
        ``"boxes"`` for prompts loaded from ``seed_boxes.json`` (the only
        mode SAM3 supports). ``"reslice"`` derives prompts from the
        volumetric segmentation (MedSAM2 only).
    seed_file
        Path to ``seed_boxes.json``. Defaults to ``<data_dir>/seed_boxes.json``.
    seed_frames
        Frames at which to apply the prompts (default ``[0]``). Multi-frame
        seeding (``[0, 250, 500]``) helps prevent drift on long sequences.
    interactive
        If true, open the matplotlib point picker before running and save
        any new points back to ``seed_file``.
    backend_kwargs
        Extra kwargs forwarded to the backend constructor (checkpoints,
        gpu IDs, ...).
    results_dir
        Directory for dynamic outputs, relative to ``data_dir`` or absolute.
        Defaults to ``"results"``. Use a unique name per model run when
        benchmarking (e.g. ``"results_sam3"``, ``"results_medsam2_reslice"``).

    Returns
    -------
    dict
        Per-sequence results dict that is also persisted into
        ``conversion_metadata.json`` under the ``dynamic_results`` key.
    """
    print("\n" + "=" * 60)
    print(f"STEP 3: {model.upper()} on 2D Dynamic Sequences")
    print("=" * 60)

    metadata = load_metadata(data_dir)
    results_base = (
        results_dir if os.path.isabs(results_dir)
        else os.path.join(data_dir, results_dir)
    )
    if seed_file is None:
        seed_file = os.path.join(data_dir, "seed_boxes.json")
    if seed_frames is None:
        seed_frames = [0]

    backend_kwargs = dict(backend_kwargs or {})
    backend_kwargs.setdefault("results_base", results_base)
    segmenter = build_segmenter(model, **backend_kwargs)

    if seed_mode == "reslice" and not segmenter.supports_mask_prompts:
        raise ValueError(
            f"Backend '{model}' does not support reslice (mask) seeds. "
            "Use --seed_mode boxes."
        )

    seed_data: dict[str, Any] = {}
    if seed_mode == "boxes":
        seed_data = load_seed_file(seed_file)
        if interactive:
            print("  Opening interactive point placement GUI...")
            seed_data = interactive_point_picker(metadata, seed_data)
            save_seed_file(seed_file, seed_data)
            print(f"  Seed points saved to {seed_file}")
        if not seed_data and not interactive:
            raise FileNotFoundError(
                f"Seed file not found or empty: {seed_file}. Either provide one "
                "or pass --interactive to place points via GUI."
            )
        print(f"  Seed mode: boxes (from {seed_file})")
    else:
        if not segmenter.supports_mask_prompts:
            raise ValueError(f"Backend '{model}' does not support reslice mode.")
        print("  Seed mode: reslice (from volumetric segmentation)")

    ref_seg_path: str | None = None
    if seed_mode == "reslice":
        if "volumetric" not in metadata or "seg_paths" not in metadata["volumetric"]:
            raise RuntimeError(
                "Volumetric segmentation not found. Run --steps volumetric first."
            )
        ref_seg_path = _pick_reference_segmentation(metadata["volumetric"]["seg_paths"])

    import torch  # local import keeps startup fast when GPU step is skipped
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("  WARNING: No CUDA GPU detected — running on CPU (will be slow).")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  Device: {device}  ({gpu_name})")
    segmenter.load(device=device)

    dynamic_results: dict[str, Any] = metadata.setdefault("dynamic_results", {})

    for seq_name, seq_meta in metadata.get("dynamic", {}).items():
        print(f"\n  [Dynamic] Processing {seq_name}...")
        seed = _build_seed_spec(
            seq_name, seq_meta, seed_data, seed_mode, seed_frames, ref_seg_path
        )
        if seed.is_empty:
            print(f"    No seeds for {seq_name}; skipping.")
            continue

        video_segments = segmenter.segment_sequence(seq_name, seq_meta, seed)
        if not video_segments:
            continue

        seg_path, organ_paths = _save_video_segments_as_nifti(
            seq_name, seq_meta, video_segments, results_base
        )
        dynamic_results[seq_name] = {
            "seg_path": seg_path,
            "organ_paths": organ_paths,
            "num_frames_segmented": len(video_segments),
            "model": model,
        }
        print(f"    Saved: {seg_path} ({len(video_segments)} frames)")

    save_metadata(data_dir, metadata)
    return dynamic_results
