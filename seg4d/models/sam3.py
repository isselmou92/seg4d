"""SAM3 backend - Sam3TrackerVideoModel via HuggingFace Transformers.

This is the same tracker that powers the SAM3 web demo. We use point
prompts only: SAM3 supports text prompts as well, but for organ tracking
on MR data we get more reliable results from a small handful of
positive/negative clicks placed by the user via
:mod:`seg4d.seeds.interactive`.
"""

from __future__ import annotations

import glob
import os
from typing import Any

import numpy as np
from PIL import Image

from ..constants import ORGAN_OBJECT_IDS, ORGANS_OF_INTEREST
from .base import SeedSpec, VideoSegmenter, VideoSegments


def _determine_visible_organs(seq_name: str) -> tuple[str, ...]:
    """Heuristic: which organs to expect in a sequence given its plane.

    Sagittal slabs only catch the left kidney reliably; coronal slabs catch
    all three organs. This avoids feeding SAM3 obviously-impossible prompts.
    """
    upper = seq_name.upper()
    if "SAG" in upper:
        return ("liver", "kidney_left")
    if "COR" in upper:
        return ("liver", "kidney_right", "kidney_left")
    return ORGANS_OF_INTEREST


def _load_jpeg_frames(jpeg_dir: str) -> list[Image.Image]:
    """Load every ``*.jpg`` / ``*.jpeg`` frame in ``jpeg_dir`` as RGB PIL images."""
    frame_paths = sorted(glob.glob(os.path.join(jpeg_dir, "*.jpg")))
    if not frame_paths:
        frame_paths = sorted(glob.glob(os.path.join(jpeg_dir, "*.jpeg")))
    return [Image.open(p).convert("RGB") for p in frame_paths]


class SAM3Segmenter(VideoSegmenter):
    """Wrapper around ``Sam3TrackerVideoModel`` from HuggingFace."""

    name = "sam3"
    supports_box_prompts = False
    supports_point_prompts = True
    supports_mask_prompts = False

    def __init__(
        self,
        hf_model_id: str = "facebook/sam3",
        gpus: list[int] | None = None,
        results_base: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.hf_model_id = hf_model_id
        self.gpus = gpus
        self.results_base = results_base
        self._model = None
        self._processor = None
        self._torch_device = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self, device: str) -> None:
        import torch  # local import keeps CLI startup fast
        from transformers import (  # type: ignore
            Sam3TrackerVideoModel,
            Sam3TrackerVideoProcessor,
        )

        if device == "cuda":
            gpu_id = self.gpus[0] if self.gpus else 0
            self._torch_device = torch.device(f"cuda:{gpu_id}")
        else:
            self._torch_device = torch.device("cpu")

        print(f"  Loading SAM3 Tracker from HuggingFace on {self._torch_device}...")
        self._model = Sam3TrackerVideoModel.from_pretrained(self.hf_model_id).to(
            self._torch_device, dtype=torch.bfloat16
        )
        self._processor = Sam3TrackerVideoProcessor.from_pretrained(self.hf_model_id)
        print("  SAM3 Tracker loaded successfully")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def segment_sequence(
        self,
        seq_name: str,
        seq_meta: dict[str, Any],
        seed: SeedSpec,
    ) -> VideoSegments:
        if self._model is None or self._processor is None:
            raise RuntimeError("SAM3Segmenter.load(...) must be called first.")

        from .base import VideoSegments  # for type stability

        jpeg_dir = seq_meta["jpeg_dir"]
        num_frames = int(seq_meta["num_frames"])
        dyn_geometry = seq_meta["geometry"]
        target_rows = int(dyn_geometry["Rows"])
        target_cols = int(dyn_geometry["Columns"])

        visible = set(_determine_visible_organs(seq_name))

        organ_points: dict[str, tuple[list[list[int]], list[list[int]]]] = {}
        for organ_name, (pos_pts, neg_pts) in seed.organ_points.items():
            if organ_name not in visible:
                continue
            if not pos_pts and not neg_pts:
                continue
            organ_points[organ_name] = (list(pos_pts), list(neg_pts))

        if not organ_points:
            print(f"    WARNING: No point prompts defined for {seq_name}. Skipping.")
            return {}

        for organ_name, (pos_pts, neg_pts) in organ_points.items():
            print(f"    {organ_name}: {len(pos_pts)} positive, "
                  f"{len(neg_pts)} negative points")

        # Optional debug visualization (only when results_base configured)
        if self.results_base:
            from ..seeds.visualize import visualize_seed_points

            visualize_seed_points(jpeg_dir, seq_name, organ_points, self.results_base)

        print(f"    Loading {num_frames} JPEG frames...")
        video_frames = _load_jpeg_frames(jpeg_dir)
        if not video_frames:
            print(f"    WARNING: No JPEG frames found in {jpeg_dir}. Skipping.")
            return {}
        print(f"    Loaded {len(video_frames)} frames ({target_cols}x{target_rows})")

        return self._run_tracker(
            seq_name, video_frames, organ_points, num_frames
        )

    # ------------------------------------------------------------------
    # Tracker glue
    # ------------------------------------------------------------------

    def _run_tracker(
        self,
        seq_name: str,
        video_frames: list[Image.Image],
        organ_points: dict[str, tuple[list[list[int]], list[list[int]]]],
        num_frames: int,
    ) -> VideoSegments:
        import torch  # type: ignore

        processor = self._processor
        model = self._model
        device = self._torch_device

        inference_session = processor.init_video_session(
            video=video_frames,
            inference_device=device,
            dtype=torch.bfloat16,
        )

        obj_ids: list[int] = []
        all_points: list[list[list[int]]] = []
        all_labels: list[list[int]] = []

        for organ_name, (pos_pts, neg_pts) in organ_points.items():
            obj_ids.append(ORGAN_OBJECT_IDS[organ_name])
            points_for_organ = list(pos_pts) + list(neg_pts)
            labels_for_organ = [1] * len(pos_pts) + [0] * len(neg_pts)
            all_points.append(points_for_organ)
            all_labels.append(labels_for_organ)

        # Sam3 expects shape: [batch=1][num_objects][num_points][2]
        input_points = [all_points]
        input_labels = [all_labels]

        print(f"    Adding prompts on frame 0 for {len(obj_ids)} organs: "
              f"{list(organ_points.keys())}")
        processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=0,
            obj_ids=obj_ids,
            input_points=input_points,
            input_labels=input_labels,
        )

        print(f"    Running initial inference on frame 0...")
        _ = model(inference_session=inference_session, frame_idx=0)

        print(f"    Propagating through {num_frames} frames...")
        video_segments: VideoSegments = {}
        frame_count = 0

        for tracker_output in model.propagate_in_video_iterator(inference_session):
            frame_idx = int(tracker_output.frame_idx)
            masks = processor.post_process_masks(
                [tracker_output.pred_masks],
                original_sizes=[[
                    inference_session.video_height,
                    inference_session.video_width,
                ]],
                binarize=True,
            )[0]

            per_obj_masks: dict[int, np.ndarray] = {}
            for i, obj_id in enumerate(inference_session.obj_ids):
                if i < masks.shape[0]:
                    mask = masks[i].cpu().numpy().squeeze().astype(bool)
                    per_obj_masks[int(obj_id)] = mask

            video_segments[frame_idx] = per_obj_masks
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"      Propagated {frame_count}/{num_frames} frames...")

        print(f"    Propagation complete: {frame_count} frames")
        return video_segments
