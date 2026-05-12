"""MedSAM2 backend - video predictor from bowang-lab/MedSAM2.

MedSAM2 reuses the SAM2 video API so it accepts box, point and mask seeds.
We support all three; in particular ``mask`` seeds make it possible to
derive prompts directly from the volumetric TotalSegmentator output via
:func:`seg4d.geometry.reslice.reslice_volume_at_dynamic_plane` ("reslice"
seed mode).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..constants import ORGAN_OBJECT_IDS
from .base import SeedSpec, VideoSegmenter, VideoSegments


class MedSAM2Segmenter(VideoSegmenter):
    """MedSAM2 video predictor wrapper."""

    name = "medsam2"
    supports_box_prompts = True
    supports_point_prompts = True
    supports_mask_prompts = True

    def __init__(
        self,
        checkpoint: str = "MedSAM2/checkpoints/MedSAM2_MRI_LiverLesion.pt",
        config: str = "configs/sam2.1_hiera_t512.yaml",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.checkpoint = checkpoint
        self.config = config
        self._predictor = None
        self._device = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self, device: str) -> None:
        from sam2.build_sam import build_sam2_video_predictor  # type: ignore

        print(f"  Loading MedSAM2 from {self.checkpoint} on {device}...")
        self._predictor = build_sam2_video_predictor(
            config_file=self.config,
            ckpt_path=self.checkpoint,
            apply_postprocessing=True,
            device=device,
        )
        self._device = device

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _add_prompts_on_frame(
        predictor,
        inference_state,
        frame_idx: int,
        organ_prompts: dict[str, dict[str, Any]],
    ) -> None:
        """Add box + point prompts for all organs on ``frame_idx``."""
        for organ_name, prompts in organ_prompts.items():
            obj_id = ORGAN_OBJECT_IDS[organ_name]
            box = prompts.get("box")
            pos_pts = prompts.get("positive") or []
            neg_pts = prompts.get("negative") or []

            all_points = list(pos_pts) + list(neg_pts)
            all_labels = [1] * len(pos_pts) + [0] * len(neg_pts)

            box_np = np.array(box, dtype=np.float32) if box else None
            points_np = np.array(all_points, dtype=np.float32) if all_points else None
            labels_np = np.array(all_labels, dtype=np.int32) if all_labels else None

            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                box=box_np,
                points=points_np,
                labels=labels_np,
            )

    @staticmethod
    def _build_organ_prompts(seed: SeedSpec) -> dict[str, dict[str, Any]]:
        """Combine box + point seeds into a single dict per organ."""
        organ_prompts: dict[str, dict[str, Any]] = {}
        organ_names = set(seed.organ_points.keys())
        if seed.organ_boxes:
            organ_names.update(seed.organ_boxes.keys())

        for organ_name in organ_names:
            pos_pts, neg_pts = seed.organ_points.get(organ_name, ([], []))
            box = (seed.organ_boxes or {}).get(organ_name)
            if not pos_pts and not neg_pts and not box:
                continue
            organ_prompts[organ_name] = {
                "box": box,
                "positive": pos_pts,
                "negative": neg_pts,
            }
        return organ_prompts

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def segment_sequence(
        self,
        seq_name: str,
        seq_meta: dict[str, Any],
        seed: SeedSpec,
    ) -> VideoSegments:
        if self._predictor is None:
            raise RuntimeError("MedSAM2Segmenter.load(...) must be called first.")

        import torch  # local import keeps CLI startup fast

        predictor = self._predictor
        jpeg_dir = seq_meta["jpeg_dir"]
        num_frames = int(seq_meta["num_frames"])

        seed_frames = [f for f in seed.seed_frames if f < num_frames]
        if not seed_frames:
            seed_frames = [0]

        organ_prompts = self._build_organ_prompts(seed)
        seed_masks = seed.organ_masks or {}

        if not organ_prompts and not any(m.any() for m in seed_masks.values()):
            print(f"    WARNING: No prompts defined for {seq_name}. Skipping.")
            return {}

        if organ_prompts:
            for organ_name, prompts in organ_prompts.items():
                parts = []
                if prompts.get("box"):
                    parts.append(f"box={prompts['box']}")
                if prompts.get("positive"):
                    parts.append(f"+pts={prompts['positive']}")
                if prompts.get("negative"):
                    parts.append(f"-pts={prompts['negative']}")
                print(f"    Seed {organ_name}: {', '.join(parts)}")
            print(f"    Seed frames: {seed_frames}")
        else:
            for organ, mask in seed_masks.items():
                if mask.any():
                    print(f"    Seed mask {organ}: {int(mask.sum())} pixels")

        print(f"    Running MedSAM2 propagation on {num_frames} frames...")

        autocast_device = self._device if self._device == "cuda" else "cpu"
        autocast_dtype = torch.bfloat16 if self._device == "cuda" else torch.float32

        video_segments: VideoSegments = {}
        with torch.inference_mode(), torch.autocast(autocast_device, dtype=autocast_dtype):
            inference_state = predictor.init_state(
                video_path=jpeg_dir,
                offload_video_to_cpu=True,
                async_loading_frames=False,
            )

            if organ_prompts:
                for frame_idx in seed_frames:
                    self._add_prompts_on_frame(
                        predictor, inference_state, frame_idx, organ_prompts
                    )
            else:
                for organ_name, mask in seed_masks.items():
                    if not mask.any():
                        continue
                    obj_id = ORGAN_OBJECT_IDS[organ_name]
                    predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=obj_id,
                        mask=mask,
                    )

            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state
            ):
                per_obj_mask = {
                    int(obj_id): (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                    for i, obj_id in enumerate(out_obj_ids)
                }
                video_segments[int(out_frame_idx)] = per_obj_mask
                if (out_frame_idx + 1) % 100 == 0:
                    print(f"      Propagated {out_frame_idx + 1}/{num_frames} frames...")

            predictor.reset_state(inference_state)

        return video_segments
