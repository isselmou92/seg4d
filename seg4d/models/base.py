"""Common interface for 2D-dynamic video segmentation backends.

Each backend (MedSAM2, SAM3, ...) implements :class:`VideoSegmenter` and is
registered in :mod:`seg4d.models.__init__`. The pipeline driver in
:mod:`seg4d.steps.dynamic` is then completely backend-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SeedSpec:
    """Per-sequence seed prompts handed to a :class:`VideoSegmenter`.

    Attributes
    ----------
    organ_points
        ``{organ_name: (positive_points, negative_points)}`` where each
        point is a ``[x, y]`` list. Both lists may be empty.
    organ_boxes
        Optional ``{organ_name: [x1, y1, x2, y2]}``. Used by backends that
        accept box prompts (MedSAM2).
    organ_masks
        Optional ``{organ_name: 2D bool array}``. Used by MedSAM2 in
        ``reslice`` mode (seed derived from the volumetric segmentation).
    seed_frames
        Frame indices on which the prompts are applied (default: ``[0]``).
        Repeating prompts on multiple frames mitigates drift on long
        sequences.
    """

    organ_points: dict[str, tuple[list[list[int]], list[list[int]]]] = field(default_factory=dict)
    organ_boxes: dict[str, list[int]] | None = None
    organ_masks: dict[str, np.ndarray] | None = None
    seed_frames: list[int] = field(default_factory=lambda: [0])

    @property
    def is_empty(self) -> bool:
        has_points = any(p or n for p, n in self.organ_points.values())
        has_boxes = bool(self.organ_boxes)
        has_masks = bool(self.organ_masks) and any(
            m is not None and m.any() for m in (self.organ_masks or {}).values()
        )
        return not (has_points or has_boxes or has_masks)


# Type alias for tracker output: {frame_idx: {object_id: 2D bool mask}}
VideoSegments = dict[int, dict[int, np.ndarray]]


class VideoSegmenter(ABC):
    """Abstract base class for 2D-dynamic segmentation backends."""

    name: str = "base"

    # Capability flags. Subclasses override these.
    supports_box_prompts: bool = False
    supports_point_prompts: bool = True
    supports_mask_prompts: bool = False

    def __init__(self, **kwargs: Any) -> None:
        self.config = kwargs

    @abstractmethod
    def load(self, device: str) -> None:
        """Materialise model weights and any expensive state on ``device``."""

    @abstractmethod
    def segment_sequence(
        self,
        seq_name: str,
        seq_meta: dict[str, Any],
        seed: SeedSpec,
    ) -> VideoSegments:
        """Run the backend on a single dynamic sequence.

        Parameters
        ----------
        seq_name
            Sequence name (e.g. ``"60_COR_b"``). Backends may use this for
            logging or to select sequence-specific behaviour.
        seq_meta
            The corresponding entry from ``conversion_metadata.json``,
            including ``num_frames``, ``jpeg_dir``, ``geometry``.
        seed
            Pre-built :class:`SeedSpec` containing the prompts to use.

        Returns
        -------
        VideoSegments
            ``{frame_idx: {object_id: 2D bool mask}}``.
        """
