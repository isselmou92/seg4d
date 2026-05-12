"""Segmentation model registry.

Backends are referenced by string name (``"medsam2"``, ``"sam3"``) so that
the heavy ML imports only happen when a backend is actually selected.
"""

from __future__ import annotations

import importlib
from typing import Any

from .base import SeedSpec, VideoSegmenter, VideoSegments


# {name: "module.path:ClassName"} -- string references avoid importing
# torch/transformers until the user picks a backend.
SEGMENTERS: dict[str, str] = {
    "medsam2": "seg4d.models.medsam2:MedSAM2Segmenter",
    "sam3": "seg4d.models.sam3:SAM3Segmenter",
}


def list_segmenters() -> list[str]:
    """Return the names of all registered video-segmentation backends."""
    return sorted(SEGMENTERS.keys())


def build_segmenter(name: str, **kwargs: Any) -> VideoSegmenter:
    """Instantiate a registered backend by name.

    Raises
    ------
    KeyError
        If ``name`` is not registered.
    """
    if name not in SEGMENTERS:
        raise KeyError(
            f"Unknown segmenter '{name}'. Available: {list_segmenters()}"
        )
    module_path, class_name = SEGMENTERS[name].split(":")
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)


__all__ = [
    "SEGMENTERS",
    "SeedSpec",
    "VideoSegmenter",
    "VideoSegments",
    "build_segmenter",
    "list_segmenters",
]
