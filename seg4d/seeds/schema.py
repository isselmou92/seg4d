"""Schema and parser for ``seed_boxes.json``.

Two formats are supported per organ, both backwards-compatible with the
files used by the original ``segment_4d_mr.py`` pipeline:

  * **legacy box-only** - ``[x1, y1, x2, y2]``. ``[0, 0, 0, 0]`` means
    "organ not visible in this sequence".
  * **extended** - ``{"box": [...], "positive": [[x, y], ...],
    "negative": [[x, y], ...]}``. Any field may be omitted.

A null/None value means "no seed for this organ".
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass
class OrganSeed:
    """Parsed seed for a single organ in a single sequence."""

    box: list[int] | None = None
    positive_points: list[list[int]] | None = None
    negative_points: list[list[int]] | None = None

    @property
    def is_empty(self) -> bool:
        return (
            self.box is None
            and not self.positive_points
            and not self.negative_points
        )


def parse_organ_seed(organ_config: Any) -> OrganSeed:
    """Convert a raw JSON value into an :class:`OrganSeed`."""
    if organ_config is None:
        return OrganSeed()

    if isinstance(organ_config, list):
        if organ_config == [0, 0, 0, 0] or not any(v > 0 for v in organ_config):
            return OrganSeed()
        return OrganSeed(box=list(organ_config))

    if isinstance(organ_config, dict):
        raw_box = organ_config.get("box", [0, 0, 0, 0])
        if isinstance(raw_box, list) and raw_box != [0, 0, 0, 0] and any(v > 0 for v in raw_box):
            box = list(raw_box)
        else:
            box = None

        pos = organ_config.get("positive") or None
        neg = organ_config.get("negative") or None

        seed = OrganSeed(box=box, positive_points=pos, negative_points=neg)
        return OrganSeed() if seed.is_empty else seed

    return OrganSeed()


def load_seed_file(path: str) -> dict[str, dict[str, Any]]:
    """Load a ``seed_boxes.json`` file. Returns ``{}`` when the file is missing."""
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_seed_file(path: str, seed_data: dict[str, dict[str, Any]]) -> None:
    """Atomically write ``seed_data`` to ``path``."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(seed_data, f, indent=2)
