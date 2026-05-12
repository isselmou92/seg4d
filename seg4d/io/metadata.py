"""Helpers for the central ``conversion_metadata.json`` artifact.

The metadata file is the authoritative description of a converted patient
dataset and is the only piece of state shared between pipeline steps.
"""

from __future__ import annotations

import json
import os
from typing import Any


METADATA_FILENAME = "conversion_metadata.json"


def metadata_path(data_dir: str) -> str:
    """Return the canonical path of the metadata file for a patient dataset."""
    return os.path.join(data_dir, "nifti", METADATA_FILENAME)


def load_metadata(data_dir: str) -> dict[str, Any]:
    """Load ``conversion_metadata.json`` for ``data_dir``.

    Raises
    ------
    FileNotFoundError
        If the file is missing (i.e. step 1 has not been run yet).
    """
    path = metadata_path(data_dir)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"conversion_metadata.json not found at {path}. "
            "Run `seg4d run --steps convert` first."
        )
    with open(path) as f:
        return json.load(f)


def save_metadata(data_dir: str, metadata: dict[str, Any]) -> str:
    """Write ``metadata`` back to ``conversion_metadata.json``."""
    path = metadata_path(data_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    return path
