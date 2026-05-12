"""seg4d - 4D medical image segmentation pipeline.

Public entry points:
  - ``python -m seg4d`` (CLI; see :mod:`seg4d.cli`)
  - :func:`seg4d.pipeline.run_pipeline` for programmatic use.

Sub-packages:
  - :mod:`seg4d.io`         DICOM parsing and NIfTI/JPEG export.
  - :mod:`seg4d.geometry`   Resampling and cross-plane reslicing.
  - :mod:`seg4d.postprocess` Morphological clean-up of organ masks.
  - :mod:`seg4d.seeds`      seed_boxes.json schema, GUI picker, previews.
  - :mod:`seg4d.models`     Pluggable segmentation backends (registry-based).
  - :mod:`seg4d.steps`      Pipeline steps (convert, volumetric, dynamic, reassemble).
  - :mod:`seg4d.reorganize` 4D MR -> RayStation per-phase DICOM exporter.
"""

from . import constants  # noqa: F401

__version__ = "0.1.0"
