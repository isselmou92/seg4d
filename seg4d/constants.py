"""Constants shared across the pipeline.

Sequence names follow the folder layout produced by the Philips Ingenia 4D MR
acquisition protocol used during development:

  data_dir/
    dicom/
      2_SAG_b/         <- VOLUMETRIC_SEQUENCE
      60_SAG_b/        |
      60_COR_b/        |- DYNAMIC_SEQUENCES
      FB_SAG_b/        |
      FB_COR_b/        |
      SURVEY/          (ignored)
"""

from __future__ import annotations

VOLUMETRIC_SEQUENCE: str = "2_SAG_b"
DYNAMIC_SEQUENCES: tuple[str, ...] = (
    "60_SAG_b",
    "60_COR_b",
    "FB_SAG_b",
    "FB_COR_b",
)

ORGANS_OF_INTEREST: tuple[str, ...] = ("liver", "kidney_right", "kidney_left")

# Object IDs assigned to organs in the 2D dynamic segmentation outputs.
# Both video tracker backends (MedSAM2, SAM3) use these.
ORGAN_OBJECT_IDS: dict[str, int] = {
    "liver": 1,
    "kidney_right": 2,
    "kidney_left": 3,
}

# Label IDs produced by TotalSegmentator's "total_mr" task.
TOTALSEG_LABEL_MAP: dict[str, int] = {
    "liver": 5,
    "kidney_right": 2,
    "kidney_left": 3,
}

# SAM3 text prompts (currently unused; the active SAM3 backend uses point
# prompts via Sam3TrackerVideoModel, which mirrors the SAM3 web demo). Kept
# for future text-conditioned variants.
SAM3_TEXT_PROMPTS: dict[str, str] = {
    "liver": "liver",
    "kidney_right": "right kidney",
    "kidney_left": "left kidney",
}

# Target near-isotropic spacing used when resampling thick-slab volumetric
# acquisitions before TotalSegmentator. See seg4d.geometry.resample.
TARGET_ISOTROPIC_SPACING_MM: float = 4.0
