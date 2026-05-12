# Data layout

`seg4d` operates on a **patient data directory** (referred to as `<data_dir>`
throughout the docs). The original DICOM data is the only required input;
every other artifact is derived from it.

## Input

```
<data_dir>/
  dicom/
    2_SAG_b/        <- volumetric (3D + time) sagittal series
    60_SAG_b/       <- 2D dynamic, sagittal, ~60 frames
    60_COR_b/       <- 2D dynamic, coronal, ~60 frames
    FB_SAG_b/       <- 2D dynamic, free-breathing sagittal
    FB_COR_b/       <- 2D dynamic, free-breathing coronal
    SURVEY/         (ignored by seg4d)
```

The volumetric and dynamic sequence names are configurable in
`seg4d/constants.py` (`VOLUMETRIC_SEQUENCE`, `DYNAMIC_SEQUENCES`). Sequences
that are missing on disk are silently skipped.

## After Step 1 (`run --steps convert`)

```
<data_dir>/
  dicom/
  nifti/
    conversion_metadata.json     <- single source of truth for all later steps
    2_SAG_b/
      timepoint_00.nii.gz
      timepoint_01.nii.gz
      ...
    60_COR_b/
      60_COR_b_frames.nii.gz
      60_COR_b_frames_jpg/
        00000.jpg
        00001.jpg
        ...
    60_SAG_b/...
    FB_COR_b/...
    FB_SAG_b/...
```

`conversion_metadata.json` records the geometry of every sequence, the
NIfTI / JPEG paths, and any seg outputs added by later steps.

## After Step 2 (`run --steps volumetric`)

```
<data_dir>/results/2_SAG_b/
  timepoint_00_seg.nii.gz       <- multilabel TotalSegmentator output
  timepoint_00_liver.nii.gz     <- per-organ binary mask (post-processed)
  timepoint_00_kidney_right.nii.gz
  timepoint_00_kidney_left.nii.gz
  ...
```

## After Step 3 (`run --steps dynamic --model {medsam2|sam3}`)

```
<data_dir>/results/<dynamic_seq>/
  segmentation_2d.nii.gz        <- multilabel mask (organ IDs in seg4d.constants)
  liver.nii.gz                  <- per-organ binary mask
  kidney_right.nii.gz
  kidney_left.nii.gz
```

If `--seed_mode boxes` was used (or `--interactive`), `seed_boxes.json` and
the rendered preview images live next to the NIfTI tree:

```
<data_dir>/
  seed_boxes.json
  seed_templates/
    60_COR_b_reference.jpg     <- gridded frame-0 reference
    60_COR_b_boxes.jpg         <- preview of the configured prompts
    ...
```

## After Step 4 (`run --steps reassemble`)

```
<data_dir>/results/
  summary.json
  2_SAG_b/
    segmentation_4d.nii.gz     <- (X, Y, Z, T) label map
    liver_4d.nii.gz
    kidney_right_4d.nii.gz
    kidney_left_4d.nii.gz
```

## After `reorganize-tps`

```
<data_dir>/dicom_4d/
  reorganization_metadata.json
  phase_0.0pctA/
    MR_0.0pctA_0001.dcm
    ...
  phase_25.0pctA/
  phase_50.0pctA/
  phase_75.0pctA/
```

Each phase is a fully independent DICOM series that RayStation imports as a
single 4D plan when the parent folder is selected.
