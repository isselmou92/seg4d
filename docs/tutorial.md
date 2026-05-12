# Full Benchmark Tutorial

This tutorial walks you through running the complete seg4d benchmark: all three
models (TotalSegmentator, SAM3, MedSAM2) with interactive point placement, storing
each model's results in its own folder and comparing them at the end.

**Platform:** Windows / PowerShell  
**Environment:** single `venv` at `C:\envs\seg4d`

---

## 0 · Prerequisites

| Requirement | Check |
|---|---|
| `venv` created and `requirements-all.txt` installed | `setup\setup_all.bat` completed |
| MedSAM2 checkpoint downloaded | file exists at `MedSAM2\checkpoints\MedSAM2_MRI_LiverLesion.pt` |
| HuggingFace login done | `python -c "from huggingface_hub import whoami; print(whoami()['name'])"` prints your username |
| CUDA GPU visible | `python -c "import torch; print(torch.cuda.get_device_name(0))"` prints GPU name |

Run these once at the start of every session:

```powershell
# activate environment
C:\envs\seg4d\Scripts\activate.bat

# go to the project
cd C:\Users\Isselmou\PycharmProjects\seg4d

# declare your data path (edit to match your folder)
$DATA = "C:\Users\Isselmou\4D_MR\PN001_MR1_MIT"

# declare checkpoint paths
$CKPT = "C:\Users\Isselmou\PycharmProjects\seg4d\MedSAM2\checkpoints\MedSAM2_MRI_LiverLesion.pt"
$CFG  = "configs/sam2.1_hiera_t512.yaml"
```

> **Tip — using a different checkpoint.**  
> If you want the general-purpose checkpoint instead of the liver/lesion one, set
> `$CKPT = "...\MedSAM2_latest.pt"`.

---

## Step 1 · Convert DICOMs to NIfTI  *(run once)*

```powershell
python -m seg4d run --data_dir $DATA --steps convert
```

**What happens:**  
- Every sub-folder inside `$DATA\dicom\` is scanned.  
- Volumetric series (thick-slab, 3D+t) → `$DATA\nifti\<series>\timepoint_NN.nii.gz`.  
- Dynamic series (2D cine) → `$DATA\nifti\<series>\frames.nii.gz` + JPEG previews.  
- `conversion_metadata.json` is written to `$DATA\nifti\`.

**Expected output (excerpt):**
```
STEP 1: DICOM to NIfTI Conversion
  [Volumetric] 2_SAG_b — 48 timepoints x 10 slices
  [Dynamic]    60_COR_b — 500 frames 256x256
  [Dynamic]    60_SAG_b — 500 frames 256x256
  Metadata saved to .../nifti/conversion_metadata.json
```

---

## Step 2 · TotalSegmentator  *(run once — results shared across models)*

```powershell
python -m seg4d run --data_dir $DATA --steps volumetric --device gpu
```

**What happens:**  
- For every timepoint in the volumetric series, TotalSegmentator is run on the
  isotropically-resampled volume, producing a label NIfTI.  
- Results are stored at `$DATA\results\<vol_series>\timepoint_NN_seg.nii.gz`.

**Expected output (excerpt):**
```
STEP 2: TotalSegmentator on Volumetric Data
  [1/48] Segmenting timepoint_00...  liver=4821 vx, kidney_right=312 vx
  ...
  Completed 48 timepoints
```

> TotalSegmentator is not run again in later steps — all models reuse these
> segmentations for reslice-mode seeding.

---

## Step 3 · Place seed points  *(interactive, run once)*

### 3a  Generate reference images

```powershell
python -m seg4d generate-seeds --data_dir $DATA
```

Opens `$DATA\seed_templates\` containing a grid image per dynamic sequence.
Open these in an image viewer to read pixel coordinates for each organ.

### 3b  Open the interactive point picker

```powershell
python -m seg4d run `
  --data_dir $DATA `
  --steps dynamic `
  --model sam3 --seed_mode boxes --interactive `
  --results_dir results_sam3
```

A matplotlib window appears showing the **first frame** of the first dynamic
sequence. The window title shows the active sequence name.

#### Controls

| Key / Action | Effect |
|---|---|
| `1` | Select **liver** (green border) |
| `2` | Select **kidney_right** (blue border) |
| `3` | Select **kidney_left** (orange border) |
| **Left-click** | Add a **positive** point (organ is here) |
| **Right-click** | Add a **negative** point (background) |
| `Z` | Undo the last point |
| `N` | Save current seed and move to next sequence |
| `Q` | Save and quit |

#### Placement strategy

- **`60_COR_b` (coronal):** all three organs are typically visible.  
  Place 1–2 positive points at the organ centre.  
  Place 1 negative point on clearly empty background near each organ.

- **`60_SAG_b` (sagittal):** kidney_right is usually not visible from this plane.  
  Skip it (do not place any points for it) and let the model propagate from
  the coronal sequence.

#### After you close the window

Points are written to `$DATA\seed_boxes.json` and copied to the free-breathing
sequences `FB_COR_b` and `FB_SAG_b` automatically.

> The picker only runs for SAM3 in this step. MedSAM2 will reuse the same
> `seed_boxes.json` automatically.

---

## Step 4 · SAM3 dynamic segmentation

Because Step 3b already ran `dynamic` for SAM3, this step only reassembles the
4D volumes and computes statistics:

```powershell
python -m seg4d run `
  --data_dir $DATA `
  --steps reassemble `
  --model sam3 `
  --results_dir results_sam3
```

Results folder: `$DATA\results_sam3\`

> If Step 3b failed or you want to re-run from scratch:
> ```powershell
> python -m seg4d run `
>   --data_dir $DATA `
>   --steps dynamic,reassemble `
>   --model sam3 --seed_mode boxes `
>   --results_dir results_sam3
> ```

---

## Step 5 · MedSAM2 — box/point seeds

```powershell
python -m seg4d run `
  --data_dir $DATA `
  --steps dynamic,reassemble `
  --model medsam2 --seed_mode boxes --seed_frames 0,250,499 `
  --medsam2_checkpoint $CKPT --medsam2_cfg $CFG `
  --results_dir results_medsam2_boxes
```

**What's different from SAM3:**  
- Reads the same `seed_boxes.json` created in Step 3.  
- Re-applies the seeds on frames 0, 250, and 499 to prevent segmentation drift
  across 500 frames.

Results folder: `$DATA\results_medsam2_boxes\`

---

## Step 6 · MedSAM2 — reslice mode *(fully automatic, no seeds needed)*

```powershell
python -m seg4d run `
  --data_dir $DATA `
  --steps dynamic,reassemble `
  --model medsam2 --seed_mode reslice `
  --medsam2_checkpoint $CKPT --medsam2_cfg $CFG `
  --results_dir results_medsam2_reslice
```

**What's different:**  
- No `seed_boxes.json` required.  
- For each dynamic frame the pipeline reslices the corresponding TotalSegmentator
  3D volume at the plane of the dynamic image and uses the resulting 2D mask as a
  MedSAM2 mask prompt.  
- Fully automatic: useful as a no-human-effort baseline.

Results folder: `$DATA\results_medsam2_reslice\`

---

## Step 7 · Compare all results

```powershell
python scripts\compare_results.py --data_dir $DATA
```

Prints a formatted comparison table to the terminal and saves two files:
- `$DATA\comparison_table.csv` — machine-readable summary
- `$DATA\comparison_chart.png` — bar charts for volumetric volumes and dynamic areas

Pass `--output_dir` to write the files somewhere else:

```powershell
python scripts\compare_results.py --data_dir $DATA --output_dir C:\results\benchmark_01
```

---

## Full command sequence at a glance

```powershell
# 0 — activate
C:\envs\seg4d\Scripts\activate.bat
cd C:\Users\Isselmou\PycharmProjects\seg4d
$DATA = "C:\Users\Isselmou\4D_MR\PN001_MR1_MIT"
$CKPT = "C:\Users\Isselmou\PycharmProjects\seg4d\MedSAM2\checkpoints\MedSAM2_MRI_LiverLesion.pt"
$CFG  = "configs/sam2.1_hiera_t512.yaml"

# 1 — convert (once)
python -m seg4d run --data_dir $DATA --steps convert

# 2 — TotalSegmentator volumetric (once)
python -m seg4d run --data_dir $DATA --steps volumetric --device gpu

# 3 — interactive seeds + SAM3 dynamic
python -m seg4d run --data_dir $DATA --steps dynamic `
  --model sam3 --seed_mode boxes --interactive `
  --results_dir results_sam3

# 3b — SAM3 reassemble
python -m seg4d run --data_dir $DATA --steps reassemble `
  --model sam3 --results_dir results_sam3

# 4 — MedSAM2 boxes
python -m seg4d run --data_dir $DATA --steps dynamic,reassemble `
  --model medsam2 --seed_mode boxes --seed_frames 0,250,499 `
  --medsam2_checkpoint $CKPT --medsam2_cfg $CFG `
  --results_dir results_medsam2_boxes

# 5 — MedSAM2 reslice (automatic)
python -m seg4d run --data_dir $DATA --steps dynamic,reassemble `
  --model medsam2 --seed_mode reslice `
  --medsam2_checkpoint $CKPT --medsam2_cfg $CFG `
  --results_dir results_medsam2_reslice

# 6 — compare
python scripts\compare_results.py --data_dir $DATA
```

---

## Output folder layout

```
<data_dir>/
  nifti/                          NIfTI + JPEG (Step 1)
  results/                        TotalSegmentator 3D volumetric (Step 2)
  results_sam3/                   SAM3 dynamic (Steps 3–3b)
    summary.json
    <seq>/segmentation_2d.nii.gz
  results_medsam2_boxes/          MedSAM2 boxes (Step 4)
    summary.json
    <seq>/segmentation_2d.nii.gz
  results_medsam2_reslice/        MedSAM2 reslice (Step 5)
    summary.json
    <seq>/segmentation_2d.nii.gz
  comparison_table.csv            ← generated by compare_results.py
  comparison_chart.png            ← generated by compare_results.py
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `No dynamic sequences found` | `%DATA%` used instead of `$DATA` | Use PowerShell `$DATA = "..."` syntax |
| `WARNING: No CUDA GPU detected` | torch CUDA build mismatch | `pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126` |
| `huggingface_hub.errors.GatedRepoError` | Not logged in to HF | `python -c "from huggingface_hub import login; login()"` |
| `FileNotFoundError: seed_boxes.json` | Skipped Step 3 | Run with `--interactive` to create the seed file |
| `reslice mode only supported with medsam2` | Wrong `--model` flag | Use `--model medsam2 --seed_mode reslice` |
