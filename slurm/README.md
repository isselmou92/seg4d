# Running seg4d on a SLURM Cluster

Most HPC clusters do not allow Docker directly (it requires root). The standard
approach is to **build** the Docker image on your local machine or a CI server,
then **convert** it to a Singularity/Apptainer image that SLURM can run without
root privileges.

---

## Architecture of the deployment

```
Local machine (Windows/Linux)          SLURM Cluster (Linux)
──────────────────────────────         ────────────────────────────────────
1. Place seed points interactively     /scratch/$USER/
2. docker build → push to registry  →     containers/seg4d.sif  (Singularity)
                                           4D_MR/PN001_MR1_MIT/  (patient data)
                                             dicom/
                                             seed_boxes.json  ← copied from local
                                           checkpoints/
                                             MedSAM2_MRI_LiverLesion.pt
```

---

## Step-by-step

### 1 — Place seed points locally (interactive, not possible on cluster)

On your **local Windows machine**:

```powershell
cd C:\Users\Isselmou\PycharmProjects\seg4d
$DATA = "C:\Users\Isselmou\4D_MR\PN001_MR1_MIT"

# convert DICOMs first so JPEG frames exist
python -m seg4d run --data_dir $DATA --steps convert

# open the interactive picker
python -m seg4d run `
  --data_dir $DATA --steps dynamic `
  --model sam3 --seed_mode boxes --interactive `
  --results_dir results_sam3_local
```

Place your seed points and close the GUI. This writes `$DATA\seed_boxes.json`.

### 2 — Copy data and seeds to the cluster

```bash
# from a Linux/Mac terminal or WSL
rsync -avz --progress \
  /path/to/4D_MR/PN001_MR1_MIT/dicom/ \
  cluster_user@cluster.example.com:/scratch/$USER/4D_MR/PN001_MR1_MIT/dicom/

# copy the seed file
scp /path/to/4D_MR/PN001_MR1_MIT/seed_boxes.json \
    cluster_user@cluster.example.com:/scratch/$USER/4D_MR/PN001_MR1_MIT/
```

From Windows PowerShell (if you have OpenSSH):

```powershell
scp "$DATA\seed_boxes.json" `
    cluster_user@cluster.example.com:/scratch/$env:USER/4D_MR/PN001_MR1_MIT/
```

### 3 — Build the Docker image

On a machine with Docker installed (your local machine or a build server):

```bash
cd /path/to/seg4d          # the folder containing Dockerfile

docker build -t seg4d:latest .

# push to Docker Hub (or another registry your cluster can reach)
docker tag seg4d:latest youruser/seg4d:latest
docker push youruser/seg4d:latest
```

> **Tip:** Docker Hub free tier allows one private repository. Alternatively
> use GitHub Container Registry (`ghcr.io`) with a personal access token.

### 4 — Build the Singularity image on the cluster

```bash
ssh cluster_user@cluster.example.com

mkdir -p /scratch/${USER}/containers
cd /scratch/${USER}/containers

# pull from Docker Hub and convert
singularity build seg4d.sif docker://youruser/seg4d:latest
```

If your cluster has Apptainer instead of Singularity, replace `singularity`
with `apptainer` — the interface is identical.

### 5 — Download MedSAM2 checkpoints on the cluster

```bash
mkdir -p /scratch/${USER}/checkpoints

# huggingface-cli is available inside the container
singularity exec /scratch/${USER}/containers/seg4d.sif \
    huggingface-cli download wanglab/MedSAM2 \
        MedSAM2_MRI_LiverLesion.pt \
        --local-dir /scratch/${USER}/checkpoints
```

### 6 — Edit cluster.env

```bash
cd /path/to/seg4d/slurm    # after cloning the repo on the cluster
nano cluster.env            # or vi
```

Key fields to update:

```bash
export CONTAINER=/scratch/${USER}/containers/seg4d.sif
export DATA_DIR=/scratch/${USER}/4D_MR/PN001_MR1_MIT
export CHECKPOINTS_DIR=/scratch/${USER}/checkpoints
export HF_TOKEN=hf_YOUR_TOKEN_HERE  # from huggingface.co/settings/tokens
export PARTITION=gpu                 # partition name on your cluster
export ACCOUNT=your_project_account
export GPU_TYPE=a100                 # leave blank for any available GPU
```

### 7 — Submit the full benchmark

```bash
# dry run first — shows what would be submitted without running anything
bash slurm/submit_benchmark.sh --dry-run

# submit for real
bash slurm/submit_benchmark.sh
```

Expected job chain:

```
convert ──► volumetric ──────────────────────────────────► compare
       └──► sam3 (boxes) ──────────────────────────────────►
       └──► medsam2 (boxes) ──────────────────────────────►
            (volumetric) ──► medsam2 (reslice) ──────────►
```

### 8 — Monitor and collect results

```bash
# watch running jobs
squeue -u ${USER}

# tail live log for a specific job
tail -f logs/dynamic_seg4d_sam3_<jobid>.out

# once compare job finishes, copy results back
scp cluster_user@cluster.example.com:/scratch/$USER/4D_MR/PN001_MR1_MIT/comparison_table.csv ./
scp cluster_user@cluster.example.com:/scratch/$USER/4D_MR/PN001_MR1_MIT/comparison_chart.png ./
```

---

## Running a single model manually

You can also submit individual steps without using `submit_benchmark.sh`:

```bash
# just SAM3
sbatch \
  --export=ALL,MODEL=sam3,SEED_MODE=boxes,RESULTS_DIR=results_sam3 \
  --job-name=seg4d_sam3 \
  slurm/job_dynamic.sh

# just MedSAM2 reslice
sbatch \
  --export=ALL,MODEL=medsam2,SEED_MODE=reslice,RESULTS_DIR=results_medsam2_reslice \
  --job-name=seg4d_m2_reslice \
  slurm/job_dynamic.sh
```

---

## Customising resources

All resource parameters live in `cluster.env`. You can override them per-job
on the command line:

```bash
sbatch --time=08:00:00 --mem=64G \
  --export=ALL,MODEL=medsam2,SEED_MODE=boxes,RESULTS_DIR=results_medsam2_boxes \
  slurm/job_dynamic.sh
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `FATAL: container file not found` | Wrong `$CONTAINER` path | Check `cluster.env`; run `ls $CONTAINER` |
| `No module named sam2` | Container not rebuilt after Dockerfile change | Rebuild with `docker build` and re-convert to `.sif` |
| `GatedRepoError` for SAM3 | Missing or expired `HF_TOKEN` | Update `HF_TOKEN` in `cluster.env` |
| `No dynamic sequences found` | `convert` step not run yet | Check `job_convert` completed; inspect `logs/convert_*.out` |
| `FileNotFoundError: seed_boxes.json` | Seed file not copied to cluster | Copy `seed_boxes.json` from local machine (see Step 2) |
| Job stays in PD (pending) | Wrong partition or account | Check `--partition` and `--account` in `cluster.env` |
| Out of GPU memory | Model too large for GPU VRAM | Use a larger GPU or reduce batch size in the model config |
