#!/bin/bash
# ============================================================================
# job_volumetric.sh — Step 2: TotalSegmentator on volumetric series
# Requires GPU.  Results are shared by all dynamic models.
# ============================================================================

#SBATCH --job-name=seg4d_volumetric
#SBATCH --output=logs/volumetric_%j.out
#SBATCH --error=logs/volumetric_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPU_CORES:-8}
#SBATCH --mem=${MEM_GB:-32}G
#SBATCH --gres=gpu:${GPU_TYPE:-1}
#SBATCH --time=${TIME_VOLUMETRIC:-04:00:00}
#SBATCH --partition=${PARTITION:-gpu}
#SBATCH --account=${ACCOUNT}

set -euo pipefail
source "$(dirname "$0")/cluster.env"

echo "===== seg4d volumetric  [job ${SLURM_JOB_ID}] ====="
echo "  Container : $CONTAINER"
echo "  Data dir  : $DATA_DIR"
echo "  Started   : $(date)"

mkdir -p logs

singularity exec \
    --nv \
    --bind "${DATA_DIR}:/data" \
    "${CONTAINER}" \
    python -m seg4d run \
        --data_dir /data \
        --steps volumetric \
        --device gpu

echo "  Finished  : $(date)"
