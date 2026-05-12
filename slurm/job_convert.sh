#!/bin/bash
# ============================================================================
# job_convert.sh — Step 1: convert DICOMs to NIfTI
# CPU-only job; no GPU needed.
# ============================================================================

#SBATCH --job-name=seg4d_convert
#SBATCH --output=logs/convert_%j.out
#SBATCH --error=logs/convert_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPU_CORES:-8}
#SBATCH --mem=${MEM_GB:-16}G
#SBATCH --time=${TIME_CONVERT:-01:00:00}
#SBATCH --partition=${PARTITION:-gpu}
#SBATCH --account=${ACCOUNT}

set -euo pipefail
source "$(dirname "$0")/cluster.env"

echo "===== seg4d convert  [job ${SLURM_JOB_ID}] ====="
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
        --steps convert

echo "  Finished  : $(date)"
