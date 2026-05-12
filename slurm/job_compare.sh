#!/bin/bash
# ============================================================================
# job_compare.sh — Final step: compare all model results
# CPU-only; fast.
# ============================================================================

#SBATCH --job-name=seg4d_compare
#SBATCH --output=logs/compare_%j.out
#SBATCH --error=logs/compare_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=${TIME_COMPARE:-00:30:00}
#SBATCH --partition=${PARTITION:-gpu}
#SBATCH --account=${ACCOUNT}

set -euo pipefail
source "$(dirname "$0")/cluster.env"

echo "===== seg4d compare  [job ${SLURM_JOB_ID}] ====="
echo "  Data dir  : $DATA_DIR"
echo "  Started   : $(date)"

mkdir -p logs

singularity exec \
    --bind "${DATA_DIR}:/data" \
    "${CONTAINER}" \
    python scripts/compare_results.py \
        --data_dir  /data \
        --output_dir /data

echo "  Comparison table : ${DATA_DIR}/comparison_table.csv"
echo "  Comparison chart : ${DATA_DIR}/comparison_chart.png"
echo "  Finished  : $(date)"
