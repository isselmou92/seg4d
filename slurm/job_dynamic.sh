#!/bin/bash
# ============================================================================
# job_dynamic.sh — Steps 3+4: dynamic segmentation + reassemble
#
# Usage (called by submit_benchmark.sh or directly):
#   sbatch --export=ALL,MODEL=sam3,SEED_MODE=boxes,RESULTS_DIR=results_sam3 \
#          slurm/job_dynamic.sh
#
# MODEL        : sam3 | medsam2          (default: sam3)
# SEED_MODE    : boxes | reslice         (default: boxes)
# RESULTS_DIR  : output sub-folder name  (default: results_${MODEL})
# SEED_FRAMES  : comma-separated frame indices for MedSAM2 re-seeding (default: 0,250,499)
#
# NOTE: interactive point placement (--interactive) cannot run on a headless
#       cluster.  Place seed points locally first, then copy seed_boxes.json
#       to DATA_DIR before submitting.  See slurm/README.md for details.
# ============================================================================

#SBATCH --job-name=seg4d_dynamic
#SBATCH --output=logs/dynamic_%x_%j.out
#SBATCH --error=logs/dynamic_%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPU_CORES:-8}
#SBATCH --mem=${MEM_GB:-32}G
#SBATCH --gres=gpu:${GPU_TYPE:-1}
#SBATCH --time=${TIME_DYNAMIC:-06:00:00}
#SBATCH --partition=${PARTITION:-gpu}
#SBATCH --account=${ACCOUNT}

set -euo pipefail
source "$(dirname "$0")/cluster.env"

# --- defaults ---
MODEL=${MODEL:-sam3}
SEED_MODE=${SEED_MODE:-boxes}
RESULTS_DIR=${RESULTS_DIR:-results_${MODEL}}
SEED_FRAMES=${SEED_FRAMES:-0,250,499}

echo "===== seg4d dynamic  [job ${SLURM_JOB_ID}] ====="
echo "  Model       : $MODEL"
echo "  Seed mode   : $SEED_MODE"
echo "  Results dir : $RESULTS_DIR"
echo "  Container   : $CONTAINER"
echo "  Data dir    : $DATA_DIR"
echo "  Started     : $(date)"

mkdir -p logs

# Build the model-specific extra arguments
EXTRA_ARGS=()

if [[ "$MODEL" == "medsam2" ]]; then
    EXTRA_ARGS+=(
        "--medsam2_checkpoint" "${MEDSAM2_CKPT}"
        "--medsam2_cfg"        "${MEDSAM2_CFG}"
        "--seed_frames"        "${SEED_FRAMES}"
    )
elif [[ "$MODEL" == "sam3" ]]; then
    EXTRA_ARGS+=(
        "--sam3_model_id" "facebook/sam3"
    )
fi

singularity exec \
    --nv \
    --bind "${DATA_DIR}:/data" \
    --bind "${CHECKPOINTS_DIR}:/checkpoints" \
    --env  "HF_TOKEN=${HF_TOKEN}" \
    --env  "MPLBACKEND=Agg" \
    "${CONTAINER}" \
    python -m seg4d run \
        --data_dir    /data \
        --steps       dynamic,reassemble \
        --model       "${MODEL}" \
        --seed_mode   "${SEED_MODE}" \
        --results_dir "${RESULTS_DIR}" \
        "${EXTRA_ARGS[@]}"

echo "  Finished  : $(date)"
