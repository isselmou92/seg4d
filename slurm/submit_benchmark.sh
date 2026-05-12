#!/bin/bash
# ============================================================================
# submit_benchmark.sh — submit the full benchmark pipeline as a SLURM job chain
#
# Usage:
#   bash slurm/submit_benchmark.sh [--dry-run]
#
# Jobs submitted (with --dependency=afterok so each waits for the previous):
#
#   1. convert      DICOM -> NIfTI (CPU)
#   2. volumetric   TotalSegmentator 3D+t (GPU)
#   3a. dynamic     SAM3        boxes   -> results_sam3
#   3b. dynamic     MedSAM2     boxes   -> results_medsam2_boxes
#   3c. dynamic     MedSAM2     reslice -> results_medsam2_reslice
#      (3b and 3c run in parallel after step 2)
#   4. compare      compare_results.py (CPU, runs after all 3x)
#
# Prerequisites:
#   - cluster.env edited with your paths and HF_TOKEN
#   - seed_boxes.json already placed in DATA_DIR  (create it locally first,
#     see slurm/README.md "Interactive seeds" section)
#   - Singularity container built and at $CONTAINER
#   - MedSAM2 checkpoint at $CHECKPOINTS_DIR
# ============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/cluster.env"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[dry-run] No jobs will be submitted."
fi

# Helper: submit or echo
submit() {
    local description="$1"; shift
    if $DRY_RUN; then
        echo "[dry-run] Would submit: sbatch $*"
        echo "999999"          # fake job ID
    else
        local jid
        jid=$(sbatch "$@" | awk '{print $NF}')
        echo "  Submitted ${description}: job ${jid}"
        echo "${jid}"
    fi
}

mkdir -p "${SCRIPT_DIR}/../logs"

echo "========================================================"
echo " seg4d benchmark — full pipeline"
echo "========================================================"
echo "  Container    : ${CONTAINER}"
echo "  Data dir     : ${DATA_DIR}"
echo "  Checkpoints  : ${CHECKPOINTS_DIR}"
echo "  Partition    : ${PARTITION}"
echo ""

# ---- Step 1: convert ----
JID_CONVERT=$(submit "convert" \
    "${SCRIPT_DIR}/job_convert.sh")

# ---- Step 2: volumetric (depends on convert) ----
JID_VOLUMETRIC=$(submit "volumetric" \
    --dependency=afterok:${JID_CONVERT} \
    "${SCRIPT_DIR}/job_volumetric.sh")

# ---- Step 3a: SAM3 boxes (depends on convert only — no volumetric needed) ----
JID_SAM3=$(submit "dynamic_sam3" \
    --dependency=afterok:${JID_CONVERT} \
    --export=ALL,MODEL=sam3,SEED_MODE=boxes,RESULTS_DIR=results_sam3 \
    --job-name=seg4d_sam3 \
    "${SCRIPT_DIR}/job_dynamic.sh")

# ---- Step 3b: MedSAM2 boxes (depends on convert) ----
JID_M2_BOXES=$(submit "dynamic_medsam2_boxes" \
    --dependency=afterok:${JID_CONVERT} \
    --export=ALL,MODEL=medsam2,SEED_MODE=boxes,RESULTS_DIR=results_medsam2_boxes,SEED_FRAMES=0,250,499 \
    --job-name=seg4d_m2_boxes \
    "${SCRIPT_DIR}/job_dynamic.sh")

# ---- Step 3c: MedSAM2 reslice (depends on volumetric — needs TotalSeg masks) ----
JID_M2_RESLICE=$(submit "dynamic_medsam2_reslice" \
    --dependency=afterok:${JID_VOLUMETRIC} \
    --export=ALL,MODEL=medsam2,SEED_MODE=reslice,RESULTS_DIR=results_medsam2_reslice \
    --job-name=seg4d_m2_reslice \
    "${SCRIPT_DIR}/job_dynamic.sh")

# ---- Step 4: compare (depends on all three dynamic jobs) ----
JID_COMPARE=$(submit "compare" \
    --dependency=afterok:${JID_SAM3}:${JID_M2_BOXES}:${JID_M2_RESLICE} \
    "${SCRIPT_DIR}/job_compare.sh")

echo ""
echo "========================================================"
echo " Job chain submitted"
echo "========================================================"
echo ""
echo "  Monitor progress:"
echo "    squeue -u \${USER}"
echo "    tail -f logs/dynamic_seg4d_sam3_*.out"
echo ""
echo "  Cancel all jobs:"
echo "    scancel ${JID_CONVERT} ${JID_VOLUMETRIC} ${JID_SAM3} ${JID_M2_BOXES} ${JID_M2_RESLICE} ${JID_COMPARE}"
echo ""
echo "  Results will appear in:"
echo "    ${DATA_DIR}/results_sam3/"
echo "    ${DATA_DIR}/results_medsam2_boxes/"
echo "    ${DATA_DIR}/results_medsam2_reslice/"
echo "    ${DATA_DIR}/comparison_table.csv"
echo "    ${DATA_DIR}/comparison_chart.png"
