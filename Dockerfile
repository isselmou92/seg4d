# ============================================================================
# seg4d — 4D medical image segmentation pipeline
# ============================================================================
# Base:  pytorch/pytorch 2.7.0 with CUDA 12.6 + cuDNN 9
#        (matches requirements-all.txt and setup_all.bat)
#
# Build:
#   docker build -t seg4d:latest .
#
# Run locally (example):
#   docker run --gpus all --rm \
#     -v /path/to/patient:/data \
#     -v /path/to/checkpoints:/checkpoints \
#     -e HF_TOKEN=hf_xxx \
#     seg4d:latest run --data_dir /data --steps all --model sam3
#
# On a SLURM cluster convert to Singularity first:
#   singularity build seg4d.sif docker://youruser/seg4d:latest
# ============================================================================

FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

# ---- system dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        wget \
        curl \
        ca-certificates \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ---- working directory ----
WORKDIR /opt/seg4d

# ---- Python dependencies (excluding PyTorch — already in base image) ----
COPY requirements-all.txt .
RUN pip install --no-cache-dir \
        pydicom>=2.4 \
        nibabel>=5.0 \
        numpy>=1.24 \
        scipy>=1.10 \
        SimpleITK>=2.3 \
        "scikit-image>=0.21" \
        "Pillow>=10.0" \
        "tqdm>=4.65" \
        "matplotlib>=3.7" \
        "TotalSegmentator>=2.4" \
        "transformers>=4.46" \
        "huggingface-hub>=0.26"

# ---- clone and install MedSAM2 (provides the `sam2` Python package) ----
RUN git clone --depth 1 https://github.com/bowang-lab/MedSAM2.git /opt/MedSAM2 \
    && pip install --no-cache-dir -e /opt/MedSAM2

# ---- copy seg4d source ----
COPY seg4d/ ./seg4d/
COPY scripts/ ./scripts/

# ---- expose Python path ----
ENV PYTHONPATH=/opt/seg4d:$PYTHONPATH

# ---- matplotlib headless backend (no display on cluster) ----
ENV MPLBACKEND=Agg

# ---- runtime defaults ----
# Checkpoints and patient data are expected to be bind-mounted at runtime:
#   /data        → patient data directory  (--data_dir /data)
#   /checkpoints → MedSAM2 checkpoint dir
# HF_TOKEN env variable should be set for SAM3 / gated HF models.
ENV MEDSAM2_CKPT=/checkpoints/MedSAM2_MRI_LiverLesion.pt
ENV MEDSAM2_CFG=configs/sam2.1_hiera_t512.yaml

ENTRYPOINT ["python", "-m", "seg4d"]
CMD ["--help"]
