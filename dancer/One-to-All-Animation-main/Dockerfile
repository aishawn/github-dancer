# Use RunPod PyTorch base image
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /

# Install system dependencies
RUN apt-get update --yes && \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
    bash \
    git \
    wget \
    curl \
    aria2 \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    bzip2 \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, wheel
RUN pip install --no-cache-dir -U pip setuptools wheel

# Install PyTorch
RUN pip install --no-cache-dir \
        torch==2.5.1 \
        torchvision==0.20.1 \
        torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu124

# Install runpod
RUN pip install --no-cache-dir runpod

# Copy and setup hfd.sh (must be before COPY . to ensure it's available)
COPY hfd.sh /usr/local/bin/hfd.sh
RUN chmod +x /usr/local/bin/hfd.sh

# Copy requirements and install Python packages
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# Install additional dependencies that might be needed
RUN pip install --no-cache-dir \
    safetensors \
    pillow \
    tqdm \
    einops \
    decord[av] \
    -i https://mirrors.aliyun.com/pypi/simple/

# Install Flash Attention 3 from source
RUN pip install packaging -i https://mirrors.aliyun.com/pypi/simple/ && \
    python -c 'import torch; print(f"PyTorch version: {torch.__version__}")' && \
    cd /tmp && \
    git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention && \
    MAX_JOBS=4 pip install . --no-build-isolation && \
    cd / && \
    rm -rf /tmp/flash-attention

# Set environment variables (before model downloads)
ENV PYTHONPATH=/:/video-generation
ENV HF_ENDPOINT=https://hf-mirror.com

# Create directories for models
RUN mkdir -p /pretrained_models \
    /checkpoints

# Copy project files (needed for download scripts)
COPY . /

# Install huggingface_hub for model downloads
RUN pip install --no-cache-dir huggingface_hub -i https://mirrors.aliyun.com/pypi/simple/

# Download pretrained models using Python script
RUN cd /pretrained_models && \
    python download_pretrained_models.py

# Download checkpoints using Python script
RUN cd /checkpoints && \
    python download_checkpoints.py

# Create necessary directories
RUN mkdir -p /tmp /output /input_cache

# ============================================================================
# SERVICE STARTUP & ENTRYPOINT
# ============================================================================
# STARTUP OPTION 2: Run app after services (Jupyter + SSH + Custom app)
# Use this for: Keep services running + run your application in parallel
# Behavior:
#   - Entrypoint: /opt/nvidia/nvidia_entrypoint.sh (CUDA setup) - from base image
#   - CMD: Runs run.sh which starts /start.sh in background, then your app
#   - CUDA support: Enabled via base image entrypoint
#   - Jupyter/SSH: Available for interactive development
#   - Handler: Runs continuously for serverless processing

# Copy and setup run.sh script
COPY run.sh /run.sh
RUN chmod +x /run.sh

# Use run.sh to start services and handler
# Base image entrypoint (/opt/nvidia/nvidia_entrypoint.sh) handles CUDA setup
CMD ["/run.sh"]

