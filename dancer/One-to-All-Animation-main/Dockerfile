# Use RunPod PyTorch base image
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

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

# Install Miniconda
RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    /opt/conda/bin/conda clean -ya

# Add conda to PATH
ENV PATH=/opt/conda/bin:$PATH

# Initialize conda for bash shell
RUN /opt/conda/bin/conda init bash

# Accept Conda Terms of Service
RUN /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create conda environment
RUN conda create -n one-to-all python=3.12 -y

# Activate conda environment and install PyTorch
RUN echo "conda activate one-to-all" >> ~/.bashrc && \
    /opt/conda/bin/conda run -n one-to-all pip install --no-cache-dir -U pip setuptools wheel && \
    /opt/conda/bin/conda run -n one-to-all pip install --no-cache-dir \
        torch==2.5.1 \
        torchvision==0.20.1 \
        torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu124

# Install runpod in conda environment
RUN /opt/conda/bin/conda run -n one-to-all pip install --no-cache-dir runpod

# Copy and setup hfd.sh (must be before COPY . to ensure it's available)
COPY hfd.sh /usr/local/bin/hfd.sh
RUN chmod +x /usr/local/bin/hfd.sh

# Copy requirements and install Python packages in conda environment
COPY requirements.txt /app/requirements.txt
RUN /opt/conda/bin/conda run -n one-to-all pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# Install additional dependencies that might be needed
RUN /opt/conda/bin/conda run -n one-to-all pip install --no-cache-dir \
    safetensors \
    pillow \
    tqdm \
    einops \
    decord[av] \
    -i https://mirrors.aliyun.com/pypi/simple/

# Install Flash Attention 3 from source
RUN /opt/conda/bin/conda run -n one-to-all bash -c "\
    pip install packaging -i https://mirrors.aliyun.com/pypi/simple/ && \
    python -c 'import torch; print(f\"PyTorch version: {torch.__version__}\")' && \
    cd /tmp && \
    git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention && \
    MAX_JOBS=4 pip install . --no-build-isolation && \
    cd / && \
    rm -rf /tmp/flash-attention"

# Create directories for models
RUN mkdir -p /app/pretrained_models \
    /app/checkpoints

# Copy project files (needed for download scripts)
COPY . /app/

# Install huggingface_hub for model downloads
RUN /opt/conda/bin/conda run -n one-to-all pip install --no-cache-dir huggingface_hub -i https://mirrors.aliyun.com/pypi/simple/

# Download pretrained models using Python script
RUN cd /app/pretrained_models && \
    /opt/conda/bin/conda run -n one-to-all python download_pretrained_models.py

# Download checkpoints using Python script
RUN cd /app/checkpoints && \
    /opt/conda/bin/conda run -n one-to-all python download_checkpoints.py

# Set environment variables
ENV PATH=/opt/conda/envs/one-to-all/bin:$PATH
ENV CONDA_DEFAULT_ENV=one-to-all
ENV CONDA_PREFIX=/opt/conda/envs/one-to-all
ENV PYTHONPATH=/app:/app/video-generation:${PYTHONPATH:-}
ENV HF_ENDPOINT=https://hf-mirror.com

# Create necessary directories
RUN mkdir -p /tmp /app/output /app/input_cache

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
COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh

# Use run.sh to start services and handler
# Base image entrypoint (/opt/nvidia/nvidia_entrypoint.sh) handles CUDA setup
CMD ["/app/run.sh"]

