#!/bin/bash

# One-to-All Animation Environment Installation Script
# Based on: https://github.com/ssj9596/One-to-All-Animation

set -e  # Exit on error

echo "=========================================="
echo "One-to-All Animation Environment Setup"
echo "=========================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create conda environment
echo ""
echo "Step 1: Creating conda environment 'one-to-all' with Python 3.12..."
conda create -n one-to-all python=3.12 -y

# Activate conda environment
echo ""
echo "Step 2: Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate one-to-all

# Upgrade pip, setuptools, wheel
echo ""
echo "Step 3: Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

# Install PyTorch
echo ""
echo "Step 4: Installing PyTorch 2.5.1 with CUDA 12.4 support..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install Python dependencies
echo ""
echo "Step 5: Installing Python dependencies from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found in current directory."
    echo "Please make sure you are running this script from the project root directory."
    exit 1
fi

# Install additional dependencies
echo ""
echo "Step 6: Installing additional dependencies..."
pip install safetensors pillow tqdm decord[av]

# Flash Attention (Optional) - Install from source
echo ""
echo "Step 7: Flash Attention Installation (Optional)"
echo "=========================================="
read -p "Do you want to install Flash Attention 3 from source? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing Flash Attention 3 from source..."
    
    # Save current directory
    ORIGINAL_DIR=$(pwd)
    
    # Install required dependencies for compilation
    echo "Installing build dependencies (packaging)..."
    pip install packaging
    
    # Clone flash-attention repository
    FLASH_ATTN_DIR="/tmp/flash-attention"
    if [ -d "$FLASH_ATTN_DIR" ]; then
        echo "Flash-attention directory exists, removing old version..."
        rm -rf "$FLASH_ATTN_DIR"
    fi
    
    echo "Cloning flash-attention repository..."
    git clone https://github.com/Dao-AILab/flash-attention.git "$FLASH_ATTN_DIR"
    cd "$FLASH_ATTN_DIR"
    
    # Install from source (using pip install method like Dockerfile)
    echo "Compiling and installing Flash Attention 3 from source..."
    echo "Note: This may take several minutes and requires CUDA toolkit..."
    MAX_JOBS=4 pip install . --no-build-isolation
    
    # Clean up and return to original directory
    cd "$ORIGINAL_DIR"
    rm -rf "$FLASH_ATTN_DIR"
    echo "Flash Attention 3 installed successfully!"
else
    echo "Skipping Flash Attention installation."
    echo "You can install it later by running:"
    echo "  git clone https://github.com/Dao-AILab/flash-attention.git"
    echo "  cd flash-attention"
    echo "  MAX_JOBS=4 pip install . --no-build-isolation"
fi

echo ""
echo "=========================================="
echo "Installation completed!"
echo "=========================================="
echo ""
echo "To activate the environment in the future, run:"
echo "  conda activate one-to-all"
echo ""

