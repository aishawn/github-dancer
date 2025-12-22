#!/bin/bash

# One-to-All Animation Environment Installation Script
# Based on: https://github.com/ssj9596/One-to-All-Animation

set -e  # Exit on error

echo "=========================================="
echo "One-to-All Animation Environment Setup"
echo "=========================================="

# Check if conda is installed
CONDA_CMD=""
if command -v conda &> /dev/null; then
    CONDA_CMD="conda"
else
    # Try common conda installation paths
    COMMON_CONDA_PATHS=(
        "$HOME/anaconda3/bin/conda"
        "$HOME/miniconda3/bin/conda"
        "$HOME/conda/bin/conda"
        "/opt/conda/bin/conda"
        "/usr/local/anaconda3/bin/conda"
        "/usr/local/miniconda3/bin/conda"
    )
    
    for conda_path in "${COMMON_CONDA_PATHS[@]}"; do
        if [ -f "$conda_path" ]; then
            CONDA_CMD="$conda_path"
            echo "Found conda at: $conda_path"
            # Add conda to PATH
            export PATH="$(dirname "$conda_path"):$PATH"
            break
        fi
    done
    
    # Try to initialize conda if found but not in PATH
    if [ -z "$CONDA_CMD" ]; then
        # Try to find conda base directory
        if [ -d "$HOME/anaconda3" ]; then
            CONDA_BASE="$HOME/anaconda3"
        elif [ -d "$HOME/miniconda3" ]; then
            CONDA_BASE="$HOME/miniconda3"
        elif [ -d "/opt/conda" ]; then
            CONDA_BASE="/opt/conda"
        fi
        
        if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
            echo "Found conda installation at: $CONDA_BASE"
            echo "Initializing conda..."
            source "$CONDA_BASE/etc/profile.d/conda.sh"
            if command -v conda &> /dev/null; then
                CONDA_CMD="conda"
            fi
        fi
    fi
    
    # If conda still not found, install Miniconda automatically
    if [ -z "$CONDA_CMD" ]; then
        echo ""
        echo "Conda not found. Installing Miniconda..."
        echo "=========================================="
        
        # Determine installation directory (use $HOME/miniconda3 for user installation)
        CONDA_INSTALL_DIR="$HOME/miniconda3"
        
        # Check if already installed but not initialized
        if [ -d "$CONDA_INSTALL_DIR" ] && [ -f "$CONDA_INSTALL_DIR/bin/conda" ]; then
            echo "Found existing Miniconda installation at $CONDA_INSTALL_DIR"
            CONDA_CMD="$CONDA_INSTALL_DIR/bin/conda"
            export PATH="$CONDA_INSTALL_DIR/bin:$PATH"
        else
            # Detect architecture first
            ARCH=$(uname -m)
            if [ "$ARCH" = "x86_64" ]; then
                MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
                MINICONDA_INSTALLER="/tmp/Miniconda3-latest-Linux-x86_64.sh"
            elif [ "$ARCH" = "aarch64" ]; then
                MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
                MINICONDA_INSTALLER="/tmp/Miniconda3-latest-Linux-aarch64.sh"
            else
                echo "Error: Unsupported architecture: $ARCH"
                exit 1
            fi
            
            # Download and install Miniconda
            echo "Downloading Miniconda for $ARCH..."
            
            if command -v curl &> /dev/null; then
                curl -L -o "$MINICONDA_INSTALLER" "$MINICONDA_URL"
            elif command -v wget &> /dev/null; then
                wget -O "$MINICONDA_INSTALLER" "$MINICONDA_URL"
            else
                echo "Error: Neither curl nor wget is available. Please install one of them."
                exit 1
            fi
            
            echo "Installing Miniconda to $CONDA_INSTALL_DIR..."
            # Use -b for batch mode (auto-accept license) and -f to overwrite if exists
            bash "$MINICONDA_INSTALLER" -b -f -p "$CONDA_INSTALL_DIR"
            rm -f "$MINICONDA_INSTALLER"
            
            # Initialize conda
            CONDA_CMD="$CONDA_INSTALL_DIR/bin/conda"
            export PATH="$CONDA_INSTALL_DIR/bin:$PATH"
            "$CONDA_CMD" clean -ya
            
            echo "Miniconda installed successfully!"
        fi
        
        # Initialize conda for this session (must be done after installation)
        if [ -f "$CONDA_INSTALL_DIR/etc/profile.d/conda.sh" ]; then
            source "$CONDA_INSTALL_DIR/etc/profile.d/conda.sh"
            CONDA_CMD="conda"
        else
            # Fallback: add to PATH if conda.sh doesn't exist
            export PATH="$CONDA_INSTALL_DIR/bin:$PATH"
            CONDA_CMD="$CONDA_INSTALL_DIR/bin/conda"
        fi
    fi
fi

# Verify conda works and get base path
if ! $CONDA_CMD --version &> /dev/null; then
    echo "Error: conda command found but not working properly."
    echo "Trying to reinitialize conda..."
    # Try to get base path from CONDA_CMD path
    if [[ "$CONDA_CMD" == *"/bin/conda" ]]; then
        CONDA_BASE_FROM_CMD=$(dirname $(dirname "$CONDA_CMD"))
        if [ -f "$CONDA_BASE_FROM_CMD/etc/profile.d/conda.sh" ]; then
            source "$CONDA_BASE_FROM_CMD/etc/profile.d/conda.sh"
            CONDA_CMD="conda"
        fi
    fi
    # Verify again
    if ! $CONDA_CMD --version &> /dev/null; then
        echo "Error: conda command still not working. Please check your installation."
        exit 1
    fi
fi

echo "Using conda: $($CONDA_CMD --version)"

# Initialize conda if not already initialized
CONDA_BASE=$($CONDA_CMD info --base 2>/dev/null || echo "")
if [ -z "$CONDA_BASE" ] && [[ "$CONDA_CMD" == *"/bin/conda" ]]; then
    # Fallback: derive base from conda command path
    CONDA_BASE=$(dirname $(dirname "$CONDA_CMD"))
fi

if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    # Update CONDA_CMD to use 'conda' if available in PATH
    if command -v conda &> /dev/null; then
        CONDA_CMD="conda"
    fi
fi

# Accept Conda Terms of Service (required for newer conda versions)
echo ""
echo "Accepting Conda Terms of Service..."
$CONDA_CMD tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
$CONDA_CMD tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create conda environment
echo ""
echo "Step 1: Creating conda environment 'one-to-all' with Python 3.12..."
$CONDA_CMD create -n one-to-all python=3.12 -y

# Activate conda environment
echo ""
echo "Step 2: Activating conda environment..."
# CONDA_BASE already set above, reuse it
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
fi
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

