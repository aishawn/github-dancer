#!/bin/bash
# This script runs /start.sh in the background to start Jupyter/SSH services,
# then executes the RunPod serverless handler.
#
# This enables both interactive development (Jupyter/SSH) and serverless processing

echo "=========================================="
echo "Starting RunPod One-to-All-Animation"
echo "=========================================="

# Ensure we're in the correct directory
cd /app || {
    echo "ERROR: Cannot change to /app directory"
    exit 1
}

# Verify conda environment is available
PYTHON_CMD="/opt/conda/envs/one-to-all/bin/python"
if [ ! -f "$PYTHON_CMD" ]; then
    echo "ERROR: Conda environment 'one-to-all' not found at $PYTHON_CMD"
    exit 1
fi

# Verify handler.py exists
if [ ! -f "/app/handler.py" ]; then
    echo "ERROR: handler.py not found at /app/handler.py"
    exit 1
fi

echo "Starting base image services (Jupyter/SSH)..."

# Start base image services (Jupyter/SSH) in background
if [ -f "/start.sh" ]; then
    /start.sh &
    START_PID=$!
    echo "Started /start.sh (PID: $START_PID)"
else
    echo "WARNING: /start.sh not found, skipping Jupyter/SSH startup"
fi

# Wait a moment for services to start
echo "Waiting for services to initialize..."
sleep 3

# Verify Python and handler
echo "Verifying Python environment..."
if ! $PYTHON_CMD --version; then
    echo "ERROR: Python command failed"
    exit 1
fi
echo "Python path: $PYTHON_CMD"

echo "Starting RunPod serverless handler..."
echo "=========================================="

# Run the RunPod serverless handler (this will run continuously)
# Use -u for unbuffered output, exec to replace shell process
exec $PYTHON_CMD -u /app/handler.py
