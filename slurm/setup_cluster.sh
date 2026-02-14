#!/bin/bash
# One-time setup script — run on the ORCD login node after syncing the code.
#
# Usage (on the cluster):
#   cd ~/analog-denseam
#   bash slurm/setup_cluster.sh

set -euo pipefail

PROJECT_DIR="$HOME/analog-denseam"
cd "$PROJECT_DIR"

# ---------- 1. Install uv if not present ----------
if ! command -v uv &>/dev/null; then
    echo "Installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# ---------- 2. Sync Python environment ----------
echo "Syncing project dependencies with uv ..."
uv sync

# ---------- 3. Prepare Shakespeare dataset ----------
# Compute nodes may lack internet, so download data on the login node.
echo "Preparing Shakespeare dataset ..."
uv run python -c "
import os, sys
os.environ['JAX_PLATFORMS'] = 'cpu'
sys.path.insert(0, '.')
from data import load_shakespeare_dataset, prepare_shakespeare_dataset
if not os.path.exists('data/shakespeare_data_train_X.txt'):
    prepare_shakespeare_dataset(ctx_length=16, train_ratio=0.9, filename_prefix='shakespeare_data')
    print('Dataset prepared.')
else:
    print('Dataset already exists, skipping.')
"

# ---------- 4. W&B login ----------
echo ""
echo "Log in to Weights & Biases (needed for logging):"
uv run wandb login

# ---------- 5. Create log directory ----------
mkdir -p slurm/logs

echo ""
echo "Setup complete. Submit jobs with:  sbatch slurm/submit_sweep.sh"