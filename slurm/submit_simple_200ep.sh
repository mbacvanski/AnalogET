#!/bin/bash
#SBATCH --job-name=shk-simple-200ep
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=72:00:00
#SBATCH --output=slurm/logs/%j.out
#SBATCH --error=slurm/logs/%j.err
#
# Submit with:  sbatch slurm/submit_simple_200ep.sh
# Monitor with: squeue -u $USER
# Cancel:       scancel <jobid>

set -euo pipefail

PROJECT_DIR="$HOME/analog-denseam"
cd "$PROJECT_DIR"

export JAX_PLATFORMS=cpu

# Activate the venv directly (avoid uv lock contention on NFS).
source "$PROJECT_DIR/.venv/bin/activate"

CONFIG_FILE="configs/shakespeare_200ep.json"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file $CONFIG_FILE not found" >&2
    exit 1
fi

echo "=== Job $SLURM_JOB_ID ==="
echo "Config: $CONFIG_FILE"
echo "Host:   $(hostname)"
echo "CPUs:   $SLURM_CPUS_PER_TASK"
echo "Start:  $(date)"
echo ""

# ---------- Run training ----------
python model_shakespeare_train_simple.py \
    --config "$CONFIG_FILE" \
    --temperature 0.8 \
    --gen_chars 32

echo ""
echo "Finished: $(date)"