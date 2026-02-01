#!/bin/bash
#SBATCH --job-name=shakespeare-sweep
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%A_%a.out
#SBATCH --error=slurm/logs/%A_%a.err
#
# Submit with:  sbatch slurm/submit_sweep.sh
# Monitor with: squeue -u $USER
# Cancel all:   scancel <jobid>

set -euo pipefail

PROJECT_DIR="$HOME/analog-denseam"
cd "$PROJECT_DIR"

# ---------- Auto-submit with correct --array if not already in a SLURM job ----------
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    NUM_CONFIGS=$(ls configs/config_*.json 2>/dev/null | wc -l)
    if [ "$NUM_CONFIGS" -eq 0 ]; then
        echo "ERROR: No config files found in configs/" >&2
        exit 1
    fi
    echo "Found $NUM_CONFIGS configs, submitting array job 1-$NUM_CONFIGS ..."
    exec sbatch --array="1-$NUM_CONFIGS" "$0"
fi

export JAX_PLATFORMS=cpu

# Activate the venv directly — do NOT use `uv run` on compute nodes.
# uv tries to lock/modify .venv on every invocation, and when 10 jobs
# hit the same NFS-mounted .venv simultaneously you get stale file handles.
source "$PROJECT_DIR/.venv/bin/activate"

# ---------- Map array index to config file ----------
CONFIG_ID=$(printf "%03d" "$SLURM_ARRAY_TASK_ID")
CONFIG_FILE="configs/config_${CONFIG_ID}.json"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file $CONFIG_FILE not found" >&2
    exit 1
fi

echo "=== Job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID ==="
echo "Config: $CONFIG_FILE"
echo "Host:   $(hostname)"
echo "CPUs:   $SLURM_CPUS_PER_TASK"
echo "Start:  $(date)"
echo ""

# ---------- Run training ----------
python model_shakespeare_train.py \
    --config "$CONFIG_FILE" \
    --temperature 0.1 \
    --gen_chars 32 \
    --sample_mode

echo ""
echo "Finished: $(date)"