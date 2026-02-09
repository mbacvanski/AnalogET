#!/bin/bash
#SBATCH --job-name=shakespeare-train
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%j.out
#SBATCH --error=slurm/logs/%j.err
#
# Submit with:  sbatch slurm/submit_run.sh
# Monitor with: squeue -u $USER
# Cancel:       scancel <jobid>

set -euo pipefail

PROJECT_DIR="$HOME/analog-denseam"
cd "$PROJECT_DIR"

export JAX_PLATFORMS=cpu

# Activate the venv directly — do NOT use `uv run` on compute nodes.
# uv tries to lock/modify .venv on every invocation, and when multiple jobs
# hit the same NFS-mounted .venv simultaneously you get stale file handles.
source "$PROJECT_DIR/.venv/bin/activate"

echo "=== Job $SLURM_JOB_ID ==="
echo "Host:   $(hostname)"
echo "CPUs:   $SLURM_CPUS_PER_TASK"
echo "Start:  $(date)"
echo ""

# Run the training script
bash run.sh

echo ""
echo "Finished: $(date)"