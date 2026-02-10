#!/bin/bash
#SBATCH --job-name=shakespeare-train
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%A_%a.out
#SBATCH --error=slurm/logs/%A_%a.err
##SBATCH --array=1-5
##SBATCH --exclude=node2704
#
# Submit single job:        sbatch slurm/submit_run.sh
# Submit 5 parallel jobs:   sbatch --array=1-5 slurm/submit_run.sh
# Submit excluding slow:    sbatch --array=1-5 --exclude=node2704 slurm/submit_run.sh
# Monitor with: squeue -u $USER
# Cancel:       scancel <jobid>

set -euo pipefail

PROJECT_DIR="$HOME/analog-denseam"
cd "$PROJECT_DIR"

export JAX_PLATFORMS=cpu

# Threading configuration - ensures JAX only uses allocated CPUs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${SLURM_CPUS_PER_TASK}"

# Activate the venv directly — do NOT use `uv run` on compute nodes.
# uv tries to lock/modify .venv on every invocation, and when multiple jobs
# hit the same NFS-mounted .venv simultaneously you get stale file handles.
source "$PROJECT_DIR/.venv/bin/activate"

ARRAY_INFO=""
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    ARRAY_INFO=" (array task $SLURM_ARRAY_TASK_ID)"
fi

echo "=== Job $SLURM_JOB_ID$ARRAY_INFO ==="
echo "Host:   $(hostname)"
echo "CPUs:   $SLURM_CPUS_PER_TASK"
echo "Start:  $(date)"
echo ""

# Run the training script
bash run.sh

echo ""
echo "Finished: $(date)"
