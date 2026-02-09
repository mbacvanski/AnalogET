#!/bin/bash
# Script to run curriculum training for Shakespeare model
# - Initializes from pre-trained weights
# - Increases T_final by 0.01 every 20 epochs (keeping step_size=0.001 constant)
# - This results in n_steps increasing by 10 every 20 epochs

# Configuration
CTX_LENGTH=64
TEMPERATURE=0.8
GEN_CHARS=200

# Source run + file to initialize from
# Accepts `entity/project/run_id` or a full run URL.
# Note: this must point to an actual checkpoint file stored in run files,
# e.g. `checkpoints/model_shakespeare_epoch_0019.npz`.
WANDB_RUN="qpaig/analog-et/REPLACE_RUN_ID"
WANDB_FILE="checkpoints/model_shakespeare_epoch_0000.npz"
WANDB_ROOT="data/wandb_downloads"

# Local config for the resumed curriculum run
CONFIG="20260205_202744/run_config.json"

# Curriculum parameters
# - base_T_final: Starting T_final (0.01 gives n_steps=10 with step_size=0.001)
# - phase_length: Number of epochs before increasing T_final
BASE_T_FINAL=0.01
PHASE_LENGTH=20

echo "=========================================="
echo "Starting Curriculum Training"
echo "=========================================="
echo "W&B source run: $WANDB_RUN"
echo "W&B source file: $WANDB_FILE"
echo "Base T_final: $BASE_T_FINAL (n_steps=${BASE_T_FINAL%.*}0 per phase)"
echo "Phase length: $PHASE_LENGTH epochs"
echo "=========================================="

python model_shakespeare_train.py \
    --config $CONFIG \
    --ctx_length $CTX_LENGTH \
    --temperature $TEMPERATURE \
    --gen_chars $GEN_CHARS \
    --init_wandb_run "$WANDB_RUN" \
    --init_wandb_file "$WANDB_FILE" \
    --init_wandb_root "$WANDB_ROOT" \
    --curriculum_T_final \
    --curriculum_base_T_final $BASE_T_FINAL \
    --curriculum_phase_length $PHASE_LENGTH

echo ""
echo "=========================================="
echo "Curriculum training completed!"
echo "=========================================="
