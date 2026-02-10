#!/bin/bash
# Script to run curriculum training for Shakespeare model
# - Initializes from pre-trained weights
# - Increases T_final by 0.01 every 20 epochs (keeping step_size=0.001 constant)
# - This results in n_steps increasing by 10 every 20 epochs

# Generation parameters
TEMPERATURE=0.8
GEN_CHARS=200

# Source run + epoch to initialize from (file auto-derived as
# checkpoints/model_shakespeare_epoch_<epoch:04d>.npz).
WANDB_RUN_ID="REPLACE_RUN_ID"
WANDB_EPOCH=0
WANDB_ENTITY="qpaig"
WANDB_PROJECT="analog-et"
WANDB_ROOT="data/wandb_downloads"

# Curriculum parameters
# - base_T_final: Starting T_final (0.01 gives n_steps=10 with step_size=0.001)
# - phase_length: Number of epochs before increasing T_final
BASE_T_FINAL=0.01
PHASE_LENGTH=20

echo "=========================================="
echo "Starting Curriculum Training"
echo "=========================================="
echo "W&B source run: $WANDB_ENTITY/$WANDB_PROJECT/$WANDB_RUN_ID"
echo "W&B source epoch: $WANDB_EPOCH"
echo "Model config: inferred from W&B run config"
echo "ctx_length: inferred from W&B run config"
echo "Base T_final: $BASE_T_FINAL (n_steps=${BASE_T_FINAL%.*}0 per phase)"
echo "Phase length: $PHASE_LENGTH epochs"
echo "=========================================="

python model_shakespeare_train.py \
    --temperature $TEMPERATURE \
    --gen_chars $GEN_CHARS \
    --init_wandb_run_id "$WANDB_RUN_ID" \
    --init_wandb_epoch "$WANDB_EPOCH" \
    --init_wandb_entity "$WANDB_ENTITY" \
    --init_wandb_project "$WANDB_PROJECT" \
    --init_wandb_root "$WANDB_ROOT" \
    --curriculum_T_final \
    --curriculum_base_T_final $BASE_T_FINAL \
    --curriculum_phase_length $PHASE_LENGTH

echo ""
echo "=========================================="
echo "Curriculum training completed!"
echo "=========================================="
