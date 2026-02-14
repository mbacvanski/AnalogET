#!/bin/bash
# Script to run curriculum training for Shakespeare model
# - Initializes from pre-trained weights
# - Increases T_final by 0.01 every 20 epochs (keeping step_size=0.001 constant)
# - This results in n_steps increasing by 10 every 20 epochs

# Configuration
CTX_LENGTH=64
TEMPERATURE=0.8
GEN_CHARS=200

# Path to pre-trained weights to initialize from
INIT_WEIGHTS="data/shakespeare/20260208_100248/model_shakespeare.npz"

# Curriculum parameters
# - base_T_final: Starting T_final (0.01 gives n_steps=10 with step_size=0.001)
# - phase_length: Number of epochs before increasing T_final
BASE_T_FINAL=0.01
PHASE_LENGTH=20

echo "=========================================="
echo "Starting Curriculum Training"
echo "=========================================="
echo "Init weights: $INIT_WEIGHTS"
echo "Base T_final: $BASE_T_FINAL (n_steps=${BASE_T_FINAL%.*}0 per phase)"
echo "Phase length: $PHASE_LENGTH epochs"
echo "=========================================="

python model_shakespeare_train.py \
    --config configs/config_curriculum.json \
    --ctx_length $CTX_LENGTH \
    --temperature $TEMPERATURE \
    --gen_chars $GEN_CHARS \
    --sample_mode \
    --init_weights "$INIT_WEIGHTS" \
    --curriculum_T_final \
    --curriculum_base_T_final $BASE_T_FINAL \
    --curriculum_phase_length $PHASE_LENGTH

echo ""
echo "=========================================="
echo "Curriculum training completed!"
echo "=========================================="
