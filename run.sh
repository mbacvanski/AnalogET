#!/bin/bash
# Script to run training for Shakespeare model

# Configuration
CTX_LENGTH=64
TEMPERATURE=0.8
GEN_CHARS=200

# Path to pre-trained weights to initialize from
CONFIG="configs/config_256_10steps.json"


echo "=========================================="
echo "Starting Training"
echo "=========================================="
echo "Init weights: $INIT_WEIGHTS"
echo "Base T_final: $BASE_T_FINAL (n_steps=${BASE_T_FINAL%.*}0 per phase)"
echo "Phase length: $PHASE_LENGTH epochs"
echo "=========================================="

python model_shakespeare_train.py \
    --config $CONFIG \
    --ctx_length $CTX_LENGTH \
    --temperature $TEMPERATURE \
    --gen_chars $GEN_CHARS \

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
