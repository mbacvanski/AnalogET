#!/bin/bash
# Script to run Shakespeare model training with all configurations in configs/ folder

# Set context length, temperature and generation characters
CTX_LENGTH=64
TEMPERATURE=0.01
GEN_CHARS=64

# Iterate through all config files in the configs directory
for config_file in configs/config_*.json; do
    if [ -f "$config_file" ]; then
        echo "=========================================="
        echo "Running training with config: $config_file"
        echo "=========================================="
        
        python model_shakespeare_train.py \
            --config "$config_file" \
            --ctx_length $CTX_LENGTH \
            --temperature $TEMPERATURE \
            --gen_chars $GEN_CHARS \
        
        echo ""
        echo "Completed: $config_file"
        echo ""
    fi
done

echo "=========================================="
echo "All configurations completed!"
echo "=========================================="
