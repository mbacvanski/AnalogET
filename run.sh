#!/bin/bash
# Example script to run Shakespeare model training with custom configuration

# Example 1: Run with default configuration
# python model_shakespeare_train.py

# Example 2: Run with custom JSON config
# python model_shakespeare_train.py --config shakespeare_config.json

# Example 3: Run with custom config and text generation settings
python model_shakespeare_train.py \
  --config shakespeare_config.json \
  --temperature 0.1 \
  --gen_chars 32
