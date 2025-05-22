#!/bin/bash

# Activate virtual environment
source /home2/hd49/multilingual-social-summary/socialsum-venv/bin/activate

# Run CodeSwitch evaluation
python src/stress_test/evaluate_codeswitch.py \
  --model_dir google/mt5-base \
  --test_data src/stress_test/codeswitch_sample_100.jsonl \
  --output_path results/codeswitch_test_100_results.jsonl \
  --target_lang en
