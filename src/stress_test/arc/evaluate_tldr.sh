#!/bin/bash

# Activate virtual environment
source /home2/hd49/multilingual-social-summary/socialsum-venv/bin/activate

# Run evaluation
python src/stress_test/evaluate_tldr.py \
  --model_dir google/mt5-base \
  --test_data src/stress_test/tldr_test_300.jsonl \
  --output_path results/tldr_test_300_results.jsonl
