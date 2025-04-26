#!/bin/bash

# Activate your virtual environment
# ❗ Make sure this path matches your venv path
source /home2/hd49/multilingual-social-summary/socialsum-venv/bin/activate

# Launch the fine-tuning script
# You can modify --data_path and --output_dir as needed
python src/model_train/train_mt5.py \
  --data_path data/dev/toy_data_10.jsonl \
  --output_dir checkpoints/mt5_test_run
