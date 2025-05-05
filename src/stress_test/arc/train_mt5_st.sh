#!/bin/bash

# Activate your virtual environment
source socialsum-venv/bin/activate

# Launch the fine-tuning script with configurable parameters
python src/model_train/train_mt5.py \
  --data_path src/stress_test/tldr_train_3000.jsonl \
  --output_dir checkpoints/mt5_st \
  --batch_size 16 \
  --grad_accum_steps 2 \
  --num_epochs 3 \
  --log_steps 100 \
  --num_workers 2 \
