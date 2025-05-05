#!/bin/bash

# Activate your virtual environment
source /home2/hd49/multilingual-social-summary/socialsum-venv/bin/activate

# Launch the fine-tuning script with configurable parameters
python src/model_train/train_mt5.py \
  --data_path data/dev/toy_data_10.jsonl \
  --output_dir checkpoints/mt5_test_run \
  --batch_size 8 \
  --grad_accum_steps 2 \
  --num_epochs 1 \
  --log_steps 100 \
  --num_workers 2 \
  --fp16
