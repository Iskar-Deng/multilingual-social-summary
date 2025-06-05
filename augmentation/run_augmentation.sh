#!/bin/bash

echo "Running full-text translation..."
python augmentation/trans_full.py --use_gpu

echo "Running sentence-level translation..."
python augmentation/trans_sent.py --use_gpu

echo "Running noun-level translation..."
python augmentation/trans_noun.py --use_gpu

echo "All augmentations completed."
