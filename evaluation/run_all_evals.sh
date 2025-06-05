#!/bin/bash

# Get CHECKPOINT_PATH from utils
CHECKPOINT_PATH=$(python -c "from utils import CHECKPOINT_PATH; print(CHECKPOINT_PATH)")
if [ ! -d "$CHECKPOINT_PATH" ]; then
  echo "CHECKPOINT_PATH not found: $CHECKPOINT_PATH"
  exit 1
fi

# List of data variants
VARIANTS=("base" "noun" "sent" "full")

echo "Running TL;DR BERTScore evaluations"
for variant in "${VARIANTS[@]}"; do
  variant_ckpt_dir="$CHECKPOINT_PATH/mt5_${variant}"
  if [ ! -d "$variant_ckpt_dir" ]; then
    echo "Checkpoint dir missing: $variant_ckpt_dir"
    continue
  fi

  for ckpt in "$variant_ckpt_dir"/checkpoint-*; do
    if [ -d "$ckpt" ]; then
      step=$(basename "$ckpt" | cut -d'-' -f2)
      echo "Evaluating BERTScore: $variant - step $step"
      python -m evaluation.run_BERT \
        --variant "$variant" \
        --step "$step"
    fi
  done
done

echo ""
echo "Running Code-Switch LaSE evaluations"
for variant in "${VARIANTS[@]}"; do
  variant_ckpt_dir="$CHECKPOINT_PATH/mt5_${variant}"
  if [ ! -d "$variant_ckpt_dir" ]; then
    echo "Checkpoint dir missing: $variant_ckpt_dir"
    continue
  fi

  for ckpt in "$variant_ckpt_dir"/checkpoint-*; do
    if [ -d "$ckpt" ]; then
      step=$(basename "$ckpt" | cut -d'-' -f2)
      echo "Evaluating LaSE: $variant - step $step"
      python -m evaluation.run_LaSE \
        --variant "$variant" \
        --step "$step"
    fi
  done
done
