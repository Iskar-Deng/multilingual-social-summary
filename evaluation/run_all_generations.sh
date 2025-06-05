#!/bin/bash

# Get CHECKPOINT_PATH from utils
CHECKPOINT_PATH=$(python3 -c "from utils import CHECKPOINT_PATH; print(CHECKPOINT_PATH)")
if [ ! -d "$CHECKPOINT_PATH" ]; then
  echo "CHECKPOINT_PATH not found: $CHECKPOINT_PATH"
  exit 1
fi

# List of data variants
VARIANTS=("base" "noun" "sent" "full")

echo "Running generation for all variants and checkpoints..."
for variant in "${VARIANTS[@]}"; do
  variant_ckpt_dir="$CHECKPOINT_PATH/mt5_${variant}"
  if [ ! -d "$variant_ckpt_dir" ]; then
    echo "Checkpoint dir missing: $variant_ckpt_dir"
    continue
  fi

  for ckpt in "$variant_ckpt_dir"/checkpoint-*; do
    if [ -d "$ckpt" ]; then
      step=$(basename "$ckpt" | cut -d'-' -f2)
      echo "Generating summaries: $variant - step $step"
      python3 -m evaluation.generate_summary \
        --variant "$variant" \
        --step "$step"
    fi
  done
done
