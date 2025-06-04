"""
This script extracts a subset of examples (default: 100k) from a 90% training split of the TL;DR dataset.
It uses a provided list of original data indices (`indices_90.json`) to map and select entries
from a HuggingFace tokenized dataset on disk. Use seed 42 for reproducibility.

This might not be random selection. Double check.

Sample usage:
python slice_100k_tokenized_tldr.py \
  --indices_path /path/to/indices_90.json \
  --tokenized_dataset_path /path/to/tokenized_tldr_train \
  --output_path /path/to/output_subset \
  --num_examples 100000 \
  --seed 42
"""

import argparse
import json
import random
from datasets import load_from_disk

def main():
    parser = argparse.ArgumentParser(description="Extract 100k random examples from a tokenized 90% TL;DR dataset.")
    parser.add_argument("--indices_path", type=str, required=True, help="Path to indices_90.json (random indices for splitting 90% data of original TL;DR).")
    parser.add_argument("--tokenized_dataset_path", type=str, required=True, help="Path to tokenized 90% dataset (HuggingFace format).")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save the 100k subset.")
    parser.add_argument("--num_examples", type=int, default=100000, help="Number of examples to extract (default: 100000).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    # Load the original 90% indices from .txt (one index per line)
    with open(args.indices_path, "r") as f:
        indices_90 = [int(line.strip()) for line in f if line.strip().isdigit()]


    if len(indices_90) < args.num_examples:
        raise ValueError(f"Only {len(indices_90)} examples available, can't select {args.num_examples}.")

    # Shuffle and select
    random.seed(args.seed)
    random.shuffle(indices_90)
    selected_orig_indices = set(indices_90[:args.num_examples])

    # Map original index → position in tokenized dataset
    index_to_tokenized_pos = {orig_idx: i for i, orig_idx in enumerate(indices_90)}
    selected_tokenized_positions = [
        index_to_tokenized_pos[idx]
        for idx in selected_orig_indices
        if idx in index_to_tokenized_pos
    ]

    # Load tokenized dataset and select subset
    dataset = load_from_disk(args.tokenized_dataset_path)
    subset = dataset.select(selected_tokenized_positions)

    # Save
    subset.save_to_disk(args.output_path)
    print(f"✅ Saved {len(subset)} examples to {args.output_path}")

if __name__ == "__main__":
    main()
