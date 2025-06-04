import os
import json
import random
import argparse

"""
This script slices and shuffles data from the .jsonl TL;DR test set.
To reproduce the same shuffling, use seed 42.

Sample use:
python slice_1k_val_tldr.py \
  --input_path /path/to/tldr_test.jsonl \
  --output_path /path/to/output_path \
  --output_name tldr_val_1k.jsonl \
  --num_examples 1000 \
  --seed 42

"""

def main():
    parser = argparse.ArgumentParser(description="Sample and shuffle a .jsonl file.")
    parser.add_argument("--input_path", required=True, help="Path to input .jsonl file")
    parser.add_argument("--output_path", required=True, help="Directory to save output")
    parser.add_argument("--output_name", default="val_1k.jsonl", help="Name of output .jsonl file")
    parser.add_argument("--num_examples", type=int, default=1000, help="Number of examples to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # === Load input ===
    print(f"🔄 Loading from {args.input_path}")
    with open(args.input_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # === Shuffle and sample ===
    print(f"🔀 Shuffling with seed {args.seed}")
    random.seed(args.seed)
    random.shuffle(data)
    subset = data[:args.num_examples]

    # === Save output ===
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_name)
    print(f"💾 Saving {args.num_examples} examples to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in subset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("✅ Done.")

if __name__ == "__main__":
    main()
