"""
This script randomly samples a specified number of rows (default 10,000) from a CSV file
and saves the result to a new CSV file.

Sample usage:
python slice_10k_csv_codeswitch.py \
  --input_path /path/to/input.csv \
  --output_path /path/to/output.csv \
  --num_samples 10000 \
  --seed 42
"""

import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Randomly sample 10k rows from a CSV file.")
    parser.add_argument("--input_path", required=True, help="Path to the input CSV file")
    parser.add_argument("--output_path", required=True, help="Path to save the sampled output CSV")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of rows to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Load CSV
    print(f"🔄 Loading: {args.input_path}")
    df = pd.read_csv(args.input_path)

    # Sample
    print(f"🔀 Sampling {args.num_samples} rows with seed {args.seed}")
    sampled_df = df.sample(n=args.num_samples, random_state=args.seed)

    # Save output
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    sampled_df.to_csv(args.output_path, index=False)
    print(f"✅ Saved to: {args.output_path}")

if __name__ == "__main__":
    main()
