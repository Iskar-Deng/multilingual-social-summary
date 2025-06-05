# preprocessing/slice_cs.py
# Author: Nathalia Xu

import os
import argparse
import pandas as pd
import random
from utils import DATA_PATH

def main():
    parser = argparse.ArgumentParser(description="Slice a CodeSwitch CSV dataset into smaller random subset.")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of rows to sample")
    parser.add_argument("--ns", action="store_true", help="Disable shuffling")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = os.path.join(DATA_PATH, "cs_main_reddit_corpus.csv")
    output_dir = os.path.join(DATA_PATH, "codeswitch")
    output_path = os.path.join(output_dir, "sliced_cs.csv")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)

    if len(df) < args.num_samples:
        raise ValueError(f"Only {len(df)} rows in file, but {args.num_samples} requested.")

    if args.ns:
        print(f"Shuffling disabled. Taking first {args.num_samples} rows.")
        sliced_df = df.iloc[:args.num_samples]
    else:
        print(f"Shuffling with seed {args.seed}. Sampling {args.num_samples} rows.")
        sliced_df = df.sample(n=args.num_samples, random_state=args.seed)

    sliced_df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()
