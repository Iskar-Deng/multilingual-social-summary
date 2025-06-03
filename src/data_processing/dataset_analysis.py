"""
analyze_hf_token_lengths.py

This script loads a tokenized Hugging Face dataset (all splits), and calculates
statistics (min, max, mean, median, std deviation) for non-padding token lengths
in the 'input_ids' and 'labels' fields.

Outputs:
- A text file summarizing dataset statistics.

Usage:
    python analyze_hf_token_lengths.py <dataset_name> <output_stats_txt>
        [--input_pad_id 0] [--output_pad_id -100]

Example:
    python analyze_hf_token_lengths.py tatsu-lab/alpaca stats.txt --input_pad_id 0 --output_pad_id -100
"""

import argparse
import numpy as np
import pandas as pd
from datasets import load_from_disk, concatenate_datasets
from tabulate import tabulate
from tqdm import tqdm

def count_non_pad(tokens, pad_id):
    return sum(t != pad_id for t in tokens)

def compute_lengths(dataset, input_key="input_ids", output_key="labels",
                    input_pad_id=0, output_pad_id=-100):
    input_lens = []
    output_lens = []
    for example in tqdm(dataset, desc="Analyzing Lengths"):
        input_lens.append(count_non_pad(example[input_key], input_pad_id))
        output_lens.append(count_non_pad(example[output_key], output_pad_id))
    return input_lens, output_lens

def calculate_statistics(lengths):
    arr = np.array(lengths)
    return {
        "Min": int(np.min(arr)),
        "Max": int(np.max(arr)),
        "Mean": round(float(np.mean(arr)), 2),
        "Median": int(np.median(arr)),
        "Std. Deviation": round(float(np.std(arr)), 2),
    }

def write_statistics(output_path, dataset_name, num_examples, input_stats, output_stats):
    df = pd.DataFrame([input_stats, output_stats], index=["Input", "Output"])
    with open(output_path, "w") as f:
        f.write("Token Length Statistics (Non-Padding Tokens)\n\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Total Examples: {num_examples}\n\n")
        f.write(tabulate(df, headers="keys", tablefmt="pretty"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", help="Hugging Face dataset name or local path")
    parser.add_argument("output_stats_txt", help="Path to save the output stats")
    parser.add_argument("--input_pad_id", type=int, default=0,
                        help="Padding token ID for inputs (default: 0)")
    parser.add_argument("--output_pad_id", type=int, default=-100,
                        help="Padding token ID for outputs (default: -100)")
    args = parser.parse_args()

    raw_dset = load_from_disk(args.dataset_name)

    if isinstance(raw_dset, dict):
        dataset = concatenate_datasets(list(raw_dset.values()))
    else:
        dataset = raw_dset

    input_lens, output_lens = compute_lengths(
        dataset,
        input_pad_id=args.input_pad_id,
        output_pad_id=args.output_pad_id
    )

    input_stats = calculate_statistics(input_lens)
    output_stats = calculate_statistics(output_lens)

    write_statistics(args.output_stats_txt, args.dataset_name, len(dataset), input_stats, output_stats)

if __name__ == "__main__":
    main()
