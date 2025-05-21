"""
generate_tldr_dataset.py

This script processes a TL;DR dataset in JSONL format and performs two tasks:
1. Tokenizes the 'content' and 'summary' fields using a specified tokenizer (default: google/mt5-base).
2. Calculates basic statistics (min, max, mean, median, std deviation) for the tokenized input and output lengths.

Outputs:
- A new JSONL file where each line contains a tokenized input-output pair.
- A text file summarizing dataset statistics.

Usage (command line):
    python src/data_processing/analyze_tldr.py <input_jsonl> <output_Dataset> <output_stats_txt>

Arguments:
    input_jsonl: Path to the original TL;DR dataset (.jsonl)
    output_Dataset: Path to save the dataset(Dataset)
    output_stats_txt: Path to save the dataset statistics (.txt)

Example:
    python src/data_processing/generate_tldr_dataset.py data/corpus-webis-tldr-17.json data/tldr_dataset stats_output.txt

Notes:
- The script defaults to using the MT5Tokenizer. If your checkpoint is T5, you may see a warning message. 
  This can generally be ignored unless switching tokenization behavior intentionally.
"""

import sys
import json
import numpy as np
import pandas as pd
import tqdm
from tabulate import tabulate
from transformers import MT5Tokenizer
from datasets import Dataset

def tokenize_dataset(input_path, tokenizer, max_input_length=512, max_output_length=128):
    input_ids_list = []
    attention_mask_list = []
    label_ids_list = []
    body_lens = []
    summary_lens = []
    unique_subreddits = set()
    num_posts = 0
    error_lines = 0

    with open(input_path, "r", encoding='utf-8') as infile:
        lines = infile.readlines()

        for idx, line in enumerate(tqdm.tqdm(lines, desc="Tokenizing Data")):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                body_text = data["content"]
                summary_text = data["summary"]

                input_enc = tokenizer(
                    body_text,
                    padding="max_length",
                    truncation=True,
                    max_length=max_input_length,
                    return_attention_mask=True,
                )
                output_enc = tokenizer(
                    summary_text,
                    padding="max_length",
                    truncation=True,
                    max_length=max_output_length,
                    return_attention_mask=False,
                )

                input_ids_list.append(input_enc["input_ids"])
                attention_mask_list.append(input_enc["attention_mask"])
                label_ids_list.append(output_enc["input_ids"])

                subreddit_id = data.get("subreddit_id")
                if subreddit_id:
                    unique_subreddits.add(subreddit_id)

                num_posts += 1
                body_lens.append(len([t for t in input_enc["input_ids"] if t != tokenizer.pad_token_id]))
                summary_lens.append(len([t for t in output_enc["input_ids"] if t != tokenizer.pad_token_id]))

            except json.JSONDecodeError:
                print(f"Error decoding line {idx}")
                error_lines += 1

    dataset = Dataset.from_dict({
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": label_ids_list,
    })

    return dataset, body_lens, summary_lens, num_posts, len(unique_subreddits), error_lines

def calculate_statistics(lengths):
    lengths = np.array(lengths)
    return {
        "Min": np.min(lengths),
        "Max": np.max(lengths),
        "Mean": np.mean(lengths),
        "Median": np.median(lengths),
        "Std. Deviation": np.std(lengths)
    }

def write_statistics(output_file, num_posts, num_subreddits, body_stats, summary_stats):
    df = pd.DataFrame([body_stats, summary_stats], index=["Body", "Summary"])
    with open(output_file, "w") as f:
        f.write("TL;DR Dataset Statistics\n\n")
        f.write(f"Total Posts: {num_posts}\n")
        f.write(f"Unique Subreddits: {num_subreddits}\n\n")
        f.write(tabulate(df, headers="keys", tablefmt="pretty"))

def main(input_path, output_dataset_path, output_stats_path, tokenizer_name="google/mt5-base"):
    tokenizer = MT5Tokenizer.from_pretrained(tokenizer_name)
    dataset, body_lens, summary_lens, num_posts, num_subreddits, error_lines = tokenize_dataset(input_path, tokenizer)

    # Save tokenized dataset
    dataset.save_to_disk(output_dataset_path)

    # Save statistics
    body_stats = calculate_statistics(body_lens)
    summary_stats = calculate_statistics(summary_lens)
    write_statistics(output_stats_path, num_posts, num_subreddits, body_stats, summary_stats)

    print(f"Finished! Dataset saved to {output_dataset_path}. {error_lines} decoding errors.")

if __name__ == "__main__":
    input_path = sys.argv[1]
    output_seq_path = sys.argv[2]
    output_stats_path = sys.argv[3]
    main(input_path, output_seq_path, output_stats_path)
