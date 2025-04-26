"""
analyze_tldr.py

This script processes a TL;DR dataset in JSONL format and performs two tasks:
1. Tokenizes the 'content' and 'summary' fields using a specified tokenizer (default: google/mt5-base).
2. Calculates basic statistics (min, max, mean, median, std deviation) for the tokenized input and output lengths.

Outputs:
- A new JSONL file where each line contains a tokenized input-output pair.
- A text file summarizing dataset statistics.

Usage (command line):
    python src/data_processing/analyze_tldr.py <input_jsonl> <output_tokenized_jsonl> <output_stats_txt>

Arguments:
    input_jsonl: Path to the original TL;DR dataset (.jsonl)
    output_tokenized_jsonl: Path to save the tokenized input/output sequences (.jsonl)
    output_stats_txt: Path to save the dataset statistics (.txt)

Example:
    python src/data_processing/analyze_tldr.py data/corpus-webis-tldr-17.json tokenized_output.jsonl stats_output.txt

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

def load_and_tokenize(input_path, output_seq_path, tokenizer):
    body_lens = []
    summary_lens = []
    unique_subreddits = set()
    num_posts = 0
    error_lines = 0

    with open(input_path, "r", encoding='utf-8') as infile, open(output_seq_path, "w", encoding="utf-8") as outfile:
        lines = infile.readlines()

        for idx, line in enumerate(tqdm.tqdm(lines, desc="Processing Data")):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                body_text = data["content"]
                summary_text = data["summary"]

                body_tokens = tokenizer.tokenize(body_text)
                summary_tokens = tokenizer.tokenize(summary_text)

                json.dump({"input": body_tokens, "output": summary_tokens}, outfile)
                outfile.write("\n")

                subreddit_id = data.get("subreddit_id")
                if subreddit_id:
                    unique_subreddits.add(subreddit_id)

                num_posts += 1
                body_lens.append(len(body_tokens))
                summary_lens.append(len(summary_tokens))

            except json.JSONDecodeError:
                print(f"Error decoding line {idx}")
                error_lines += 1

    return body_lens, summary_lens, num_posts, unique_subreddits, error_lines

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

def main(input_path, output_seq_path, output_stats_path, tokenizer_name="google/mt5-base"):
    tokenizer = MT5Tokenizer.from_pretrained(tokenizer_name)
    body_lens, summary_lens, num_posts, unique_subreddits, error_lines = load_and_tokenize(
        input_path, output_seq_path, tokenizer
    )
    body_stats = calculate_statistics(body_lens)
    summary_stats = calculate_statistics(summary_lens)
    write_statistics(output_stats_path, num_posts, len(unique_subreddits), body_stats, summary_stats)
    print(f"Finished! {error_lines} decoding errors.")

if __name__ == "__main__":
    input_path = sys.argv[1]
    output_seq_path = sys.argv[2]
    output_stats_path = sys.argv[3]
    main(input_path, output_seq_path, output_stats_path)
