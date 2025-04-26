"""
generate_dataset.py

This script tokenizes a TL;DR dataset and generates a small toy dataset 
limited to a specified number of samples.

Outputs a JSONL file where each line contains:
- input_text: original body text
- summary_text: original summary text
- input_tokens: tokenized input (subword tokens)
- output_tokens: tokenized output (subword tokens)

Usage (command line):
    python src/data_processing/generate_dataset.py <input_jsonl> <output_jsonl> <max_samples> [tokenizer_name]

Arguments:
    input_jsonl: Path to the original dataset (.jsonl)
    output_jsonl: Path to save the tokenized toy dataset (.jsonl)
    max_samples: Maximum number of samples to include
    tokenizer_name: (Optional) Huggingface tokenizer name (default: google/mt5-base)

Example:
    python src/data_processing/generate_dataset.py data/corpus-webis-tldr-17.json data/toy_data.jsonl 100
"""

import sys
import json
from transformers import MT5Tokenizer

def generate_toy_dataset(input_path, output_path, max_samples=100, tokenizer_name="google/mt5-base"):
    tokenizer = MT5Tokenizer.from_pretrained(tokenizer_name)

    count = 0
    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                body_text = data["content"]
                summary_text = data["summary"]

                body_tokens = tokenizer.tokenize(body_text)
                summary_tokens = tokenizer.tokenize(summary_text)

                json.dump({
                    "input_text": body_text,
                    "summary_text": summary_text,
                    "input_tokens": body_tokens,
                    "output_tokens": summary_tokens
                }, outfile)
                outfile.write("\n")

                count += 1
                if count >= max_samples:
                    break
            except json.JSONDecodeError:
                print(f"Warning: Skipped a corrupted line.")
                continue

    print(f"Tokenized toy dataset saved to {output_path} ({count} samples)")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python generate_dataset.py <input_jsonl> <output_jsonl> <max_samples> [tokenizer_name]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    max_samples = int(sys.argv[3])
    tokenizer_name = sys.argv[4] if len(sys.argv) > 4 else "google/mt5-base"

    generate_toy_dataset(input_path, output_path, max_samples, tokenizer_name)
