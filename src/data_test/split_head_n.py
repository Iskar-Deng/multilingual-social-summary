# split_head_jsonl.py

import argparse
import os
import json

def extract_head(input_path, output_path, num_lines):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    count = 0

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            if count >= num_lines:
                break
            outfile.write(line)
            count += 1

    print(f"✅ Wrote first {count} lines to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract first N lines from a JSONL file.")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSONL file")
    parser.add_argument("--num_lines", type=int, default=5000, help="Number of lines to extract")

    args = parser.parse_args()
    extract_head(args.input, args.output, args.num_lines)
