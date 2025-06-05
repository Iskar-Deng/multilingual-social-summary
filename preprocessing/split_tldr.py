# preprocessing/split_tldr.py
# Author: Iskar Deng

import os
import argparse
import random
import json
from utils import DATA_PATH

def load_lines(jsonl_path, max_lines):
    lines = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            lines.append(line)
    return lines

def load_all_lines(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return f.readlines()

def split_lines(lines, train_size, val_size):
    return lines[:train_size], lines[train_size:train_size + val_size]

def extract_content_summary(line):
    data = json.loads(line)
    return json.dumps({
        "content": data["content"],
        "summary": data["summary"]
    }, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description="Split TL;DR JSONL file into train/val sets.")
    parser.add_argument("--train_size", type=int, default=100000)
    parser.add_argument("--val_size", type=int, default=1000)
    parser.add_argument("--ns", action="store_true", help="Disable shuffling")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    jsonl_path = os.path.join(DATA_PATH, "corpus-webis-tldr-17.json")
    output_dir = os.path.join(DATA_PATH, "tldr_split")
    os.makedirs(output_dir, exist_ok=True)

    total = args.train_size + args.val_size

    if args.ns:
        print(f"Reading first {total} lines without shuffling from: {jsonl_path}")
        lines = load_lines(jsonl_path, max_lines=total)
        if len(lines) < total:
            raise ValueError(f"File contains only {len(lines)} lines, but {total} requested.")
    else:
        print(f"Reading full file and shuffling (seed={args.seed}) from: {jsonl_path}")
        lines = load_all_lines(jsonl_path)
        if len(lines) < total:
            raise ValueError(f"File contains only {len(lines)} lines, but {total} requested.")
        random.seed(args.seed)
        random.shuffle(lines)

    train_lines, val_lines = split_lines(lines, args.train_size, args.val_size)

    train_path = os.path.join(output_dir, "tldr_train_base.jsonl")
    val_path = os.path.join(output_dir, "tldr_val.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for line in train_lines:
            f.write(extract_content_summary(line) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for line in val_lines:
            f.write(extract_content_summary(line) + "\n")

    print(f"Saved {args.train_size} lines to {train_path}")
    print(f"Saved {args.val_size} lines to {val_path}")

if __name__ == "__main__":
    main()
