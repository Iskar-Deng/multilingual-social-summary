import os
import argparse

# === Parse command-line arguments ===
parser = argparse.ArgumentParser(description="Split a JSONL file into train/test using an index file.")
parser.add_argument("--data_path", required=True, help="Path to the full dataset (JSONL file)")
parser.add_argument("--index_path", required=True, help="Path to the file with test indices")
parser.add_argument("--output_dir", required=True, help="Directory to save train/test splits")
args = parser.parse_args()

train_path = os.path.join(args.output_dir, "tldr_train.jsonl")
test_path = os.path.join(args.output_dir, "tldr_test.jsonl")

# === Load test indices into memory ===
with open(args.index_path, "r") as f:
    test_indices = set(int(line.strip()) for line in f)

# === Create output dir if needed ===
os.makedirs(args.output_dir, exist_ok=True)

# === Split the large file line-by-line ===
with open(args.data_path, "r", encoding="utf-8") as infile, \
     open(train_path, "w", encoding="utf-8") as train_out, \
     open(test_path, "w", encoding="utf-8") as test_out:

    for i, line in enumerate(infile):
        if i in test_indices:
            test_out.write(line)
        else:
            train_out.write(line)

print(f"Done. Output saved to:\n  - {train_path}\n  - {test_path}")
