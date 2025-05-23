import os
import random
import argparse

# === Parse command-line arguments ===
parser = argparse.ArgumentParser(description="Generate test indices for aligned JSONL files.")
parser.add_argument("--tldr_path", required=True, help="Path to the tldr dataset")
parser.add_argument("--translated_sent_path", required=True, help="Path to the translated_sent file")
parser.add_argument("--translated_20_path", required=True, help="Path to the translated_20 file")
parser.add_argument("--translated_nouns_path", required=True, help="Path to the translated_nouns file")
parser.add_argument("--output_dir", required=True, help="Directory to save test_indices.txt")

args = parser.parse_args()

# === Set seed for reproducibility ===
random.seed(42)

# === Input files ===
input_files = {
    "tldr": args.tldr_path,
    "translated_sent": args.translated_sent_path,
    "translated_20": args.translated_20_path,
    "translated_nouns": args.translated_nouns_path
}

# === Create output directory if it doesn't exist ===
os.makedirs(args.output_dir, exist_ok=True)

# === Check file consistency ===
length = None
for key, path in input_files.items():
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if length is None:
            length = len(lines)
        elif len(lines) != length:
            raise ValueError(f"File {path} has inconsistent number of lines ({len(lines)} vs {length})")

print(f"✅ All files verified. Total lines: {length}")

# === Generate 10% test indices ===
indices = list(range(length))
random.shuffle(indices)
split_idx = int(0.1 * length)
test_indices = sorted(indices[:split_idx])

# === Save test indices ===
index_path = os.path.join(args.output_dir, "test_indices.txt")
with open(index_path, "w") as f:
    for idx in test_indices:
        f.write(f"{idx}\n")

print(f"📝 Test indices saved to: {index_path}")
