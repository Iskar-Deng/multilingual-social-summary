import os
import json
import random

# Set seed for reproducibility
random.seed(42)

# === Input files (adjust as needed) ===
input_files = {
    "tldr": "/mmfs1/gscratch/stf/mx727/corpus-webis-tldr-17.json",
    "translated_sent": "/mmfs1/gscratch/stf/mx727/combined_translated_sentences.jsonl",
    "translated_20": "/mmfs1/gscratch/stf/mx727/translate_20_reordered.jsonl",
    "translated_nouns": "/mmfs1/gscratch/stf/mx727/translate_nouns_reordered.jsonl"
}

# === Output directory (/gscratch/stf/mx727/splits)
output_root = "/gscratch/stf/mx727"
output_dir = os.path.join(output_root, "splits")
os.makedirs(output_dir, exist_ok=True)

# === Load all files ===
data = {}
length = None

for key, path in input_files.items():
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        data[key] = lines
        if length is None:
            length = len(lines)
        elif len(lines) != length:
            raise ValueError(f"File {path} has inconsistent number of lines ({len(lines)} vs {length})")

print(f"✅ All files loaded. Total lines: {length}")

# === Generate consistent 10% test split
indices = list(range(length))
random.shuffle(indices)
split_idx = int(0.1 * length)
test_indices = sorted(indices[:split_idx])
train_indices = sorted(indices[split_idx:])

# === Save test indices
index_path = os.path.join(output_dir, "test_indices.txt")
with open(index_path, "w") as f:
    for idx in test_indices:
        f.write(f"{idx}\n")
print(f"📝 Test indices saved to: {index_path}")

# === Write aligned train/test files
for key in input_files:
    train_path = os.path.join(output_dir, f"{key}_train.jsonl")
    test_path = os.path.join(output_dir, f"{key}_test.jsonl")

    with open(train_path, "w", encoding="utf-8") as train_f, open(test_path, "w", encoding="utf-8") as test_f:
        for i, line in enumerate(data[key]):
            if i in test_indices:
                test_f.write(line)
            else:
                train_f.write(line)

    print(f"{key}: wrote {len(train_indices)} train and {len(test_indices)} test lines")

print(f"\n✅ All files saved to: {output_dir}")
