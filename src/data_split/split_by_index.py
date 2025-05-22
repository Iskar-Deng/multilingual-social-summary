import os

# === Paths ===
data_path = "/gscratch/stf/mx727/multilingual-social-summary/data/corpus-webis-tldr-17.json"
index_path = "/gscratch/stf/mx727/multilingual-social-summary/data/splits/test_indices.txt"
output_dir = "/gscratch/stf/mx727/multilingual-social-summary/data/splits"
train_path = os.path.join(output_dir, "tldr_train.jsonl")
test_path = os.path.join(output_dir, "tldr_test.jsonl")

# === Load test indices into memory (very small) ===
with open(index_path, "r") as f:
    test_indices = set(int(line.strip()) for line in f)

# === Create output dir if needed ===
os.makedirs(output_dir, exist_ok=True)

# === Split the large file line-by-line ===
with open(data_path, "r", encoding="utf-8") as infile, \
     open(train_path, "w", encoding="utf-8") as train_out, \
     open(test_path, "w", encoding="utf-8") as test_out:

    for i, line in enumerate(infile):
        if i in test_indices:
            test_out.write(line)
        else:
            train_out.write(line)

print(f"Done. Output saved to:\n  - {train_path}\n  - {test_path}")
