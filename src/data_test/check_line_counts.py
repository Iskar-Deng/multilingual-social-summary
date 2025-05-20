import os

# Relative or partial file names — update as needed
file_names = [
    "corpus-webis-tldr-17.json",
    "combined_translated_sentences.jsonl",
    "translate_20_reordered.jsonl",
    "translate_nouns_reordered.jsonl"
]

# === Resolve absolute paths ===
resolved_paths = []
for file_name in file_names:
    abs_path = os.path.abspath(os.path.expanduser(file_name))
    if os.path.exists(abs_path):
        resolved_paths.append(abs_path)
    else:
        print(f"[ERROR] File not found: {abs_path}")

# === Count lines in each file ===
def count_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

line_counts = {}
for path in resolved_paths:
    count = count_lines(path)
    line_counts[path] = count
    print(f"{path}: {count} lines")

# === Compare ===
unique_counts = set(line_counts.values())
if len(unique_counts) == 1:
    print("\n✅ All files have the same number of lines.")
else:
    print("\n❌ Mismatch in line counts:")
    for path, count in line_counts.items():
        print(f"{path}: {count}")
