import os
import glob

input_dir = "split_chunks"
output_dir = "output_chunks"
output_file = "file_list.txt"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Open file_list.txt for writing
with open(output_file, "w", encoding="utf-8") as f_out:
    for input_path in sorted(glob.glob(os.path.join(input_dir, "*.jsonl"))):
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{base}_translated.jsonl")
        f_out.write(f"{input_path} {output_path}\n")

print(f"Generated {output_file} with input/output file pairs.")
