# split_large_jsonl_files.py

import os
import jsonlines
import argparse

def split_jsonl_files(input_dir, output_dir, max_lines=5000):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if fname.endswith(".jsonl"):
            input_path = os.path.join(input_dir, fname)
            base_name = fname.replace(".jsonl", "")
            with jsonlines.open(input_path) as reader:
                buffer = []
                part = 0
                for i, obj in enumerate(reader):
                    buffer.append(obj)
                    if (i + 1) % max_lines == 0:
                        out_path = os.path.join(output_dir, f"{base_name}_part{part}.jsonl")
                        with jsonlines.open(out_path, 'w') as writer:
                            writer.write_all(buffer)
                        buffer = []
                        part += 1
                if buffer:
                    out_path = os.path.join(output_dir, f"{base_name}_part{part}.jsonl")
                    with jsonlines.open(out_path, 'w') as writer:
                        writer.write_all(buffer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Path to directory containing large .jsonl files")
    parser.add_argument("output_dir", help="Path to save split files")
    parser.add_argument("--max_lines", type=int, default=5000, help="Number of lines per split file")
    args = parser.parse_args()
    split_jsonl_files(args.input_dir, args.output_dir, args.max_lines)

