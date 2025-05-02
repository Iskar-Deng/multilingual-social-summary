#!/bin/bash

# Usage:
# ./prepare_translation_jobs.sh filtered_jsonl_splits split_chunks output_chunks

INPUT_DIR="$1"
SPLIT_DIR="$2"
OUTPUT_DIR="$3"

mkdir -p "$SPLIT_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

echo "Splitting files from $INPUT_DIR into $SPLIT_DIR..."
python3 split_large_jsonl_files.py "$INPUT_DIR" "$SPLIT_DIR"

echo "Generating file_list.txt..."
> file_list.txt
find "$SPLIT_DIR" -name '*.jsonl' | while read input; do
    base=$(basename "$input" .jsonl)
    echo "$input $OUTPUT_DIR/${base}_translated.jsonl" >> file_list.txt
done

echo "✅ Done: file_list.txt generated with job entries."
