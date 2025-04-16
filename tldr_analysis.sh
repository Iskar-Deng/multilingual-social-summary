#!/bin/bash
if [ $# -eq 0 ]; then
    echo "Usage: $0 <dataset_file> <output_file>"
    exit 1
fi

DATASET_FILE=$1
OUTPUT_FILE=$2

python ./scripts/tldr_analysis.py "$DATASET_FILE" "$OUTPUT_FILE"