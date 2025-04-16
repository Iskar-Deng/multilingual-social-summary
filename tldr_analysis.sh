#!/bin/bash
if [ $# -eq 0 ]; then
    echo "Usage: $0 <dataset_file> <statistics_file> <tokenized_sequences_file>"
    exit 1
fi

DATASET_FILE=$1
OUTPUT_FILE=$2
SEQUENCES_FILE=$3

python ./scripts/tldr_analysis.py "$DATASET_FILE" "$OUTPUT_FILE" "$SEQUENCES_FILE"