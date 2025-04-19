#!/bin/bash
# run_paraphrasing.sh

# Run your paraphrasing script with GPU
python paraphrasing.py \
  --input_file step1_output.jsonl \
  --src_tgt_model Helsinki-NLP/opus-mt-en-de \
  --tgt_src_model Helsinki-NLP/opus-mt-de-en \
  --output_file step2_output.jsonl \
  --use_gpu
