#!/bin/bash
# run_paraphrasing.sh
python extractive_summarization.py \
--input_file toy_data_tokenized.jsonl \
--depth_ratio 0.5 \
--group_tokens \
--output_file step1_output.jsonl