#!/bin/bash
exec ~/miniconda3/envs/nllb/bin/python all_language.py\
    --input    toy_data_tokenized.jsonl  \
    --output   all_translated_dataset.jsonl \
    --use_gpu 