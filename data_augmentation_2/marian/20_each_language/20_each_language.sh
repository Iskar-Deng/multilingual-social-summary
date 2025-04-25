#!/bin/bash
exec ~/miniconda3/envs/nllb/bin/python 20_each_language.py \
    --input    toy_data_tokenized.jsonl \
    --output   20_each_dataset.jsonl \
    --use_gpu
   