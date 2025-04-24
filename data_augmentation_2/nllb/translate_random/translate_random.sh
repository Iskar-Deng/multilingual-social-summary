#!/bin/bash
python3 translate_random.py\
    --input    toy_data_tokenized.jsonl  \
    --output   translate_random.jsonl \
    --seed 123\
    --use_gpu 