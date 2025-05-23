#!/bin/bash
exec ~/miniconda3/envs/nllb/bin/python translate_sentences.py\
    --input    "$1" \
    --output   "$2" \
    --seed "$3"\
    --use_gpu "$4"
