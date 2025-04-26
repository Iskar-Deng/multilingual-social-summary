#!/bin/bash
exec ~/miniconda3/envs/nllb/bin/python all_language.py\
    --input    "$1" \
    --output   "$2"\
    --use_gpu  "$3"