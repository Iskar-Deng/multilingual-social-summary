#!/bin/bash

echo "Tokenizing base dataset..."
python preprocessing/tokenize_tldr.py --variant base --use_cache

echo "Tokenizing full translated dataset..."
python preprocessing/tokenize_tldr.py --variant full --use_cache

echo "Tokenizing sentence-level translated dataset..."
python preprocessing/tokenize_tldr.py --variant sent --use_cache

echo "Tokenizing noun-level translated dataset..."
python preprocessing/tokenize_tldr.py --variant noun --use_cache

echo "All tokenization completed."