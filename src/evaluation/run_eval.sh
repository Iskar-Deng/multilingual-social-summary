#!/bin/bash

# JSON_PATH=$1

# python src/evaluation/run_eval.py "$DATASET_FILE"

# python src/evaluation/run_eval_with_reference.py "src/evaluation/sum_ref.jsonl" --bert --rouge --get_average

python src/evaluation/run_eval_no_reference.py "src/evaluation/source_sum.jsonl" --LaSE --get_average
# python src/evaluation/run_eval_no_reference.py "data_augmentation_2/marian/toy_data_tokenized.jsonl" --LaSE --get_average