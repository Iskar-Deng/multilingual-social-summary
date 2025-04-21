#!/bin/bash

# JSON_PATH=$1

# python src/evaluation/run_eval.py "$DATASET_FILE"

python src/evaluation/run_eval.py "src/evaluation/run_eval_test.jsonl" --bert --rouge --get_average

python src/evaluation/run_eval.py "data_augmentation/step2_output.jsonl" --bert --rouge --get_average --hide_individual_scores