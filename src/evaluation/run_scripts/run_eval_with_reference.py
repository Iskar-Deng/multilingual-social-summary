"""
run_eval_with_reference.py

This script evaluates model predictions using different evaluation metrics.
It supports the following metrics:
1. **BERTScore**: A metric based on BERT to evaluate the similarity between predicted and reference summaries.
2. **ROUGE**: A set of metrics (ROUGE-1, ROUGE-2, ROUGE-L) used to evaluate the quality of summaries.

Usage:
    python run_scripts/run_eval_with_reference.py <path_to_input_jsonl> --bert [--get_average] [--hide_individual_scores]
    python run_scripts/run_eval_with_reference.py <path_to_input_jsonl> --rouge [--get_average] [--hide_individual_scores]

Arguments:
    json_path: Path to the JSON file containing input and summary fields.
    --bert: Enable BERTScore evaluation.
    --rouge: Enable ROUGE-1, ROUGE-2, ROUGE-L evaluation.
    --get_average: Compute and display average scores across all examples.
    --hide_individual_scores: Suppress printing individual scores for each evaluation.

Notes:
- The input JSONL file must contain "input_text" and "summary_text" fields.
- Currently only BERTScore and ROUGE evaluation are supported.
"""

import json
import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "evaluation", "evaluation_scripts")))
from eval_bert_score import evaluate_bert_score

import json
import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "evaluation", "evaluation_scripts")))
from eval_bert_score import evaluate_bert_score

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Evaluate predictions based on a JSON file with 'input_text' as reference and another key as prediction."
    )

    # Add command line arguments
    parser.add_argument("json_path", help="Path to the JSON file containing evaluation data.")
    parser.add_argument("--bert", action="store_true", help="Evaluate using BERTScore metric.")
    parser.add_argument("--get_average", action="store_true", help="Get average scores for each metric.")
    parser.add_argument("--hide_individual_scores", action="store_true", help="Hide individual scores for each evaluation.")

    # Parse arguments
    args = parser.parse_args()

    # Ensure at least one evaluation metric is selected
    if not args.bert:
        print("No evaluation metric selected. Use --bert.")
        sys.exit(1)

    # Initialize score dictionary
    score_types = ["f1"]  # Only compute F1 score (simplified version)
    bert_scores = {score_type: [] for score_type in score_types}
    
    num_lines = 0  # Count total number of samples

    # Read the evaluation data file
    with open(args.json_path, "r") as f:
        for line in f:
            data = json.loads(line)
            num_lines += 1

            # Get reference and prediction
            if "summary_text" in data and "reference_text" in data:
                reference = data["reference_text"]
                prediction = data["summary_text"]
            else:
                print("'summary_text' or 'reference_text' not found in the data.")
                sys.exit(1)

            # Compute BERTScore
            if args.bert:
                scores = evaluate_bert_score(prediction=prediction, reference=reference, get_all_scores=True)
                for score_type, score in zip(["precision", "recall", "f1"], scores):
                    if score_type in score_types:
                        bert_scores[score_type].append(score.item())

    # Print individual scores (if not hidden)
    if not args.hide_individual_scores:
        if args.bert:
            print("BERT scores:")
            print(bert_scores)
            print()
    
    # Compute and print average scores (if enabled)
    if args.get_average:
        if args.bert:
            avg_bert_scores = {score_type: sum(bert_scores[score_type]) / num_lines for score_type in score_types}
            print(f"\nBERT Score averages across all summarizations:\n{avg_bert_scores}")
            print()

if __name__ == "__main__":
    main()