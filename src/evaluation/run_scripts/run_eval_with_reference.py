"""
run_eval_with_reference.py

This script evaluates model predictions using different evaluation metrics.
It supports the following metrics:
1. **BERTScore**: A metric based on BERT to evaluate the similarity between predicted and reference summaries.

Usage:
    python run_scripts/run_eval_with_reference.py <path_to_prediction_jsonl> <path_to_reference_jsonl> --bert [--get_average] [--hide_individual_scores]

Arguments:
    json_path: Path to the JSON file containing input and summary fields.
    --bert: Enable BERTScore evaluation.
    --get_average: Compute and display average scores across all examples.
    --hide_individual_scores: Suppress printing individual scores for each evaluation.

Notes:
- The input JSONL file must contain "input_text" and "summary_text" fields.
- Currently only BERTScore evaluation is supported.
"""

import json
import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "evaluation", "evaluation_scripts")))
from eval_bert_score import evaluate_bert_score
from tqdm import tqdm

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Evaluate predictions based on a JSON file with 'input_text' as reference and another key as prediction."
    )

    # Add command line arguments
    parser.add_argument("prediction_path", help="Path to the JSON file containing prediction data.")
    parser.add_argument("reference_path", help="Path to the JSON file containing reference data.")
    parser.add_argument("--bert", action="store_true", help="Evaluate using BERTScore metric.")
    parser.add_argument("--get_average", action="store_true", help="Get average scores for each metric.")
    parser.add_argument("--hide_individual_scores", action="store_true", help="Hide individual scores for each evaluation.")

    # Parse arguments
    args = parser.parse_args()

    # Ensure at least one evaluation metric is selected
    if not args.bert:
        print("No evaluation metric selected. Use --bert.")
        sys.exit(1)

    # # Initialize score dictionary
    # score_types = ["f1"]  # Only compute F1 score (simplified version)
    # bert_scores = {score_type: [] for score_type in score_types}
    
    # num_lines = 0  # Count total number of samples

    # # Read the evaluation data file
    # with open(args.json_path, "r") as f:
    #     for line in tqdm(f, desc="Evaluating", unit="example"):
    #         data = json.loads(line)
    #         num_lines += 1

    #         # Get reference and prediction
    #         if "summary_text" in data and "reference_text" in data:
    #             reference = data["reference_text"]
    #             prediction = data["summary_text"]
    #         else:
    #             print("'summary_text' or 'reference_text' not found in the data.")
    #             sys.exit(1)

    #         # Compute BERTScore
    #         if args.bert:
    #             scores = evaluate_bert_score(prediction=prediction, reference=reference, get_all_scores=True)
    #             for score_type, score in zip(["precision", "recall", "f1"], scores):
    #                 if score_type in score_types:
    #                     bert_scores[score_type].append(score.item())

    # # Print individual scores (if not hidden)
    # if not args.hide_individual_scores:
    #     if args.bert:
    #         print("BERT scores:")
    #         print(bert_scores)
    #         print()
    
    num_lines = 0  # Count total number of samples
    predictions = []
    references = []

    # Read the prediction data file
    with open(args.prediction_path, "r") as f:
        for line in f:
            data = json.loads(line)
            num_lines += 1
            predictions.append(data["summary"])

    # Read the reference data file
    with open(args.reference_path, "r") as f:
        for line in f:
            data = json.loads(line)
            references.append(data["summary"])

    # Compute BERTScore for all predictions against references
    if args.bert:
        bert_scores = []
        num_per_iteration = 10
        for i in tqdm(range(num_lines // num_per_iteration), desc="Evaluating BERTScore", unit="batch"):
            start_index = i * num_per_iteration
            end_index = start_index + num_per_iteration
            if end_index > num_lines:
                end_index = num_lines
            bert_scores.extend(
                evaluate_bert_score(predictions=predictions[start_index:end_index], references=references[start_index:end_index], get_all_scores=False).tolist()
            )
        
    # Compute and print average scores (if enabled)
    if args.get_average:
        if args.bert:
            avg_bert_score = sum(bert_scores) / num_lines

    filename = os.path.basename(args.prediction_path)
    name, _ = os.path.splitext(filename)
    out_filename = f"{name}_bertscore.txt"
    # Save BERT scores to a file
    with open(out_filename, "w") as out_file:
        if args.get_average:
            out_file.write(f"Average BERTScore: {avg_bert_score}\n")
        out_file.write(f"BERTScore: {bert_scores}\n")

if __name__ == "__main__":
    main()