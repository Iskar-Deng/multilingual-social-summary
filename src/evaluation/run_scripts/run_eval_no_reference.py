"""
run_eval_no_reference.py

This script evaluates model predictions without using reference summaries.
It computes LaSE scores based on input texts and generated summaries.

Usage:
    python run_scripts/run_eval_no_reference.py <path_to_input_jsonl> --LaSE [--get_average] [--hide_individual_scores]

Arguments:
    json_path: Path to the JSONL file containing input and summary fields.
    --LaSE: Enable LaSE evaluation.
    --get_average: Compute and display average scores across all examples.
    --hide_individual_scores: Suppress printing individual scores.

Notes:
- The input JSONL file must contain "input_text" and "summary_text" fields.
- Currently only LaSE evaluation is supported.
"""

import json
import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "evaluation", "evaluation_scripts")))
from eval_LaSE import evaluate_LaSE



def main():
    parser = argparse.ArgumentParser(
        description="Evaluate predictions based on a JSONL file with 'input_text' and 'summary_text' fields."
    )

    parser.add_argument("json_path", help="Path to the JSONL file containing evaluation data.")
    parser.add_argument("--LaSE", action="store_true", help="Evaluate using the LaSE metric.")
    parser.add_argument("--get_average", action="store_true", help="Display average scores across all examples.")
    parser.add_argument("--hide_individual_scores", action="store_true", help="Suppress printing individual scores.")

    args = parser.parse_args()

    if not args.LaSE:
        print("No evaluation metric selected. Use --LaSE.")
        sys.exit(1)
    
    LaSE_score_types = ["LaSE"]  # Simplified to only LaSE; omitting ms, lc, lp components
    LaSE_scores = {score_type: [] for score_type in LaSE_score_types}
    
    num_lines = 0

    try:
        with open(args.json_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                num_lines += 1

                if "summary_text" in data and "input_text" in data:
                    reference = data["input_text"]
                    prediction = data["summary_text"]
                else:
                    print("'summary_text' or 'input_text' not found in the data.")
                    sys.exit(1)

                scores = evaluate_LaSE(prediction, reference, get_all_scores=True)
                for score_type, score in scores.items():
                    if score_type in LaSE_score_types:
                        LaSE_scores[score_type].append(float(score))
    except FileNotFoundError:
        print(f"File not found: {args.json_path}")
        sys.exit(1)

    if not args.hide_individual_scores:
        print("LaSE scores:")
        print(LaSE_scores)
        print()
    
    if args.get_average:
        avg_LaSE_scores = {score_type: sum(LaSE_scores[score_type]) / num_lines for score_type in LaSE_score_types}
        print(f"\nLaSE Score averages across all summarizations:\n{avg_LaSE_scores}")
        print()


if __name__ == "__main__":
    main()
