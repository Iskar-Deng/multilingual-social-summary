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
- The input JSONL file must contain "input" and "summary" fields.
- Currently only LaSE evaluation is supported.
"""

import json
import argparse
import sys
import os
from tqdm import tqdm

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

    LaSE_scores = []
    num_lines = 0

    try:
        with open(args.json_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Evaluating", unit="example"):
                data = json.loads(line)
                num_lines += 1

                if "summary" in data and "input" in data:
                    reference = data["input"]
                    prediction = data["summary"]
                else:
                    print("'summary' or 'input' not found in the data.")
                    sys.exit(1)

                print(f"Evaluating {num_lines}", file=sys.stderr)
                LaSE_score = evaluate_LaSE(prediction, reference, get_all_scores=False) 
                LaSE_scores.append(LaSE_score)
    except FileNotFoundError:
        print(f"File not found: {args.json_path}")
        sys.exit(1)

    filename = os.path.basename(args.json_path)
    name, _ = os.path.splitext(filename)
    out_filename = f"{name}_LaSE.txt"

    # Save the results to a file
    with open(out_filename, "w", encoding="utf-8") as out_file:
        if args.get_average:
            avg_LaSE_score = sum(LaSE_scores) / num_lines
            out_file.write(f"Average LaSE scores across all summarizations: {avg_LaSE_score}\n")
        if not args.hide_individual_scores:
            out_file.write(f"LaSE_scores: {LaSE_scores}")


if __name__ == "__main__":
    main()
