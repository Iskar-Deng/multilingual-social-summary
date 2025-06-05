# evaluation/run_scripts/run_eval_no_reference.py
# Author: Jordan Jin
import json
import argparse
import sys
import os
from tqdm import tqdm
from utils import RESULTS_PATH

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evaluation", "evaluation_scripts")))
from eval_LaSE import evaluate_LaSE

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate predictions using LaSE metric."
    )

    parser.add_argument("--variant", required=True, choices=["base", "noun", "sent", "full", "val"], help="Dataset variant")
    parser.add_argument("--step", required=True, type=int, help="Checkpoint step used for generation")

    args = parser.parse_args()

    pred_file = os.path.join(RESULTS_PATH, args.variant, f"{args.variant}_{args.step}_cs.jsonl")
    if not os.path.exists(pred_file):
        print(f"File not found: {pred_file}")
        sys.exit(1)

    LaSE_scores = []
    num_lines = 0

    with open(pred_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Evaluating", unit="example"):
            data = json.loads(line)
            num_lines += 1

            if "summary" in data and "input" in data:
                reference = data["input"]
                prediction = data["summary"]
            else:
                print("'summary' or 'input' not found in the data.")
                sys.exit(1)

            LaSE_score = evaluate_LaSE(prediction, reference, get_all_scores=False)
            LaSE_scores.append(LaSE_score)

    avg_LaSE_score = sum(LaSE_scores) / num_lines

    # Save the results to a JSONL file
    out_dir = os.path.join(RESULTS_PATH, "scores")
    os.makedirs(out_dir, exist_ok=True)
    out_filename = os.path.join(out_dir, f"{args.variant}_{args.step}_cs_LaSE.jsonl")

    with open(out_filename, "w", encoding="utf-8") as out_file:
        json.dump({
            "variant": args.variant,
            "step": args.step,
            "data": "CodeSwitch",
            "average": avg_LaSE_score,
            "scores": LaSE_scores
        }, out_file, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
