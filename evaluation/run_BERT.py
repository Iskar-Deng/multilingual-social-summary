# evaluation/run_BERT.py
# Author: Jordan Jin

import json
import argparse
import sys
import os
from tqdm import tqdm
from utils import RESULTS_PATH, DATA_PATH

# Import BERTScore evaluation script from local directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evaluation", "evaluation_scripts")))
from eval_bert_score import evaluate_bert_score

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate predictions using BERTScore metric."
    )
    parser.add_argument("--variant", required=True, choices=["base", "noun", "sent", "full", "val"], help="Dataset variant")
    parser.add_argument("--step", required=True, type=int, help="Checkpoint step used for generation")
    args = parser.parse_args()

    # Construct paths to prediction and reference files
    pred_file = os.path.join(RESULTS_PATH, args.variant, f"{args.variant}_{args.step}_tldr.jsonl")
    ref_file = os.path.join(DATA_PATH, "tldr_split", "tldr_val.jsonl")

    # Check file existence
    if not os.path.exists(pred_file):
        print(f"Prediction file not found: {pred_file}")
        sys.exit(1)
    if not os.path.exists(ref_file):
        print(f"Reference file not found: {ref_file}")
        sys.exit(1)

    # Load predicted summaries and reference summaries
    predictions, references = [], []
    with open(pred_file, "r") as f:
        for line in f:
            predictions.append(json.loads(line)["summary"])

    with open(ref_file, "r") as f:
        for line in f:
            references.append(json.loads(line)["summary"])

    if len(predictions) != len(references):
        print("Mismatch between number of predictions and references")
        sys.exit(1)

    # Compute BERTScore for batches
    bert_scores = []
    batch_size = 10
    for i in tqdm(range(0, len(predictions), batch_size), desc="Evaluating BERTScore", unit="batch"):
        batch_preds = predictions[i:i + batch_size]
        batch_refs = references[i:i + batch_size]
        scores = evaluate_bert_score(predictions=batch_preds, references=batch_refs, get_all_scores=False)
        bert_scores.extend(scores.tolist())

    avg_score = sum(bert_scores) / len(bert_scores)

    # Save results to file
    out_dir = os.path.join(RESULTS_PATH, "scores")
    os.makedirs(out_dir, exist_ok=True)
    out_filename = os.path.join(out_dir, f"{args.variant}_{args.step}_tldr_bertscore.jsonl")

    with open(out_filename, "w", encoding="utf-8") as out_file:
        json.dump({
            "variant": args.variant,
            "step": args.step,
            "data": "tldr",
            "average": avg_score,
            "scores": bert_scores
        }, out_file, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
