import json
import argparse
import math
from tqdm import tqdm
import torch
from transformers import MT5Tokenizer, MT5ForConditionalGeneration, logging as hf_logging
from LaSE import LaSEScorer

hf_logging.set_verbosity_error()
lase_scorer = LaSEScorer()

def clean_text(text):
    if text is None:
        return ""
    if isinstance(text, (list, dict)):
        return json.dumps(text)
    if isinstance(text, float):
        if math.isnan(text) or math.isinf(text):
            return ""
        return str(text)
    return str(text).strip()

def evaluate_lase(predictions, references, target_lang='en'):
    cleaned_pairs = []
    for r, p in zip(references, predictions):
        clean_r = clean_text(r)
        clean_p = clean_text(p)
        if clean_r and clean_p:
            cleaned_pairs.append((clean_r, clean_p))

    if not cleaned_pairs:
        raise ValueError("No valid (non-empty) reference-prediction pairs found!")

    cleaned_refs, cleaned_preds = zip(*cleaned_pairs)

    scores_list = []
    print("Calculating LaSE scores...")
    for ref, pred in tqdm(zip(cleaned_refs, cleaned_preds), total=len(cleaned_refs), desc="Computing LaSE"):
        try:
            score = lase_scorer.score([ref], [pred], target_lang=target_lang)
            scores_list.append(float(score.LaSE))
        except Exception as e:
            print(f"Warning: Skipped a pair due to error → {e}")
            scores_list.append(0.0)

    if scores_list:
        avg_lase = sum(scores_list) / len(scores_list)
    else:
        avg_lase = 0.0
    return avg_lase, scores_list

def main():
    parser = argparse.ArgumentParser(description="Evaluate MT5 on CodeSwitch task using LaSE only.")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--target_lang", type=str, default="en")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = MT5Tokenizer.from_pretrained(args.model_dir)
    model = MT5ForConditionalGeneration.from_pretrained(args.model_dir).to(device)
    model.eval()

    predictions = []
    input_texts = []

    # Count total lines
    with open(args.test_data, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    # Generate predictions
    with open(args.test_data, "r", encoding="utf-8") as infile, tqdm(total=total_lines, desc="Generating summaries") as pbar:
        for line in infile:
            obj = json.loads(line)
            input_text = obj.get("input_text") or obj.get("Text", "")

            if not input_text.strip():
                continue

            try:
                inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(device)
                summary_ids = model.generate(**inputs, max_length=128)
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            except Exception as e:
                print(f"Warning: Skipped generation due to error → {e}")
                continue

            input_texts.append(input_text)
            predictions.append(summary)
            pbar.update(1)

    if not predictions:
        print("Error: No predictions generated. Exiting.")
        return

    # Evaluate LaSE with progress
    avg_lase, all_lase_scores = evaluate_lase(predictions, input_texts, target_lang=args.target_lang)
    print(f"Average LaSE Score: {avg_lase:.4f}")

    # Save predictions and LaSE scores
    with open(args.output_path, "w", encoding="utf-8") as outfile:
        for inp, pred, score in zip(input_texts, predictions, all_lase_scores):
            outfile.write(json.dumps({
                "input_text": inp,
                "prediction": pred,
                "lase_score": score
            }, ensure_ascii=False) + "\n")
        outfile.write(json.dumps({
            "average_lase": avg_lase
        }, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
