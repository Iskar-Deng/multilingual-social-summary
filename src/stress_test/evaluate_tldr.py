import json
import argparse
from tqdm import tqdm
from transformers import MT5Tokenizer, MT5ForConditionalGeneration, logging as hf_logging
import bert_score

hf_logging.set_verbosity_error()

def evaluate_bert_score(predictions, references):
    # tqdm for BERTScore — iterate chunk by chunk if needed
    print("Calculating BERTScore...")
    P, R, F1 = bert_score.score(predictions, references, model_type="microsoft/deberta-xlarge-mnli", verbose=True)
    avg_f1 = F1.mean().item()
    return avg_f1

def main():
    parser = argparse.ArgumentParser(description="Evaluate MT5 on summarization task.")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    tokenizer = MT5Tokenizer.from_pretrained(args.model_dir)
    model = MT5ForConditionalGeneration.from_pretrained(args.model_dir)
    model.eval()

    predictions = []
    references = []

    # First, count total lines for tqdm
    with open(args.test_data, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    # Open file again for reading
    with open(args.test_data, "r", encoding="utf-8") as infile, tqdm(total=total_lines, desc="Generating summaries") as pbar:
        for line in infile:
            obj = json.loads(line)
            inputs = tokenizer(obj["input_text"], return_tensors="pt", truncation=True, padding=True)
            summary_ids = model.generate(**inputs, max_length=128)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            predictions.append(summary)
            references.append(obj["summary_text"])
            pbar.update(1)

    # Evaluate BERTScore with progress display
    avg_f1 = evaluate_bert_score(predictions, references)
    print(f"Average BERTScore F1: {avg_f1:.4f}")

    # Save predictions and score
    with open(args.output_path, "w", encoding="utf-8") as outfile:
        for pred, ref in zip(predictions, references):
            outfile.write(json.dumps({"prediction": pred, "reference": ref}) + "\n")
        outfile.write(json.dumps({"average_bert_f1": avg_f1}) + "\n")

if __name__ == "__main__":
    main()
