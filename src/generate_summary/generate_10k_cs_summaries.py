import os
import argparse
import json
import pandas as pd
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from peft import PeftModel
import torch
from tqdm import tqdm

def load_model(checkpoint_path):
    base_model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

def generate_summary_batch(texts, tokenizer, model, device, max_input_len=512, max_output_len=64):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_len
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_output_len,
            num_beams=4,
            early_stopping=True
        )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True, help="Path to the 10k input CSV")
    parser.add_argument("--checkpoint", required=True, help="Path to LoRA-adapted checkpoint")
    parser.add_argument("--output_path", required=True, help="Path to save the .jsonl output")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for generation")

    args = parser.parse_args()

    print("📄 Loading CSV...")
    df = pd.read_csv(args.csv_path)
    texts = df["Text"].astype(str).tolist()
    ids = df["id"].astype(str).tolist()

    # === Load previously written IDs (if output exists) ===
    existing_ids = set()
    if os.path.exists(args.output_path):
        print(f"🔍 Checking existing output file: {args.output_path}")
        with open(args.output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    existing_ids.add(obj["id"])
                except json.JSONDecodeError:
                    continue
        print(f"🔁 Found {len(existing_ids)} already processed examples.")

    # === Filter only the remaining inputs ===
    filtered = [(tid, txt) for tid, txt in zip(ids, texts) if tid not in existing_ids]
    if not filtered:
        print("✅ All examples already summarized. Exiting.")
        return

    ids, texts = zip(*filtered)

    print(f"⚙️ Loading model from {args.checkpoint}...")
    tokenizer, model, device = load_model(args.checkpoint)

    print(f"🧠 Generating summaries for {len(ids)} examples...")
    results = []
    for i in tqdm(range(0, len(texts), args.batch_size)):
        batch_texts = texts[i:i + args.batch_size]
        batch_ids = ids[i:i + args.batch_size]
        batch_summaries = generate_summary_batch(batch_texts, tokenizer, model, device)
        for id_val, input_text, output_text in zip(batch_ids, batch_texts, batch_summaries):
            results.append({
                "id": id_val,
                "input": input_text,
                "summary": output_text
            })

    print(f"💾 Appending {len(results)} new results to {args.output_path}")
    with open(args.output_path, "a", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("✅ Done.")

if __name__ == "__main__":
    main()
