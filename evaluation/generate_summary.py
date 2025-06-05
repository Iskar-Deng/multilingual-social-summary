# evaluation/generate_summary.py
# Author: Nathalia Xu (Modified)

import os
import argparse
import json
import pandas as pd
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from peft import PeftModel
import torch
from tqdm import tqdm
from utils import DATA_PATH, RESULTS_PATH

def load_model(ckpt_path):
    base_model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
    model = PeftModel.from_pretrained(base_model, ckpt_path)
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


def generate_for_dataset(name, data, tokenizer, model, device, batch_size, output_path):
    print(f"Generating summaries for {name} ({len(data)} examples)...")
    results = []
    for i in tqdm(range(0, len(data), batch_size), desc=f"{name} Batches"):
        batch = data[i:i + batch_size]
        batch_summaries = generate_summary_batch(batch, tokenizer, model, device)
        for input_text, summary in zip(batch, batch_summaries):
            results.append({"input": input_text, "summary": summary})

    print(f"Saving {name} results to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["base", "noun", "sent", "full", "val"],
                        help="Dataset variant (also used to resolve checkpoint)")
    parser.add_argument("--step", required=True, type=int, help="Checkpoint step to load")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for generation")
    args = parser.parse_args()

    # === Resolve model checkpoint ===
    ckpt_dir = os.path.join("checkpoints", f"mt5_{args.variant}", f"checkpoint-{args.step}")
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_dir}")
    print(f"Using model from {ckpt_dir}")

    tokenizer, model, device = load_model(ckpt_dir)

    # === Output directory ===
    output_dir = os.path.join(RESULTS_PATH, args.variant)
    os.makedirs(output_dir, exist_ok=True)

    # === TL;DR data ===
    tldr_path = os.path.join(DATA_PATH, "tldr_split", "tldr_val.jsonl")
    print(f"Loading TL;DR from {tldr_path}...")
    with open(tldr_path, "r", encoding="utf-8") as f:
        tldr_data = [json.loads(line)["content"] for line in f]
    tldr_out_path = os.path.join(output_dir, f"{args.variant}_{args.step}_tldr.jsonl")
    generate_for_dataset("TL;DR", tldr_data, tokenizer, model, device, args.batch_size, tldr_out_path)

    # === CodeSwitch data ===
    cs_path = os.path.join(DATA_PATH, "codeswitch", "sliced_cs.csv")
    print(f"Loading CodeSwitch from {cs_path}...")
    df_cs = pd.read_csv(cs_path)
    cs_data = df_cs["Text"].astype(str).tolist()
    cs_out_path = os.path.join(output_dir, f"{args.variant}_{args.step}_cs.jsonl")
    generate_for_dataset("CodeSwitch", cs_data, tokenizer, model, device, args.batch_size, cs_out_path)

    print("All done.")


if __name__ == "__main__":
    main()
