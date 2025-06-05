# preprocessing/tokenize_tldr.py
# Author: Nathalia Xu

import os
import argparse
from datasets import load_dataset
from transformers import MT5Tokenizer
from utils import DATA_PATH, HF_CACHE_PATH
from tqdm import tqdm
import json

def maybe_set_hf_cache(use_cache):
    if use_cache and HF_CACHE_PATH:
        os.environ["HF_DATASETS_CACHE"] = HF_CACHE_PATH
        os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_PATH
        os.environ["HF_MODULES_CACHE"] = HF_CACHE_PATH
        os.environ["XDG_CACHE_HOME"] = HF_CACHE_PATH

def tokenize_and_save(data_path, output_path):
    print(f"Loading dataset from {data_path}")
    raw_dataset = load_dataset("json", data_files=data_path, split="train")

    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")

    def tokenize_fn(batch):
        inputs = tokenizer(
            batch["content"],
            max_length=512,
            truncation=True,
            padding="max_length"
        )
        labels = tokenizer(
            batch["summary"],
            max_length=64,
            truncation=True,
            padding="max_length"
        )
        labels["input_ids"] = [
            [(t if t != tokenizer.pad_token_id else -100) for t in label]
            for label in labels["input_ids"]
        ]
        inputs["labels"] = labels["input_ids"]
        return inputs

    print("Tokenizing with mT5...")
    tokenized = raw_dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=4,
        remove_columns=["content", "summary"],
        desc="Tokenizing"
    )

    os.makedirs(output_path, exist_ok=True)
    print(f"Saving tokenized dataset to {output_path}")
    tokenized.save_to_disk(output_path)

def main():
    parser = argparse.ArgumentParser(description="Tokenize TL;DR dataset (base, augmented, or val).")
    parser.add_argument("--variant", choices=["base", "sent", "noun", "full", "val"], required=True,
                        help="Dataset variant to tokenize (e.g., 'base', 'noun', etc.)")
    parser.add_argument("--use_cache", action="store_true", help="Use HuggingFace cache if defined in utils.py")
    args = parser.parse_args()

    maybe_set_hf_cache(args.use_cache)

    if args.variant == "val":
        input_path = os.path.join(DATA_PATH, "tldr_split", "tldr_val.jsonl")
    else:
        input_path = os.path.join(DATA_PATH, "tldr_split", f"tldr_train_{args.variant}.jsonl")

    output_path = os.path.join(DATA_PATH, "tokenized", f"tldr_{args.variant}_tokenized")
    tokenize_and_save(input_path, output_path)

if __name__ == "__main__":
    main()
