import os
import argparse
from datasets import load_dataset
from transformers import MT5Tokenizer

# Force HuggingFace to use quota-safe directories
os.environ["HF_DATASETS_CACHE"] = "/gscratch/stf/mx727/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/gscratch/stf/mx727/hf_cache"
os.environ["HF_MODULES_CACHE"] = "/gscratch/stf/mx727/hf_cache"
os.environ["XDG_CACHE_HOME"] = "/gscratch/stf/mx727/hf_cache"

DEFAULT_OUTPUT_PATH = "/gscratch/stf/mx727/tokenized_train.arrow"

def tokenize_and_save(data_path, output_path):
    print("🔄 Loading raw dataset...")
    raw_dataset = load_dataset("json", data_files=data_path, split="train")

    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")

    def tokenize_fn(batch):
        inputs = tokenizer(
            batch["input_text"],
            max_length=512,
            truncation=True,
            padding="max_length"
        )
        labels = tokenizer(
            batch["summary_text"],
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

    print("🧠 Tokenizing...")
    tokenized = raw_dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=4,
        remove_columns=raw_dataset.column_names  # ✅ Drop all original fields
    )

    os.makedirs(output_path, exist_ok=True)
    print(f"💾 Saving to {output_path}")
    tokenized.save_to_disk(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Raw JSONL file")
    parser.add_argument("--output_path", default=DEFAULT_OUTPUT_PATH, help="Where to save tokenized dataset")
    args = parser.parse_args()

    tokenize_and_save(args.data_path, args.output_path)
