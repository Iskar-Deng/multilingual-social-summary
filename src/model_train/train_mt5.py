"""
train_mt5.py

This script fine-tunes a pre-trained MT5 model on a summarization dataset (TL;DR format).

Usage:
    python src/model_train/train_mt5.py --data_path <path_to_input_jsonl> --output_dir <path_to_save_checkpoint>

Arguments:
    --data_path: Path to the input dataset (default: data/standard_data.jsonl)
    --output_dir: Directory to save the fine-tuned model and tokenizer (default: checkpoints/mt5_0425_2)

Notes:
- Input texts are truncated to 512 tokens, summaries to 128 tokens.
- Uses Huggingface Seq2SeqTrainer with mixed-precision (fp16) training.

Example:
    python src/model_train/train_mt5.py --data_path data/standard_data.jsonl --output_dir checkpoints/mt5_finetuned
"""

import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import (
    MT5Tokenizer, 
    MT5ForConditionalGeneration, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments, 
    DataCollatorForSeq2Seq
)
from datasets import Dataset

# === Constants ===
MODEL_NAME = "google/mt5-base"
DEFAULT_DATA_PATH = "data/standard_data.jsonl"
DEFAULT_OUTPUT_DIR = "./checkpoints/mt5_0425_2"
LOG_DIR = "./logs"

MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 128

# === Utility Functions ===

def load_dataset(path):
    """Load dataset from a JSONL file into Huggingface Dataset format."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading data"):
            obj = json.loads(line)
            data.append({
                "input_text": obj["input_text"],
                "summary_text": obj["summary_text"]
            })
    return Dataset.from_pandas(pd.DataFrame(data))

def preprocess(example, tokenizer):
    """Tokenize input and output texts with truncation and padding."""
    model_inputs = tokenizer(
        example["input_text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_INPUT_LENGTH
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["summary_text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_OUTPUT_LENGTH
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    parser = argparse.ArgumentParser(description="Fine-tune MT5 on summarization task.")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH, help="Path to input JSONL data file.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save checkpoints.")
    args = parser.parse_args()

    # Create necessary directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Load model and tokenizer
    tokenizer = MT5Tokenizer.from_pretrained(MODEL_NAME)
    model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    # Load and preprocess dataset
    dataset = load_dataset(args.data_path)
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess(x, tokenizer),
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        auto_find_batch_size=True,
        num_train_epochs=5,
        logging_dir=LOG_DIR,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        report_to="none",
        logging_first_step=True,
        fp16=True,
        dataloader_num_workers=2
    )

    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model)
    )

    # Start training
    trainer.train()

    # Save model and tokenizer
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model and tokenizer saved to {args.output_dir}")

if __name__ == "__main__":
    main()
