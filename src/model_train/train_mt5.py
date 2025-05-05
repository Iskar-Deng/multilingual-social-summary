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
DEFAULT_DATA_PATH = "data/full_data.jsonl"
DEFAULT_OUTPUT_DIR = "./checkpoints/mt5"
LOG_DIR = "./logs"

MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 128

# === Utility Functions ===

def load_dataset(path):
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
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    tokenizer = MT5Tokenizer.from_pretrained(MODEL_NAME)
    model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    dataset = load_dataset(args.data_path)
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess(x, tokenizer),
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.num_epochs,
        logging_dir=LOG_DIR,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.log_steps,
        report_to="none",
        logging_first_step=True,
        fp16=args.fp16,
        dataloader_num_workers=args.num_workers
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model)
    )

    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model and tokenizer saved to {args.output_dir}")

if __name__ == "__main__":
    main()
