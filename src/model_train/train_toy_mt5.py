from transformers import MT5ForConditionalGeneration, MT5Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import json
from tqdm import tqdm
import pandas as pd

# Load toy data
def load_toy_dataset(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            data.append({"input_text": obj["input_text"], "summary_text": obj["summary_text"]})
    return Dataset.from_pandas(pd.DataFrame(data))

# Tokenize
def preprocess(example):
    model_inputs = tokenizer(example["input_text"], truncation=True, padding="max_length", max_length=512)
    labels = tokenizer(example["summary_text"], truncation=True, padding="max_length", max_length=128)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# === CONFIG ===
MODEL_NAME = "google/mt5-base"
DATA_PATH = "toy_data_tokenized.jsonl"
OUTPUT_DIR = "./checkpoints/toy_mt5"

# Load model and tokenizer
tokenizer = MT5Tokenizer.from_pretrained(MODEL_NAME)
model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# Prepare dataset
dataset = load_toy_dataset(DATA_PATH)
tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# Training args
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    num_train_epochs=5,
    logging_dir="./logs",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    evaluation_strategy="no",
    fp16=False
)

# Initialize trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model)
)

# Train
trainer.train()

# Save model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
