from transformers import MT5ForConditionalGeneration, MT5Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import json
import os
import pandas as pd
from tqdm import tqdm

# === CONFIG ===
MODEL_NAME = "google/mt5-base"
DATA_PATH = "standard_data.jsonl"  # ← 改成你本地数据路径
OUTPUT_DIR = "./checkpoints/mt5_0425_2"
LOG_DIR = "./logs"

# 自动创建保存路径
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# === 数据加载 ===
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

# === 预处理函数 ===
def preprocess(example):
    model_inputs = tokenizer(
        example["input_text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["summary_text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# === 加载模型和 tokenizer ===
tokenizer = MT5Tokenizer.from_pretrained(MODEL_NAME)
model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# === 准备数据集 ===
dataset = load_dataset(DATA_PATH)

# 显示 tokenization 进度条
print("Tokenizing dataset...")
tokenized_dataset = dataset.map(
    preprocess,
    remove_columns=dataset.column_names,
    desc="Tokenizing"
)

# === 训练参数设置 ===
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,            # 为避免爆显存，设置小一些
    gradient_accumulation_steps=2,            # 相当于 batch size 8
    auto_find_batch_size=True,                # 如果出错可自动调整
    num_train_epochs=5,
    logging_dir=LOG_DIR,
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100,
    report_to="none",
    logging_first_step=True,
    evaluation_strategy="no",
    fp16=True,                                # 启用混合精度
    dataloader_num_workers=2
)

# === 初始化 Trainer ===
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model)
)

# === 启动训练 ===
trainer.train()

# === 保存模型和 tokenizer ===
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
