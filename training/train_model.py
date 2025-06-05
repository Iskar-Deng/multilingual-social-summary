# training/train_model.py
# Author: Zoey Zhou & Nathalia Xu

import os
import torch
from datasets import load_from_disk
from transformers import (
    MT5Tokenizer,
    MT5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
from utils import TRAINING_ARGS, DATA_PATH, CHECKPOINT_PATH

class TqdmCallback(TrainerCallback):
    def __init__(self):
        self.pbar = None

    def on_train_begin(self, args, state, control, **kwargs):
        total_steps = state.max_steps
        self.pbar = tqdm(total=total_steps, desc="Training Progress", dynamic_ncols=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.pbar:
            self.pbar.update(1)

    def on_train_end(self, args, state, control, **kwargs):
        if self.pbar:
            self.pbar.close()

def main(variant):
    tokenized_path = os.path.join(DATA_PATH, "tokenized", f"tldr_{variant}_tokenized")
    checkpoint_path = os.path.join(CHECKPOINT_PATH, f"mt5_{variant}")

    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")
    dataset = load_from_disk(tokenized_path)
    print(f"Loaded {len(dataset)} examples from {tokenized_path}")

    base_model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    model.to("cpu")

    os.makedirs(os.path.join(checkpoint_path, "logs"), exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=checkpoint_path,
        overwrite_output_dir=True,
        per_device_train_batch_size=TRAINING_ARGS["batch_size"],
        gradient_accumulation_steps=TRAINING_ARGS["grad_accum_steps"],
        num_train_epochs=TRAINING_ARGS["num_epochs"],
        learning_rate=TRAINING_ARGS["lr"],
        warmup_steps=TRAINING_ARGS["warmup_steps"],
        save_steps=TRAINING_ARGS["save_steps"],
        logging_dir=os.path.join(checkpoint_path, "logs"),
        logging_steps=TRAINING_ARGS["log_steps"],
        # evaluation_strategy="no",
        predict_with_generate=True,
        fp16=False,
        bf16=torch.cuda.is_bf16_supported(),
        dataloader_num_workers=TRAINING_ARGS["num_workers"],
        load_best_model_at_end=False,
        save_strategy="steps",
        logging_first_step=True,
        resume_from_checkpoint=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        callbacks=[TqdmCallback()]
    )

    last_ckpt = None
    if os.path.isdir(checkpoint_path):
        ckpts = [os.path.join(checkpoint_path, d) for d in os.listdir(checkpoint_path) if d.startswith("checkpoint")]
        if ckpts:
            last_ckpt = max(ckpts, key=os.path.getmtime)
            print(f"Resuming from checkpoint: {last_ckpt}")

    print("Starting training...")
    trainer.train(resume_from_checkpoint=last_ckpt if last_ckpt else None)

    print("Saving final model...")
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    print(f"Model and tokenizer saved to {checkpoint_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["base", "noun", "sent", "full"],
                        help="Which dataset variant to train on")
    args = parser.parse_args()
    main(args.variant)