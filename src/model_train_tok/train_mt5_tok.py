import os
import argparse
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
from tqdm import tqdm

DEFAULT_TOKENIZED_PATH = "/gscratch/stf/mx727/multilingual-social-summary/data/splits/tokenized_train_clean"

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenized_path", default=DEFAULT_TOKENIZED_PATH, help="Tokenized dataset directory")
    parser.add_argument("--output_dir", required=True, help="Checkpoint and model output directory")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--fp16", action="store_true")  # this flag is ignored below
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")
    dataset = load_from_disk(args.tokenized_path)
    print(f"📊 Loaded {len(dataset)} examples from {args.tokenized_path}")

    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    logging_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(logging_dir, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=3e-5,
        warmup_steps=500,
        save_steps=2000,
        logging_dir=logging_dir,
        logging_steps=args.log_steps,
        evaluation_strategy="no",
        predict_with_generate=True,
        fp16=False,  # ✅ Force fp16 off regardless of args.fp16
        dataloader_num_workers=args.num_workers,
        load_best_model_at_end=False,
        save_strategy="steps",
        logging_first_step=True,
        resume_from_checkpoint=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        callbacks=[TqdmCallback()]  # ✅ Add progress bar callback
    )

    # Resume if any checkpoint exists
    last_ckpt = None
    if os.path.isdir(args.output_dir):
        ckpts = [
            os.path.join(args.output_dir, d)
            for d in os.listdir(args.output_dir)
            if d.startswith("checkpoint")
        ]
        if ckpts:
            last_ckpt = max(ckpts, key=os.path.getmtime)
            print(f"🔁 Resuming from checkpoint: {last_ckpt}")

    print("🚀 Starting training...")
    trainer.train(resume_from_checkpoint=last_ckpt if last_ckpt else None)

    print("💾 Saving final model...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"✅ Model and tokenizer saved to {args.output_dir}")

if __name__ == "__main__":
    main()
