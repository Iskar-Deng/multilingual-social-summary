"""
test_finetuned_model.py

This script loads a fine-tuned MT5 model checkpoint and generates a summary for a given input text.

Usage:
    python src/model_test/test_finetuned_model.py "<input_text>"

Arguments:
    input_text: Text to summarize (required)

Model configuration:
- Model checkpoint is loaded from ./checkpoints/toy_mt5
- Max input length: 512
- Max output length: 128

Example:
    python src/model_test/test_finetuned_model.py "Today is a beautiful day to learn AI."
"""

import sys
from transformers import MT5Tokenizer, MT5ForConditionalGeneration

MODEL_DIR = "./checkpoints/toy_mt5"
MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 128

def load_model(model_dir):
    tokenizer = MT5Tokenizer.from_pretrained(model_dir, local_files_only=True)
    model = MT5ForConditionalGeneration.from_pretrained(model_dir, local_files_only=True)
    return tokenizer, model

def summarize(text, tokenizer, model, max_input_length=512, max_output_length=128):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=max_input_length)
    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_output_length,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return summary

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_finetuned_model.py '<input_text>'")
        sys.exit(1)

    input_text = sys.argv[1]

    tokenizer, model = load_model(MODEL_DIR)

    print("=" * 60)
    print("Input Text:")
    print(input_text)

    print("\nGenerating Summary...\n")
    summary = summarize(input_text, tokenizer, model, MAX_INPUT_LENGTH, MAX_OUTPUT_LENGTH)

    print("Summary:")
    print(summary)
    print("=" * 60)

if __name__ == "__main__":
    main()
