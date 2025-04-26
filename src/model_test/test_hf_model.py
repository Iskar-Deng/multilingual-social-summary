"""
test_hf_model.py

This script loads the Huggingface MT5-base model and generates a summary for a given input text.

Usage:
    python src/model_test/test_hf_model.py "<input_text>"

Arguments:
    input_text: Text to summarize (required)

Example:
    python src/model_test/test_hf_model.py "The stock market saw a significant rally today..."
"""

import sys
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

MODEL_NAME = "google/mt5-base"

def load_model():
    tokenizer = MT5Tokenizer.from_pretrained(MODEL_NAME)
    model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    return tokenizer, model

def summarize(text, tokenizer, model, max_input_length=512, max_output_length=64):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input_length)
    outputs = model.generate(**inputs, max_length=max_output_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_hf_model.py '<input_text>'")
        sys.exit(1)

    input_text = sys.argv[1]

    print("=" * 60)
    print(f"Loading model: {MODEL_NAME}")
    tokenizer, model = load_model()
    print("Model loaded.\n")

    print("Input Text:")
    print(input_text)

    print("\nGenerating Summary...")
    summary = summarize(input_text, tokenizer, model)

    print("\nSummary:")
    print(summary)
    print("=" * 60)

if __name__ == "__main__":
    main()
