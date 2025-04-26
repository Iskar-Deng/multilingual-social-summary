import sys
from transformers import MT5Tokenizer, MT5ForConditionalGeneration

MODEL_DIR = "./checkpoints/toy_mt5"
MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 128

def load_model(model_dir):
    tokenizer = MT5Tokenizer.from_pretrained(model_dir)
    model = MT5ForConditionalGeneration.from_pretrained(model_dir)
    return tokenizer, model

def summarize(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_INPUT_LENGTH)
    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=MAX_OUTPUT_LENGTH,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return summary

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/model_test/test_finetuned_model.py 'Your input text here'")
        sys.exit(1)

    input_text = sys.argv[1]

    tokenizer, model = load_model(MODEL_DIR)

    print("\nInput Text:")
    print(input_text)
    print("\nGenerating Summary...\n")

    summary = summarize(input_text, tokenizer, model)

    print("Summary:")
    print(summary)
