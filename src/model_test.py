import sys
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

def load_model(model_name="google/mt5-base"):
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    model = MT5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

def summarize(text, tokenizer, model, max_input_length=512, max_output_length=64):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input_length)
    outputs = model.generate(**inputs, max_length=max_output_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    if len(sys.argv) < 2:
        print("Usage: python model_run.py '<your input text>'")
        sys.exit(1)

    input_text = sys.argv[1]

    print("=" * 60)
    print("Loading model...")
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
