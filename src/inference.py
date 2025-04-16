# src/inference.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return tokenizer, model

def generate_summary(model, tokenizer, text, max_length=128):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    output_ids = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    model_path = "checkpoints/mt5-finetuned"
    test_file = "data/test_inputs.json"
    output_file = "data/generated_outputs.json"

    tokenizer, model = load_model(model_path)

    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    outputs = []
    for text in test_data:
        summary = generate_summary(model, tokenizer, text)
        outputs.append({"input": text, "summary": summary})

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

    print(f"Inference complete. Output saved to {output_file}")
