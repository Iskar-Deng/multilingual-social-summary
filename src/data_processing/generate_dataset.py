import json
from transformers import MT5Tokenizer

input_path = "data/corpus-webis-tldr-17.json"
output_path = "toy_data.jsonl"
MAX_SAMPLES = 100

tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")

count = 0
with open(input_path, "r", encoding="utf-8") as infile, \
     open(output_path, "w", encoding="utf-8") as outfile:

    for line in infile:
        if line.strip():
            try:
                data = json.loads(line)
                body_text = data["content"]
                summary_text = data["summary"]

                body_tokens = tokenizer.tokenize(body_text)
                summary_tokens = tokenizer.tokenize(summary_text)

                json.dump({
                    "input_text": body_text,
                    "summary_text": summary_text,
                    "input_tokens": body_tokens,
                    "output_tokens": summary_tokens
                }, outfile)
                outfile.write("\n")

                count += 1
                if count >= MAX_SAMPLES:
                    break
            except json.JSONDecodeError:
                continue

print(f"Tokenized toy dataset saved to {output_path} ({count} samples)")
