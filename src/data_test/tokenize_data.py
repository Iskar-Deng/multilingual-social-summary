from transformers import MT5Tokenizer
import json

tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")

with open("/mmfs1/gscratch/stf/mx727/tldr_head_10000.jsonl", "r", encoding="utf-8") as fin, open("/mmfs1/gscratch/stf/mx727/tokenized_data_10000.jsonl", "w", encoding="utf-8") as fout:
    for line in fin:
        obj = json.loads(line)
        input_text = obj["content"]
        summary_text = obj["summary"]

        model_inputs = tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            padding="max_length"
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                summary_text,
                max_length=128,
                truncation=True,
                padding="max_length"
            )

        model_inputs["labels"] = labels["input_ids"]
        json.dump(model_inputs, fout)
        fout.write("\n")
