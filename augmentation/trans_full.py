# augmentation/trans_full.py
# Author: Zoey Zhou

import json
import jsonlines
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import argparse
import os
from utils import DATA_PATH, NLLB_MODEL_NAME, NLLB_SRC_LANG, LANG_CODES, LANG_NAMES

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_NAME, src_lang=NLLB_SRC_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_NAME)
    return tokenizer, model

def translate_text(text, tokenizer, model, tgt_lang_code):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    bos_id = tokenizer.convert_tokens_to_ids(tgt_lang_code)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            forced_bos_token_id=bos_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

def translate_fixed_blocks(dataset, use_gpu):
    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    tokenizer, model = load_model_and_tokenizer()
    model.to(device)

    total = len(dataset)
    per_lang = total // 5
    lang_seq = []
    for lang in list(LANG_CODES.keys()):
        lang_seq.extend([lang] * per_lang)

    out = []
    for entry, lang in tqdm(zip(dataset, lang_seq), total=len(lang_seq), desc="Translating entries", unit="entry"):
        raw = entry.get("input_text", "")
        tgt_lang_code = LANG_CODES[lang]
        translated = translate_text(raw, tokenizer, model, tgt_lang_code)
        out.append({
            "content": translated,
            "summary": entry.get("summary_text", ""),
            "translated_to": LANG_NAMES[lang]
        })

    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for inference')
    args = parser.parse_args()

    input_path = os.path.join(DATA_PATH, "tldr_split", "tldr_train_base.jsonl")
    output_path = os.path.join(DATA_PATH, "tldr_split", "tldr_train_full.jsonl")

    with open(input_path, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]

    translated = translate_fixed_blocks(dataset, args.use_gpu)

    with jsonlines.open(output_path, mode='w') as writer:
        writer.write_all(translated)

    print(f"Saved full translated data to {output_path}")

if __name__ == "__main__":
    main()