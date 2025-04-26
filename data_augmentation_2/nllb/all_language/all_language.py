#!/usr/bin/env python3
#Each entry is translated into all 5 languages

import json
import jsonlines
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm


MODEL_NAME = "facebook/nllb-200-distilled-600M"
SRC_LANG = "eng_Latn"

LANG_CODES = {
    "tl": "tgl_Latn",      # Tagalog
    "el": "ell_Grek",      # Greek
    "ro": "ron_Latn",      # Romanian
    "id": "ind_Latn",      # Indonesian
    "ru": "rus_Cyrl",      # Russian
}
LANG_NAMES = {
    "tl": "Tagalog",
    "el": "Greek",
    "ro": "Romanian",
    "id": "Indonesian",
    "ru": "Russian",
}

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        src_lang=SRC_LANG    
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

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

#Translate 1 input into all languages
def translate_all_lang(dataset, use_gpu):
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    tokenizer, model = load_model_and_tokenizer()
    model.to(device)

    out = []

    for entry in tqdm(dataset, desc="Translating entries", unit="entry"):
        raw = entry.get("input_text", "")

        for lang, tgt_lang_code in LANG_CODES.items():
            translated = translate_text(raw, tokenizer, model, tgt_lang_code)
            out.append({
                "input_text": translated,
                "summary_text": entry.get("summary_text", ""),
                "translated_to": LANG_NAMES[lang]
            })

    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Translate each JSON entry into all target languages using NLLB."
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Path to source JSON (list of {input_text, summary_text})."
    )
    parser.add_argument(
        "-o", "--output", default="translated_dataset.json",
        help="Path to write translated JSON."
    )
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for inference')
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    translated = translate_all_lang(data, args.use_gpu)

    with jsonlines.open(args.output, mode='w') as writer:
        writer.write_all(translated)

    print(f"Done — translated {len(translated)} entries → {args.output}")
