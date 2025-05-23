#!/usr/bin/env python3
#Translate 20% entries for each language.
#Each block is fully translated into a fixed target language:
#the first block into Tagalog, the second into Greek, the third into Romanian, the fourth into Indonesian, and the fifth into Russian.


import json
import jsonlines
import argparse
import torch
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

def count_lines(file_path):
    """Count total number of lines in a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        total_line = sum(1 for _ in f)
        return total_line 

MODEL_NAME = "facebook/nllb-200-distilled-600M"
SRC_LANG = "eng_Latn"

LANG_CODES = {
    "tl": "tgl_Latn",      
    "el": "ell_Grek",      
    "ro": "ron_Latn",      
    "id": "ind_Latn",      
    "ru": "rus_Cyrl",      
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
def translate_fixed_lang(total_line, dataset, use_gpu):
    """
    Translate exactly `per_lang` entries for each target language, in dataset order.
    The first `per_lang` entries → Tagalog, next `per_lang` → Greek, etc.
    """
    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    per_lang = int(total_line / 5)
    tokenizer, model = load_model_and_tokenizer()
    model.to(device)
    langs = list(LANG_CODES.keys())
    seq = []
    for lang in langs:
        seq.extend([lang] * per_lang)

    out = []
    for entry, lang in tqdm(zip(dataset, seq),
                            total=len(seq),
                            desc="Translating entries",
                            unit="entry"):
        tokens = entry.get("input_tokens")
        if tokens:
            raw = tokenizer.convert_tokens_to_string(tokens)
        else:
            raw = entry.get("input_text", "")
        tgt_lang_code = LANG_CODES[lang]
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

    total_line = count_lines(args.input)

    translated = translate_fixed_lang(total_line, data,args.use_gpu)

    with jsonlines.open(args.output, mode='w') as writer:
        writer.write_all(translated)

    print(f"Done — translated {len(translated)} entries → {args.output}")


    with jsonlines.open(args.output, mode='w') as writer:
        writer.write_all(translated)

    print(f"Done — translated {len(translated)} entries → {args.output}")
