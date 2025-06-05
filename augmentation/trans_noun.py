# augmentation/trans_noun.py
# Author: Zoey Zhou

import os
import json
import jsonlines
import random
import argparse
import torch
import spacy
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils import DATA_PATH, NLLB_MODEL_NAME, NLLB_SRC_LANG, LANG_CODES, LANG_NAMES

INPUT_PATH = os.path.join(DATA_PATH, "tldr_split", "tldr_train_base.jsonl")
OUTPUT_PATH = os.path.join(DATA_PATH, "tldr_split", "tldr_train_noun.jsonl")

# === Load spaCy model for POS tagging ===
nlp = spacy.load("en_core_web_sm")

# Load NLLB model and tokenizer
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_NAME, src_lang=NLLB_SRC_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_NAME)
    return tokenizer, model

# Translate a single word or phrase into the target language
def translate_text(text, tokenizer, model, tgt_lang_code):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    bos_id = tokenizer.convert_tokens_to_ids(tgt_lang_code)

    with torch.no_grad():
        out = model.generate(**inputs, forced_bos_token_id=bos_id)

    return tokenizer.decode(out[0], skip_special_tokens=True)

# Translate a random subset of NOUNs in the input text to a random target language
def translate_random_lang(dataset, seed, use_gpu):
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    tokenizer, model = load_model_and_tokenizer()
    model.to(device)
    random.seed(seed)
    out = []

    for entry in tqdm(dataset, desc="Translating entries", unit="entry"):
        raw = entry.get("input_text", "")
        doc = nlp(raw)
        words = [token.text for token in doc]

        # Identify NOUNs
        noun_indices = [i for i, token in enumerate(doc) if token.pos_ == "NOUN"]
        if not noun_indices:
            out.append(entry)
            continue

        # Randomly sample nouns to translate
        num_to_translate = random.randint(1, max(1, len(noun_indices) // 2))
        selected_noun_indices = random.sample(noun_indices, num_to_translate)
        lang = random.choice(list(LANG_CODES.keys()))
        tgt_lang_code = LANG_CODES[lang]

        translated_from_to = []
        for idx in selected_noun_indices:
            original_word = words[idx]
            translated_word = translate_text(original_word, tokenizer, model, tgt_lang_code)
            words[idx] = translated_word
            translated_from_to.append({"from": original_word, "to": translated_word})

        mixed_input = " ".join(words)
        out.append({
            "input_text": mixed_input,
            "summary_text": entry.get("summary_text", ""),
            "translated_from_to": translated_from_to,
            "lang": LANG_NAMES[lang]
        })

    return out

def main():
    parser = argparse.ArgumentParser(description="Word-level noun translation using NLLB.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]

    translated = translate_random_lang(dataset, args.seed, args.use_gpu)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with jsonlines.open(OUTPUT_PATH, mode="w") as writer:
        writer.write_all(translated)

    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
