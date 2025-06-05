# augmentation/trans_sent.py
# Author: Zoey Zhou

import os
import json
import jsonlines
import random
import argparse
import torch
import nltk
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils import DATA_PATH, NLLB_MODEL_NAME, NLLB_SRC_LANG, LANG_CODES, LANG_NAMES

# Ensure sentence tokenizer is downloaded
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

INPUT_PATH = os.path.join(DATA_PATH, "tldr_split", "tldr_train_base.jsonl")
OUTPUT_PATH = os.path.join(DATA_PATH, "tldr_split", "tldr_train_sent.jsonl")

# Load NLLB model and tokenizer
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_NAME, src_lang=NLLB_SRC_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_NAME)
    return tokenizer, model

# Translate a given sentence into the target language
def translate_text(text, tokenizer, model, tgt_lang_code):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    bos_id = tokenizer.convert_tokens_to_ids(tgt_lang_code)
    with torch.no_grad():
        out = model.generate(**inputs, forced_bos_token_id=bos_id)
    return tokenizer.decode(out[0], skip_special_tokens=True)

# Randomly translate one or more sentences in each input text to a random target language
def translate_random_sentences(dataset, seed, use_gpu):
    if use_gpu and torch.cuda.is_available():
        device = "cuda"
    elif use_gpu and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    tokenizer, model = load_model_and_tokenizer()
    model.to(device)
    random.seed(seed)
    out = []

    for entry in tqdm(dataset, desc="Translating entries", unit="entry"):
        raw = entry.get("input_text", "")
        sentences = sent_tokenize(raw)
        if not sentences:
            out.append(entry)
            continue

        # Select subset of sentences to translate
        num_to_translate = random.randint(1, max(1, len(sentences) // 2))
        selected_idxs = random.sample(range(len(sentences)), num_to_translate)
        lang = random.choice(list(LANG_CODES.keys()))
        tgt_lang_code = LANG_CODES[lang]

        for idx in selected_idxs:
            translated = translate_text(sentences[idx], tokenizer, model, tgt_lang_code)
            sentences[idx] = translated

        mixed_input = " ".join(sentences)
        out.append({
            "input_text": mixed_input,
            "summary_text": entry.get("summary_text", ""),
            "translated_sentences": selected_idxs,
            "translated_to": LANG_NAMES[lang]
        })

    return out

def main():
    parser = argparse.ArgumentParser(description="Sentence-level translation using NLLB.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]

    translated = translate_random_sentences(dataset, args.seed, args.use_gpu)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with jsonlines.open(OUTPUT_PATH, mode="w") as writer:
        writer.write_all(translated)

    print(f"✅ Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
