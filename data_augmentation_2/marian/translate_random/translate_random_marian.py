#!/usr/bin/env python3
import json
import jsonlines
import random
import argparse
import torch
from transformers import MarianTokenizer, MarianMTModel
from tqdm import tqdm

#  1) Define your target languages & corresponding Opus-MT checkpoints and names
LANG_MODELS = {
    "tl": "Helsinki-NLP/opus-mt-en-tl",  # Tagalog
    "el": "Helsinki-NLP/opus-mt-en-el",  # Greek
    "ro": "Helsinki-NLP/opus-mt-en-ro",  # Romanian
    "id": "Helsinki-NLP/opus-mt-en-id",  # Indonesian
    "ru": "Helsinki-NLP/opus-mt-en-ru",  # Russian
}
LANG_NAMES = {
    "tl": "Tagalog",
    "el": "Greek",
    "ro": "Romanian",
    "id": "Indonesian",
    "ru": "Russian",
}

def load_model(name: str):
    """Load & return (tokenizer, model) for a given HF checkpoint."""
    tok = MarianTokenizer.from_pretrained(name)
    mdl = MarianMTModel.from_pretrained(name)
    return tok, mdl


def translate_text(text: str, tokenizer: MarianTokenizer, model: MarianMTModel) -> str:
    """Tokenize & translate a single string."""
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def translate_random_lang (dataset, seed, use_gpu):
    # 2) Pre-load all models & tokenizers onto GPU if available
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    preloaded = {
        lang: load_model(checkpt)
        for lang, checkpt in LANG_MODELS.items()
    }
    for tok, mdl in preloaded.values():
        mdl.to(device)

    random.seed(seed)
    out = []
    langs = list(LANG_MODELS.keys())

    for entry in tqdm(dataset, desc="Translating entries", unit="entry"):
        lang = random.choice(langs)
        tokenizer, model = preloaded[lang]

        # Reconstruct from tokens if available
        raw = entry.get("input_text", "")
        translated = translate_text(raw, tokenizer, model)
        out.append({
            "input_text": translated,
            "summary_text": entry.get("summary_text", ""),
            "translated_to": LANG_NAMES.get(lang)
        })


    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Translate each JSON entry's tokenized input into a random target language."
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Path to source JSON (list of {input_tokens, summary_text})."
    )
    parser.add_argument(
        "-o", "--output", default="translated_dataset.json",
        help="Path to write translated JSON."
    )
    parser.add_argument(
        "--seed", type=int, default=123,
        help="Random seed for reproducible language assignment."
    )
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for inference')
    args = parser.parse_args()

    # Load data
    with open(args.input, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # Translate
    translated = translate_random_lang(data, args.seed, args.use_gpu)

    # Save
    with jsonlines.open(args.output, mode='w') as writer:
        writer.write_all(translated)

    print(f"Done — translated {len(translated)} entries → {args.output}")

    # load your data
    