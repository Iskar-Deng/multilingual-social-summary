##Devide input into senetnces, choose at least one to at most half of the total sentences to be randomly translated 
#into one of the five languages.
import json
import jsonlines
import random
import argparse
import torch
from transformers import MarianTokenizer, MarianMTModel
from tqdm import tqdm
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")  
from nltk.tokenize import sent_tokenize

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


def translate_random_lang(dataset, seed, use_gpu):
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
        raw = entry.get("input_text", "")
        sentences = sent_tokenize(raw)

        num_to_translate = random.randint(1, max(1, len(sentences) // 2))
        selected_idxs = random.sample(range(len(sentences)), num_to_translate)

        for idx in selected_idxs:
            lang = random.choice(langs)
            tok, mdl = preloaded[lang]
            translated = translate_text(sentences[idx], tok, mdl)
            sentences[idx] = translated  

        mixed_input = " ".join(sentences)

        out.append({
            "input_text": mixed_input,
            "summary_text": entry.get("summary_text", ""),
            "translated_sentences": selected_idxs,
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

    with open(args.input, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    translated = translate_random_lang(data, args.seed, args.use_gpu)

    with jsonlines.open(args.output, mode='w') as writer:
        writer.write_all(translated)

    print(f"Done — translated {len(translated)} entries → {args.output}")

