##Choose at least one to at most half of the total NN and NNs to be randomly translated into one of the five languages.
import json
import jsonlines
import random
import argparse
import torch
from transformers import MarianTokenizer, MarianMTModel
from tqdm import tqdm
import nltk
from nltk import word_tokenize, pos_tag
nltk.download("averaged_perceptron_tagger_eng")

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

#Randomly select nouns and a language to translate the input text to
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

    for entry in tqdm(dataset, desc="Translating entries", unit="entry"):
        raw = entry.get("input_text", "")
        words = word_tokenize(raw)
        tags = pos_tag(words)
        noun_indices = [i for i, (_, tag) in enumerate(tags) if tag == ("NN" or "NNS") ]
        if not noun_indices:
            out.append(entry)
            continue

        num_to_translate = random.randint(1, max(1, len(noun_indices) // 2))
        selected_noun_indices = random.sample(noun_indices, num_to_translate)

        translated_from_to = []
        lang = random.choice(list(LANG_MODELS.keys()))

        for idx in selected_noun_indices:
            original_word = words[idx]
            tok, mdl = preloaded[lang]
            translated_word = translate_text(original_word, tok, mdl)
            words[idx] = translated_word
            translated_from_to.append({
                "from": original_word,
                "to": translated_word
            })

        mixed_input = " ".join(words)

        out.append({
            "input_text": mixed_input,
            "summary_text": entry.get("summary_text", ""),
            "translated_sentences": translated_from_to,
            "translated_to": LANG_NAMES[lang]
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

