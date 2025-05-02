#Choose at least one to at most half of the total NN and NNs to be randomly translated into one of the five languages.
import json
import jsonlines
import argparse
import torch
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import spacy
import os

# load once
nlp = spacy.load("en_core_web_sm")



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
        src_lang=SRC_LANG    # tells the tokenizer your source is English by default
    )
    # AutoModelForSeq2SeqLM loads the correct translation model class
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
            #max_length=inputs["input_ids"].shape[-1] + 50  # optional
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

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

        noun_indices = [
            i for i, token in enumerate(doc)
            if token.pos_ == "NOUN"
        ]

        if not noun_indices:
            out.append(entry)
            continue

        num_to_translate = random.randint(1, max(1, len(noun_indices) // 2))
        selected_noun_indices = random.sample(noun_indices, num_to_translate)

        translated_from_to = []
        lang = random.choice(list(LANG_CODES.keys()))
        tgt_lang_code = LANG_CODES[lang]

        for idx in selected_noun_indices:
            original_word = words[idx]
            translated_word = translate_text(original_word, tokenizer, model, tgt_lang_code)
            words[idx] = translated_word
            translated_from_to.append({
                "from": original_word,
                "to": translated_word
            })

        mixed_input = " ".join(words)

        out.append({
            "input_text": mixed_input,
            "summary_text": entry.get("summary_text", ""),
            "translated_from_to": translated_from_to,
            "lang": LANG_NAMES[lang]
        })

    return out
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Translate each JSON entry into all target languages using NLLB."
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Path to a source JSONL file or directory containing .jsonl files."
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Full path to the output JSONL file."
    )

    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible language assignment."
    )
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for inference')
    args = parser.parse_args()

    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))

    if os.path.isdir(args.input):
        files = sorted([f for f in os.listdir(args.input) if f.endswith(".jsonl")])
    else:
        files = [args.input]

    for fname in files:
        input_path = fname if os.path.isfile(fname) else os.path.join(args.input, fname)
        output_path = args.output

        data = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))

        translated = translate_random_lang(data, args.seed, args.use_gpu)

        with jsonlines.open(output_path, mode='w') as writer:
            writer.write_all(translated)

    print(f"Done — translated {len(files)} file(s) into {args.output}")
