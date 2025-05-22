#Devide input into senetnces, choose at least one to at most half of the total sentences to be randomly translated 
#into one of the five languages.

import json
import jsonlines
import argparse
import torch
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import nltk
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")
from nltk.tokenize import sent_tokenize

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
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

def translate_random_lang(dataset, seed, use_gpu):
    if use_gpu and torch.cuda.is_available():
        device = "cuda"
    elif use_gpu and torch.mps.is_available():
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
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible language assignment."
    )
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for inference')
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        print("GPU is available")
    else:
        print("GPU is not available")

    with open(args.input, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    translated = translate_random_lang(data, args.seed,args.use_gpu)

    with jsonlines.open(args.output, mode='w') as writer:
        writer.write_all(translated)

    print(f"Done — translated {len(translated)} entries → {args.output}")
