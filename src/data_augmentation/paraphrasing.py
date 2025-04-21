import json
import jsonlines
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def paraphrasing(input_file, src_tgt_model, tgt_src_model, output_file, use_gpu):
    final_output = []
    
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    
    # Initialize translation models
    src_tgt_tokenizer = AutoTokenizer.from_pretrained(src_tgt_model)
    src_tgt_model = AutoModelForSeq2SeqLM.from_pretrained(src_tgt_model).to(device)
    
    tgt_src_tokenizer = AutoTokenizer.from_pretrained(tgt_src_model)
    tgt_src_model = AutoModelForSeq2SeqLM.from_pretrained(tgt_src_model).to(device)
    
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]
    
    for item in data:
        input_text = item["input_text"]
        to_paraphrase = item["pseudo_summary_extractive"]
        
        # Translate to target language
        inputs = src_tgt_tokenizer([to_paraphrase], return_tensors="pt", padding=True).to(device)
        outputs = src_tgt_model.generate(**inputs)
        tgt_text = src_tgt_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Translate back to source language
        inputs = tgt_src_tokenizer([tgt_text], return_tensors="pt", padding=True).to(device)
        outputs = tgt_src_model.generate(**inputs)
        paraphrased = tgt_src_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        final_output.append({
                "input_text": input_text,
                "pseudo_summary_paraphrased": paraphrased
            })
    
    with jsonlines.open(output_file, mode='w') as writer:
        writer.write_all(final_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Paraphrasing')
    parser.add_argument('--input_file', type=str, help='Path to input text file in JSON')
    parser.add_argument('--src_tgt_model', type=str, help='Model for translation from source to target language')
    parser.add_argument('--tgt_src_model', type=str, help='Model for translation from target back to source language')
    parser.add_argument('--output_file', type=str, help='Path to output file')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for inference')
    
    args = parser.parse_args()
    
    paraphrasing(args.input_file, args.src_tgt_model, args.tgt_src_model, args.output_file, args.use_gpu)