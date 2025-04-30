"""
analyze_code-switch.py

This script process the Reddit Code-switch dataset in CSV format and performs two tasks:
1. Tokenizes the 'text' field using a specified tokenizer (default: google/mt5-base).
2. Calculates basic statistics (min, max, mean, median, std deviation) for the tokenized input and output lengths

Outputs:
- A new JSONL file where each line contains the primary language of the text, the second language, and the tokenized text for use as test data
- A text file summarizing dataset statistics

Usage (command line):
    python src/data_processing/analyze_code-switch.py <input_csv> <output_tokenized_jsonl> <output_stats_txt>

Arguments:
    input_csv: Path to the original Reddit code-switch dataset (.csv)
    output_tokenized_jsonl: path to save the tokenized text, along with the languages contained within them
    output_stats_txt: path to save the dataset statistics

Example:
    python src/data_processing/analyze_code-switch.py data/cs_main_reddit_corpus.csv tokenized_codeswitch.jsonl code-swtich_stats.txt

Notes:
- The script defaults to using the MT5Tokenizer. If your checkpoint is T5, you may see a warning message. 
  This can generally be ignored unless switching tokenization behavior intentionally.

"""

import sys
import json
import numpy as np
import pandas as pd
import tqdm
from tabulate import tabulate
from transformers import MT5Tokenizer

def load_and_tokenize(input_path, tokenizer:MT5Tokenizer):

    data = pd.read_csv(input_path, header=0)

    data["tokens"] = data["Text"].apply(lambda x: tokenizer.tokenize(x))

    return data

import pandas as pd

def write_tokenized_data(data: pd.DataFrame, output_data_path: str, chunk_size: int = 10000):

    with open(output_data_path, 'w') as file:
        for start in range(0, len(data), chunk_size):
            chunk = data.iloc[start:start + chunk_size]

            chunk[["Lang1", "Lang2", "tokens"]].to_json(file, orient="records", lines=True)


def calculate_stats(data:pd.DataFrame):

    data["token_lengths"] = data["tokens"].apply(len)

    num_posts = len(data["tokens"])
    unique_subreddits = len(data["Subreddit"].unique())
    lang1 = data["Lang1"].unique()
    lang2 = data["Lang2"].unique()

    text_min = data["token_lengths"].min()
    text_max = data["token_lengths"].max()
    text_mean = data["token_lengths"].mean()
    text_median = data["token_lengths"].median()
    text_std = data["token_lengths"].std()

    return {
        "num_posts":num_posts,
        "unique_subreddits":unique_subreddits,
        "lang1":lang1,
        "lang2":lang2
    },{
        "Min":text_min,
        "Max":text_max,
        "Mean":text_mean,
        "Median":text_median,
        "Std. Deviation":text_std
    }

def write_stats(gen_stats, text_stats, output_stats_path):

    df = pd.DataFrame([text_stats], index=["Text"])

    with open(output_stats_path, "w") as f:

        f.write("CodeSwitch-Reddit Dataset Statistcs\n\n")
        f.write(f'Total posts: {gen_stats["num_posts"]}\n')
        f.write(f'Unique Subreddits: {gen_stats["unique_subreddits"]}\n')
        lang1 = ", ".join(gen_stats["lang1"])
        lang2 = ", ".join(gen_stats["lang2"])
        f.write(f"Unique Primary Languages: {lang1}\n")
        f.write(f"Unique Secondary Languages: {lang2}\n")
        f.write(tabulate(df, headers="keys", tablefmt="pretty"))


def main(input_path, output_data_path, output_stats_path, tokenizer_name="google/mt5-base"):

    tokenizer = MT5Tokenizer.from_pretrained(tokenizer_name)
    data = load_and_tokenize(input_path, tokenizer)

    write_tokenized_data(data, output_data_path)
    
    gen_stats, text_stats = calculate_stats(data)
    write_stats(gen_stats, text_stats, output_stats_path)

if __name__ == "__main__":
    
    input_path = sys.argv[1]
    output_data_path = sys.argv[2]
    output_stats_path = sys.argv[3]
    main(input_path, output_data_path, output_stats_path)
    