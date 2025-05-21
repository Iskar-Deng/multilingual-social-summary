import sys
import pandas as pd
import numpy as np
from tabulate import tabulate
from transformers import MT5Tokenizer
from datasets import Dataset

def load_and_tokenize(input_path, tokenizer, max_input_length=512):
    df = pd.read_csv(input_path)

    input_ids_list = []
    attention_mask_list = []
    lang1_list = []
    lang2_list = []

    for _, row in df.iterrows():
        text = str(row["Text"])
        lang1 = row["Lang1"]
        lang2 = row["Lang2"]

        input_enc = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_input_length,
            return_attention_mask=True
        )

        input_ids_list.append(input_enc["input_ids"])
        attention_mask_list.append(input_enc["attention_mask"])
        lang1_list.append(lang1)
        lang2_list.append(lang2)

    dataset = Dataset.from_dict({
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "Lang1": lang1_list,
        "Lang2": lang2_list
    })

    return dataset

def calculate_stats(dataset):
    lengths = [len([t for t in ex if t != 0]) for ex in dataset["input_ids"]]
    return {
        "num_posts": len(dataset),
        "Min": np.min(lengths),
        "Max": np.max(lengths),
        "Mean": np.mean(lengths),
        "Median": np.median(lengths),
        "Std. Deviation": np.std(lengths),
    }

def write_stats(stats, output_stats_path, dataset):
    df = pd.DataFrame([{
        "Min": stats["Min"],
        "Max": stats["Max"],
        "Mean": stats["Mean"],
        "Median": stats["Median"],
        "Std. Deviation": stats["Std. Deviation"]
    }], index=["Text"])

    with open(output_stats_path, "w") as f:
        f.write("CodeSwitch-Reddit Test Dataset Statistics\n\n")
        f.write(f"Total Posts: {stats['num_posts']}\n")
        f.write(f"Unique Primary Languages: {', '.join(set(dataset['Lang1']))}\n")
        f.write(f"Unique Secondary Languages: {', '.join(set(dataset['Lang2']))}\n\n")
        f.write(tabulate(df, headers="keys", tablefmt="pretty"))

def main(input_path, output_dataset_path, output_stats_path, tokenizer_name="google/mt5-base"):
    tokenizer = MT5Tokenizer.from_pretrained(tokenizer_name)
    dataset = load_and_tokenize(input_path, tokenizer)

    dataset.save_to_disk(output_dataset_path)

    stats = calculate_stats(dataset)
    write_stats(stats, output_stats_path, dataset)

if __name__ == "__main__":
    input_path = sys.argv[1]
    output_dataset_path = sys.argv[2]
    output_stats_path = sys.argv[3]
    main(input_path, output_dataset_path, output_stats_path)
