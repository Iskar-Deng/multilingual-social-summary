import sys
import nltk
import json
import numpy as np
import pandas as pd
import tqdm
from tabulate import tabulate

if __name__ == "__main__":

    tldr_file = sys.argv[1]
    output_file = sys.argv[2]
    unique_subreddits = set()
    num_posts = 0
    body_lens = []
    summary_lens = []
    total_summary_tokens = 0

    with open(tldr_file, "r", encoding='utf-8') as file:

        total_lines = sum(1 for _ in tqdm.tqdm(file, desc="Counting Lines"))

        file.seek(0) # return to start of file

        for line in tqdm.tqdm(file, total=total_lines, desc="Processing Data"):
            
            if line.strip():
                try:
                    data = json.loads(line)
                    body_tokens = nltk.word_tokenize(data["normalizedBody"])
                    summary_tokens = nltk.word_tokenize(data["summary"])
                    subreddit_id = data.get("subreddit_id")
                    if subreddit_id is not None:
                        unique_subreddits.add(data.get("subreddit_id"))
                    num_posts += 1
                    body_lens.append(len(body_tokens))
                    summary_lens.append(len(summary_tokens))

                except json.JSONDecodeError as e:
                    print(f"Error Decoding Line")

    num_subreddits = len(unique_subreddits)
    body_lens = np.array(body_lens)
    summary_lens = np.array(summary_lens)

    mean_body_len = np.mean(body_lens)
    median_body_len = np.median(body_lens)
    min_body_len = np.min(body_lens)
    max_body_len = np.max(body_lens)
    std_body_len = np.std(body_lens)

    mean_summary_len = np.mean(summary_lens)
    median_summary_len = np.median(summary_lens)
    min_summary_len = np.min(summary_lens)
    max_summary_len = np.max(summary_lens)
    std_summary_len = np.std(summary_lens)

    df_data = {
        "Min" : [min_body_len, min_summary_len],
        "Max" : [max_body_len, max_summary_len],
        "Mean" : [mean_body_len, mean_summary_len],
        "Median" : [median_body_len, median_summary_len],
        "Std. Deviation" : [std_body_len, std_summary_len]
    }

    row_names = ["Body", "Summary"]

    df = pd.DataFrame(data=df_data, index=row_names)

    with open(output_file, "w") as file:

        file.write("TL;DR Dataset Statistics\n\n")
        file.write(f"Total Posts: {num_posts}\nUnique Subreddits: {num_subreddits}\n\n")

        file.write(tabulate(df, headers="keys", tablefmt="pretty"))
