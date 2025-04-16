# Multilingual Social Media Summarization

This project explores multilingual summarization of user-generated content on Reddit, including code-switched and conversational posts.

We fine-tune the [mT5-base model](https://huggingface.co/google/mt5-base) on English TL;DR datasets, and adapt it for multilingual and informal inputs using a combination of Reddit datasets.

## Datasets

-  [TL;DR Reddit dataset](https://zenodo.org/records/1043504)
-  [CodeSwitch-Reddit](https://www.cs.toronto.edu/~ella/code-switch.reddit.tar.gz)

## Model

- [google/mt5-base](https://huggingface.co/google/mt5-base)
- Fine-tuned using Hugging Face Transformers
- Optional PEFT/LoRA support for efficiency

## Evaluation

- ROUGE
- BERTScore (XLM-R)
- LaSE (optional, reference-free)

## 📁 Project Structure

<details>
<summary>Click to expand</summary>

```text
multilingual-social-summary/
├── data/
│   ├── corpus-webis-tldr-17.json         # Raw TL;DR dataset
│
├── results/
│   ├── stats.txt                         # Summary statistics (post/summary lengths)
│   ├── io_pairs.jsonl                    # Tokenized input/output pairs
│
├── scripts/
│   ├── tldr_analysis.sh                  # Shell script to run preprocessing
│
├── src/
│   ├── tldr_analysis.py                  # Main tokenizer + stat script
│   ├── generate_toy_tokenized.py         # Script to generate toy data
│
├── requirements.txt                      # Python dependencies
├── README.md                             # Project overview and usage
├── .gitignore                            # Files excluded from Git
├── toy_data_tokenized.jsonl          # 100-sample tokenized toy dataset
```

</details>

---

## Folder Descriptions

| Folder/File | Description |
|-------------|-------------|
| `data/`     | Raw and intermediate data files (local only) |
| `results/`  | Processed outputs such as statistics and model inputs |
| `scripts/`  | Shell for running tasks |
| `src/`      | Main Python modules (preprocessing, future model use) |
| `requirements.txt` | Dependency file for environment setup |
| `.gitignore`       | Prevents unnecessary files from being tracked |
| `README.md`        | This file |

