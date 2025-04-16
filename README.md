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

## Directory Structure

multilingual-social-summary/
├── data/
│   ├── corpus-webis-tldr-17.json         # Original Reddit TL;DR dataset (input)
│   └── ...                               # Other raw or intermediate datasets
│
├── results/
│   ├── stats.txt                         # Length statistics for posts and summaries
│   ├── io_pairs.jsonl                    # Input/output tokenized pairs (for model training)
│   └── ...                               # Other output files (e.g., evaluation results)
│
├── scripts/
│   ├── tldr_analysis.sh                  # Shell script to run TL;DR preprocessing pipeline
│   └── ...                               # Additional utility scripts
│
├── src/
│   ├── tldr_analysis.py                  # Main data processing script (tokenization + stats)
│   ├── generate_toy_tokenized.py         # Script to generate tokenized toy dataset
│   └── ...                               # (Recommended) Other core modules for training/inference
│
├── README.md                             # Project overview and usage instructions
├── requirements.txt                      # Python dependencies
├── .gitignore                            # Files/directories excluded from version control

