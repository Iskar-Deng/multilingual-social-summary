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

