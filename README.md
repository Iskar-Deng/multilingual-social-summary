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

## Environment and Dependency Usage Guidelines

This project is structured to run in a specific environment to ensure compatibility. We are currently using **Python 3.6.8** for consistency across the Patas server. Below are the key environment details:

- **Python Version**: 3.6.8
- **Transformers**: 4.12.0
- **Datasets**: 1.15.1
- **PyTorch**: 1.10.2

Please ensure that any new dependencies added are compatible with these versions. If you need to install new packages for local experiments, verify their compatibility with **Python 3.6.8** before adding them. It's recommended to run and test code directly on the server to maintain consistency.

**Note:** 
- Only the `patas-gn3.ling.washington.edu` node supports Python 3.6.8. Other nodes are running **Python 3.4**.
- Since Condor jobs are scheduled on the `patas-gn3` node, which uses **Python 3.6.8** by default, we will proceed based on this environment.
- **Downgrading to Python 3.4** is not feasible due to compatibility issues with key libraries.

---

## File Structure and Usage

The project is organized as follows:

```
├── data_augmentation
│   ├── marian              # MarianMT models for data augmentation
│   ├── nllb                # NLLB-200 models for data augmentation
├── data_processing
│   ├── analyze_tldr.py     # Script to analyze TLDR data
│   ├── generate_dataset.py # Script to generate datasets
├── evaluation
│   ├── evaluation_scripts
│   │   ├── eval_bert_score.py  # BERTScore evaluation script
│   │   ├── eval_LaSE.py        # LaSE evaluation script
│   │   ├── eval_rouge.py       # ROUGE evaluation script
│   ├── run_scripts
│   │   ├── run_eval_no_reference.py # Evaluation without reference summaries
│   │   ├── run_eval_with_reference.py # Evaluation with reference summaries
│   └── sample_data
│       ├── source_sum.jsonl    # Example input file for summarization tasks
│       └── sum_ref.jsonl       # Example reference summaries
├── model_test
│   ├── test_finetuned_model.py  # Test script for fine-tuned models
│   └── test_hf_model.py         # Test script for HuggingFace models
├── model_train
│   ├── train_mt5.py             # Training script for the MT5 model
│   ├── train_mt5.condor         # Condor job script for training
│   └── train_mt5.sh             # Shell script for training
```

### Key Folders and Files:
- **data_augmentation**: Contains subfolders for MarianMT and NLLB models, and scripts for different data augmentation strategies (e.g., multilingual, monolingual).
- **data_processing**: Scripts to generate and analyze datasets.
- **evaluation**: Contains evaluation scripts for BERTScore, LaSE, and ROUGE, as well as related data.
- **model_test**: Scripts to test the performance of fine-tuned and HuggingFace models.
- **model_train**: Scripts to train the MT5 model and associated Condor jobs.

---

## Important Notes

- Ensure compatibility with the specified **Python 3.6.8** environment, especially when adding new dependencies.
- **ROUGE** evaluation cannot be used on the Patas node due to version mismatches.
- If encountering issues with data augmentation using **fasttext**, fallback to using **langid** for language identification.
- **Testing** and **training** should be done on the specified server (patas-gn3) to avoid discrepancies with local setups.

## Contribution

🥚
