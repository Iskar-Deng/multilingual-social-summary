# Multilingual Social Media Summarization

![Test Image](assets/some-more-mygo-ave-mujica-emotes-v0-i4y71m1d5wte1.gif)

## Project Overview

This project explores multilingual summarization of user-generated content on Reddit, with a focus on handling code-switched, multilingual, and conversational posts.

We fine-tune the [mT5-base model](https://huggingface.co/google/mt5-base) on the English TL;DR dataset as a baseline system. To enable multilingual and informal summarization, we augment the TL;DR training data using five translation strategies that cover multiple target languages drawn from the CodeSwitch-Reddit dataset.

The system is evaluated through two complementary tracks:
- **Reference-based evaluation** on English TL;DR test data using BERTScore;
- **Reference-free evaluation** on CodeSwitch-Reddit data using LaSE to measure multilingual summarization quality without gold summaries.

This end-to-end workflow allows us to assess the impact of multilingual data augmentation on both monolingual and multilingual summarization tasks.

## Datasets

-  [TL;DR Reddit dataset](https://zenodo.org/records/1043504)
    An English monolingual summarization dataset collected from Reddit, where users provide short TL;DR summaries for their posts.
-  [CodeSwitch-Reddit](https://www.cs.toronto.edu/~ella/code-switch.reddit.tar.gz)
    A multilingual Reddit dataset containing code-switched posts across several languages, without human-written summaries.

## Model

- [google/mt5-base](https://huggingface.co/google/mt5-base)  
  Fine-tuned using Hugging Face Transformers  
  Optional PEFT/LoRA support for efficiency

- [Helsinki-NLP/MarianMT](https://huggingface.co/Helsinki-NLP)  
  Used for data augmentation (multilingual translation)

- [facebook/nllb-200](https://huggingface.co/facebook/nllb-200-distilled-600M)  
  Used for data augmentation (multilingual translation)

## Evaluation

- BERTScore (reference-based)
- LaSE (reference-free)
- ROUGE (disabled due to environment limitations)

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

The src is organized as follows:

```
src/
├── data_augmentation
│   ├── marian              # MarianMT models for data augmentation
│   ├── nllb                # NLLB-200 models for data augmentation
├── data_processing
│   ├── analyze_tldr.py            # Script to analyze TLDR data
│   ├── analyze_code-switch.py     # Script to analyze CodeSwitch data
│   ├── generate_dataset.py        # Script to generate datasets
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

## How to Run

### 1. Setup environment

- Use Python 3.6.8.
- Install dependencies inside your virtual environment:

```bash
pip install -r requirements.txt
```

### 2. Prepare datasets

- Download TL;DR dataset: https://zenodo.org/records/1043504
- Download CodeSwitch-Reddit dataset: https://www.cs.toronto.edu/~ella/code-switch.reddit.tar.gz

Unpack and place them under a `data/` directory.

### 3. Train baseline model

```bash
python src/model_train/train_mt5.py --config configs/train_config.yaml
```

Or submit as a Condor job:

```bash
condor_submit src/model_train/train_mt5.condor
```

### 4. Run evaluation

- With references (TL;DR):

```bash
python src/evaluation/run_scripts/run_eval_with_reference.py path_to_your_output_file.jsonl --bert
```

Example format of `output_file.jsonl` (each line is a JSON object):
```json
{"summary_text": "This is the predicted summary.", "reference_text": "This is the gold summary."}
```

- Without references (CodeSwitch):

```bash
python src/evaluation/run_scripts/run_eval_no_reference.py path_to_your_output_file.jsonl --LaSE
```

Example format of `output_file.jsonl` (each line is a JSON object):
```json
{"input_text": "This is the input post text.", "summary_text": "This is the predicted summary."}
```

### 5. Run data augmentation (optional)

For example, to run multilingual input translation:

```bash
python src/data_augmentation/marian/translate_muiltilingual_nouns/translate_muiltilingual_nouns.py data/corpus-webis-tldr-17.json output.jsonl 42
```

Or submit via Condor:

```bash
condor_submit src/data_augmentation/marian/translate_muiltilingual_nouns/translate_muiltilingual_nouns.cmd
```

---

## Important Notes

- Ensure compatibility with the specified **Python 3.6.8** environment, especially when adding new dependencies.
- **ROUGE** evaluation cannot be used on the Patas node due to version mismatches.
- If encountering issues with data augmentation using **fasttext**, fallback to using **langid** for language identification.
- **Testing** and **training** should be done on the specified server (patas-gn3) to avoid discrepancies with local setups.

## GitIgnore Rules

```plaintext
# Checkpoints
checkpoints/

# Datasets
data/

# Python virtual environments
venv/
.venv/
socialsum-venv/

# Model files
*.bin
*.pt

# MacOS system files
.DS_Store

# Log files
*.log
logs/
```

## Contribution

This project was developed collaboratively by the team for a multilingual summarization research task.

- Data augmentation: Zoey Zhou  
- Model fine-tuning: Nathalia Xu  
- Benchmark building: Jordan Jin  
- Dataset analysis: Bartosz Mamro  
- Code integration: Iskar Deng



