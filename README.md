# Summarization of Code-Switched Social Media Content

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
  An English monolingual summarization dataset collected from Reddit, where users provide short TL;DR summaries for each entry.
-  [CodeSwitch-Reddit](https://www.cs.toronto.edu/~ella/code-switch.reddit.tar.gz)  
  A multilingual Reddit dataset containing code-switched entries across several languages, without human-written summaries.

## Augmented Data

- https://drive.google.com/drive/folders/1ffvffJ2Bki3H7e64C9JP4fmqI8BQAwJu?usp=drive_link

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
├── data_split
│   ├── generate_indices.py         # Script to generate random test set indices
│   ├── generate_indices.slurm      # Slurm script to submit for index generation
│   ├── split_by_index.py           # Script to split test/train sets by index
│   ├── split_by_index.slurm        # Slrum script to submit for test/train splits
├── data_tokenization
│   ├── clean_tokenized.py                # Script to remove unnecessary fields in tokenized tldr
│   ├── tokenize_tldr.py                  # Slurm script to tokenize tldr
│   ├── tokenize_tldr.slurm               # Slurm script to submit for tldr tokenization
│   ├── tokenize_trans_full.py            # Slurm script to tokenize trans_full
│   ├── tokenize_trans_full.slurm         # Slurm script to submit for trans_full tokenization
│   ├── tokenize_trans_noun.py            # Slurm script to tokenize trans_noun
│   ├── tokenize_trans_noun.slurm         # Slurm script to submit for trans_noun tokenization
│   ├── tokenize_trans_sent.py            # Slurm script to tokenize trans_sent
│   ├── tokenize_trans_sent.slurm         # Slurm script to submit for trans_sent tokenization
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
├── model_train_tokenized
│   ├── train_mt5_tok.py            # Training script for the MT5 model on tokenized tldr
│   ├── train_mt5_tok.slurm         # Slurm script to submit for training
│   ├── train_trans_full.py         # Training script for the MT5 model on tokenized trans_full
│   ├── train_trans_full.slurm      # Slurm script to submit for training
│   ├── train_trans_noun.py         # Training script for the MT5 model on tokenized trans_noun
│   ├── train_trans_noun.slurm      # Slurm script to submit for training
│   ├── train_trans_sent.py         # Training script for the MT5 model on tokenized trans_sent
│   └── train_trans_sent.slurm      # Slurm script to submit for training
├── stress_test
│   ├── stress_pipeline.sh       # Full training and eval pipeline
│   ├── stress_pipeline.submit   # HTCondor submit script
│   ├── stress_pipeline.slurm    # SLURM submit script (Hyak)
│   ├── evaluate_tldr.py         # TL;DR evaluation with BERTScore
│   ├── evaluate_codeswitch.py   # CodeSwitch evaluation with LaSE
│   ├── tldr_train_3000.jsonl    # Sample training data
│   ├── tldr_test_300.jsonl      # Sample TL;DR test data
│   ├── codeswitch_test_100.jsonl # Sample CodeSwitch test data
```

### Key Folders and Files:
- **data_augmentation**: Contains subfolders for MarianMT and NLLB models, and scripts for different data augmentation strategies (e.g., multilingual, monolingual).
- **data_processing**: Scripts to generate and analyze datasets.
- **evaluation**: Contains evaluation scripts for BERTScore, LaSE, and ROUGE, as well as related data.
- **model_test**: Scripts to test the performance of fine-tuned and HuggingFace models.
- **model_train**: Scripts to train the MT5 model and associated Condor jobs.
- **stress_test**: Scripts to test the pipeline.

## How to Run Stress Test Pipeline (Update)

### 1. On Patas (Condor)

```bash
condor_submit src/stress_test/stress_pipeline.submit
```

### 2. On Hyak (SLURM)

```bash
sbatch src/stress_test/stress_pipeline.slurm
```

### 3. Locally or directly on a node (debugging only)

```bash
bash src/stress_test/stress_pipeline.sh
```

This script will:
- Fine-tune the model on TL;DR data (3,000 training samples)
- Evaluate the fine-tuned model on TL;DR using BERTScore (300 test samples)
- Evaluate on code-switched data using LaSE (100 test samples)
- Log time, GPU, CPU usage, and evaluation results to `logs/`

- **Note:**: Update to your absolute path.
## How to Run

### 1. Setup environment

- Use Python 3.6.8.
- Install dependencies inside your virtual environment:

```bash
pip install -r requirements.txt
```

### 2. Prepare datasets

#### Download datasets
- Download TL;DR dataset: https://zenodo.org/records/1043504
- Download CodeSwitch-Reddit dataset: https://www.cs.toronto.edu/~ella/code-switch.reddit.tar.gz
- Download augmented TL;DR datasets: https://drive.google.com/drive/folders/1ffvffJ2Bki3H7e64C9JP4fmqI8BQAwJu?usp=drive_link

Unpack and place them under a `data/` directory.

#### Tokenize datasets
Example commands are for training sets. Modify ```data_path``` to tokenize test sets.

- Tokenize TL;DR 
```bash
python src/data_tokenization/tokenize_tldr.py \
  --data_path data/splits/tldr_train.jsonl \
  --output_path data/splits/tokenized_train
```
Or submit as a Slurm job on Hyak:

```bash
sbatch src/data_tokenization/tokenize_tldr.slurm
```

- Clean tokenized TL;DR (Optional) to remove unnecessary fields
```bash
python src/data_tokenization/clean_tokenized.py
```

- Tokenize translate_full
```bash
python src/data_tokenization/tokenize_trans_full.py \
  --data_path data/splits/trans_full_train.jsonl \
  --output_path data/splits/tokenized_train
```
Or submit as a Slurm job on Hyak:

```bash
sbatch src/data_tokenization/tokenize_trans_full.slurm
```

- Tokenize translate_nouns
```bash
python src/data_tokenization/tokenize_trans_noun.py \
  --data_path data/splits/trans_noun_train.jsonl \
  --output_path data/splits/tokenized_train
```
Or submit as a Slurm job on Hyak:

```bash
sbatch src/data_tokenization/tokenize_trans_noun.slurm
```

- Tokenize translate_sentence
```bash
python src/data_tokenization/tokenize_trans_sent.py \
  --data_path data/splits/trans_sent_train.jsonl \
  --output_path data/splits/tokenized_train
```
Or submit as a Slurm job on Hyak:

```bash
sbatch src/data_tokenization/tokenize_trans_sent.slurm
```

### 3. Train baseline model

```bash
python3 src/model_train_tokenized/train_mt5_tok.py \
  --tokenized_path data/splits/tokenized_train_clean \
  --output_dir checkpoints/mt5_base \
  --batch_size 4 \
  --grad_accum_steps 4 \
  --num_epochs 2 \
  --log_steps 100 \
  --num_workers 4 
```

Or submit as a Slurm job on Hyak:

```bash
sbatch src/model_train_tokenized/train_mt5_tok.slurm
```

### 4. Train augmented model
#### With translate_full dataset
```bash
python3 /src/model_train_tokenized/train_trans_full.py \
  --tokenized_path /data/splits/tokenized_train/trans_full \
  --output_dir /checkpoints/trans_full \
  --batch_size 4 \
  --grad_accum_steps 4 \
  --num_epochs 2 \
  --log_steps 100 \
  --num_workers 4 
```

- Or submit as a Slurm job on Hyak:

```bash
sbatch src/model_train_tokenized/train_trans_full.slurm
```

#### With translate_nouns dataset
```bash
python3 /src/model_train_tokenized/train_trans_noun.py \
  --tokenized_path /data/splits/tokenized_train/trans_noun \
  --output_dir /checkpoints/trans_noun \
  --batch_size 4 \
  --grad_accum_steps 4 \
  --num_epochs 2 \
  --log_steps 100 \
  --num_workers 4 
```

- Or submit as a Slurm job on Hyak:

```bash
sbatch src/model_train_tokenized/train_trans_noun.slurm
```
#### With translate_sentence dataset
```bash
python3 /src/model_train_tokenized/train_trans_sent.py \
  --tokenized_path /data/splits/tokenized_train/trans_sent \
  --output_dir /checkpoints/trans_sent \
  --batch_size 4 \
  --grad_accum_steps 4 \
  --num_epochs 2 \
  --log_steps 100 \
  --num_workers 4 
```

- Or submit as a Slurm job on Hyak:

```bash
sbatch src/model_train_tokenized/train_trans_sent.slurm
```

### 5. Run evaluation

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

### 6. Run data augmentation (optional)

For example, to run multilingual input translation:

```bash
python src/data_augmentation/nllb/translate_nouns/translate_nouns.py data/corpus-webis-tldr-17.json output.jsonl 42
```

Or submit via Condor:

```bash
condor_submit src/data_augmentation/nllb/translate_nouns/translate_nouns.cmd
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



