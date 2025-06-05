# Summarization of Code-Switched Social Media Content

![Test Image](assets/some-more-mygo-ave-mujica-emotes-v0-i4y71m1d5wte1.gif)

## Project Overview

This project explores multilingual summarization of user-generated content on Reddit, with a focus on handling code-switched, multilingual, and conversational posts.

We fine-tune the [mT5-base model](https://huggingface.co/google/mt5-base) on the English TL;DR dataset as a baseline system. To enable multilingual and informal summarization, we augment the TL;DR training data using three translation strategies that cover multiple target languages drawn from the CodeSwitch-Reddit dataset.

The system is evaluated through two complementary tracks:
- **Reference-based evaluation** on English TL;DR test data using BERTScore;
- **Reference-free evaluation** on CodeSwitch-Reddit data using LaSE to measure multilingual summarization quality without gold summaries.

This end-to-end workflow allows us to assess the impact of multilingual data augmentation on both monolingual and multilingual summarization tasks.

Let's get started!

---

## Experiment Pipeline
### Setup

First, clone the repo and install dependencies:
```python
git clone https://github.com/Iskar-Deng/multilingual-social-summary
cd mission-impossible-language-models
pip install -r requirements.txt
```

Then, download the dataset [Zenodo (Webis-TLDR-17)](https://zenodo.org/records/1043504) and [CodeSwitch (UofT)](https://www.cs.toronto.edu/~ella/code-switch.reddit.tar.gz), extract them, and place the required files in the same folder:
| Dataset Name         | Source                                                                                     | Filename                            |
|----------------------|--------------------------------------------------------------------------------------------|-------------------------------------|
| TL;DR Reddit         | [Zenodo (Webis-TLDR-17)](https://zenodo.org/records/1043504)                              | `corpus-webis-tldr-17.json`         |
| CodeSwitch-Reddit    | [CodeSwitch (UofT)](https://www.cs.toronto.edu/~ella/code-switch.reddit.tar.gz)          | `cs_main_reddit_corpus.csv`         |

Your external data directory should look like this:

```
/path/to/your/DATA_ROOT/
├── corpus-webis-tldr-17.json
└── cs_main_reddit_corpus.csv
```

Finally, set the `DATA_ROOT` variable in `utils.py` to the absolute path of this directory on your system.

---

### Data preprocessing

After downloading the datasets, you will need to split the training and validation data from the TL;DR dataset:

```bash
python -m preprocessing.split_tldr --train_size 100000 --val_size 1000
```

Also, sample a subset from the CodeSwitch-Reddit corpus:

```bash
python -m preprocessing.slice_cs --num_samples 10000
```

### Data augmentation
We simulate code-switching by translating parts of the TL;DR training set into five target languages: Tagalog, Greek, Romanian, Indonesian, and Russian. 

First, download the translation models from HuggingFace:
```bash
python -m spacy download en_core_web_sm
```

Then, run `augmentation\run_augmentation.sh` to generate the augmented data, or use each script seperately:

- `trans_full.py`: full-document translation into 5 target languages.
- `trans_sent.py`: partial sentence-level code-switching.
- `trans_noun.py`: code-switch common nouns only.

### Tokenization

We tokenize all TL;DR variants using the mT5 tokenizer (`google/mt5-base`) with max lengths of 512 (input) and 64 (summary). Padding is applied to both.
Run the `preprocessing\run_tokenize.sh` to tokenize all datasets:

Use `--use_cache` to enable HuggingFace cache if needed.
> To use `--use_cache`, first set `HF_CACHE_PATH` in `utils.py`.

---

### Training

We fine-tune the mT5 model (`google/mt5-base`) with LoRA adapters on each TL;DR variant. The training script supports checkpoint resumption and progress logging.

Set the `CHECKPOINT_PATH` in `utils.py`, and then run:

```bash
python -m training/train_model --variant $variant
```

You can also use the batch script:

```bash
bash run_train_all.sh
```
---

### Evaluation

We evaluate summaries using:
- **BERTScore** on TL;DR (with reference)
- **LaSE** on CodeSwitch (reference-free)

Set `RESULTS_PATH` in `utils.py` before running.

#### Step 1: Generate summaries

To generate summaries from a checkpoint:

```bash
python -m evaluation.generate_summary --variant base --step 500
```

To generate from all checkpoints:

```bash
bash run_generate_all.sh
```

#### Step 2: Evaluate with BERTScore and LaSE

Run the batch script `evaluation/run_eval_all.sh`, or evaluate on a single checkpoint:

```bash
python -m evaluation.run_BERT --variant base --step 500
python -m evaluation.run_LaSE --variant base --step 500
```

### Analysis

We provide scripts to analyze the dataset and the results. See the notebooks under `/analysis`:

- `analyze_cs.ipynb`: Inspect the CodeSwitch dataset
- `analyze_tldr.ipynb`: Analyze TL;DR content and summary characteristics
- `plot_eval_scores.ipynb`: Visualize BERTScore and LaSE across checkpoints

---

## File Structure and Usage
```
.
├── preprocessing/          # Scripts for data splitting, sampling, and tokenization
├── augmentation/           # Data augmentation via code-switching (full, noun, sent)
├── training/               # Training scripts with LoRA support
├── evaluation/             # Generation and evaluation (BERTScore / LaSE)
├── analysis/               # Jupyter notebooks for data & score analysis
├── utils.py                # Paths and shared constants
└── run_*.sh                # Scripts for batch training, generation, and evaluation
```

## Contribution

- Data augmentation: Zoey Zhou  
- Model fine-tuning: Nathalia Xu  
- Benchmark building: Jordan Jin  
- Dataset analysis: Bartosz Mamro  
- Code integration: Iskar Deng
