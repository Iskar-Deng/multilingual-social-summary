# utils.py
# Author: Iskar Deng

# ===============================
# Path Constants
# ===============================
DATA_PATH = "/Users/denghaowen/Desktop/multilingual-social-summary/data"
HF_CACHE_PATH = "/Users/denghaowen/Desktop/multilingual-social-summary/hf_cache"
CHECKPOINT_PATH = "/Users/denghaowen/Desktop/multilingual-social-summary/checkpoints"
RESULTS_PATH = "/Users/denghaowen/Desktop/multilingual-social-summary/results"

# ===============================
# Model Training & Evaluation Wrappers
# ===============================
TRAINING_ARGS = {
    "batch_size": 4,
    "grad_accum_steps": 2,
    "num_epochs": 3,
    "lr": 3e-5,
    "warmup_steps": 0,
    "save_steps": 8,
    "log_steps": 2,
    "num_workers": 0,
}

# ===============================
# Augmentation Constants
# ===============================
NLLB_MODEL_NAME = "facebook/nllb-200-distilled-600M"
NLLB_SRC_LANG = "eng_Latn"

LANG_CODES = {
    "tl": "tgl_Latn",      # Tagalog
    "el": "ell_Grek",      # Greek
    "ro": "ron_Latn",      # Romanian
    "id": "ind_Latn",      # Indonesian
    "ru": "rus_Cyrl",      # Russian
}

LANG_NAMES = {
    "tl": "Tagalog",
    "el": "Greek",
    "ro": "Romanian",
    "id": "Indonesian",
    "ru": "Russian",
}