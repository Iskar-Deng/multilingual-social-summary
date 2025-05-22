#!/bin/bash
#SBATCH --job-name=ctokenize_mt5
#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/gscratch/stf/mx727/multilingual-social-summary/logs/tokenize_mt5_%j.out
#SBATCH --error=/gscratch/stf/mx727/multilingual-social-summary/logs/tokenize_mt5_%j.err

# Activate your environment
source /gscratch/stf/mx727/multilingual-social-summary/socialsum-venv/bin/activate

export HF_DATASETS_CACHE=/gscratch/stf/mx727/hf_cache
export TRANSFORMERS_CACHE=/gscratch/stf/mx727/hf_cache
export HF_MODULES_CACHE=/gscratch/stf/mx727/hf_cache
export XDG_CACHE_HOME=/gscratch/stf/mx727/hf_cache

# Tokenize data
python /gscratch/stf/mx727/multilingual-social-summary/src/model_train/tokenize_mt5.py \
  --data_path /gscratch/stf/mx727/multilingual-social-summary/data/splits/tldr_train.jsonl \
  --output_path /gscratch/stf/mx727/multilingual-social-summary/data/splits/tokenized_tldr_train_4c.jsonl
