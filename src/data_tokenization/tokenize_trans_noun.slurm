#!/bin/bash
#SBATCH --job-name=tokenize_aug
#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/gscratch/stf/mx727/multilingual-social-summary/logs/tokenize_noun_%j.out
#SBATCH --error=/gscratch/stf/mx727/multilingual-social-summary/logs/tokenize_noun_%j.err

# Activate your environment
source /gscratch/stf/mx727/multilingual-social-summary/socialsum-venv/bin/activate

export HF_DATASETS_CACHE=/gscratch/stf/mx727/hf_cache
export TRANSFORMERS_CACHE=/gscratch/stf/mx727/hf_cache
export HF_MODULES_CACHE=/gscratch/stf/mx727/hf_cache
export XDG_CACHE_HOME=/gscratch/stf/mx727/hf_cache

# Tokenize data
python /gscratch/stf/mx727/multilingual-social-summary/src/data_tokenization/tokenize_trans_noun.py \
  --data_path /gscratch/stf/mx727/multilingual-social-summary/data/splits/trans_noun_train.jsonl \
  --output_path /gscratch/stf/mx727/multilingual-social-summary/data/splits/tokenized_train/trans_noun
