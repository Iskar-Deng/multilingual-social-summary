#!/bin/bash
#SBATCH --job-name=full_sum
#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --gres=gpu:1
#SBATCH --qos=ckpt-gpu
#SBATCH --constraint=a100
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/gscratch/stf/mx727/logs/0530_full_sum_tldr_10k_%j.out
#SBATCH --error=/gscratch/stf/mx727/logs/0530_full_sum_tldr_10k_%j.err
#SBATCH --requeue

# === Activate your conda environment ===
source /gscratch/scrubbed/dinuoz/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/scrubbed/dinuoz/loraenv

# === Run script ===
python /gscratch/stf/mx727/summary/src/generate_1k_tldr_summaries.py \
  --jsonl_path /gscratch/stf/mx727/summary/data/val_1k.jsonl \
  --checkpoint /gscratch/stf/mx727/multilingual-social-summary/checkpoints/zoey/trans_full_100k/checkpoint-2000 \
  --output_path /gscratch/stf/mx727/summary/tldr_val/full_tldr_2000.jsonl \
  --batch_size 16
