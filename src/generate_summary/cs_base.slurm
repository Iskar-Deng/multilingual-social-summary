#!/bin/bash
#SBATCH --job-name=base_cs
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
#SBATCH --output=/gscratch/stf/mx727/logs/0530_baseline_sum_cs_10k_%j.out
#SBATCH --error=/gscratch/stf/mx727/logs/0530_baseline_sum_cs_10k_%j.err
#SBATCH --requeue

# === Activate your conda environment ===
source /gscratch/scrubbed/dinuoz/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/scrubbed/dinuoz/loraenv

# === Run script ===
python /gscratch/stf/mx727/summary/src/generate_10k_cs_summaries.py \
  --csv_path /gscratch/stf/mx727/summary/data/cs_main_reddit_10k.csv \
  --checkpoint /gscratch/stf/mx727/multilingual-social-summary/checkpoints/zoey/tldr_clean_100k/checkpoint-4000 \
  --output_path /gscratch/stf/mx727/summary/cs_test/baseline_cs_4000.jsonl \
  --batch_size 16
