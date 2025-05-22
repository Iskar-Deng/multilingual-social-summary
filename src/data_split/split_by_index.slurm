#!/bin/bash
#SBATCH --job-name=split_jsonl
#SBATCH --account=stf               # Replace with your account if needed
#SBATCH --partition=ckpt            # Use ckpt or compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00             # Should be quick unless disk is very slow
#SBATCH --output=logs/split_jsonl_%j.out
#SBATCH --error=logs/split_jsonl_%j.err

# === Activate environment ===
source /gscratch/stf/mx727/multilingual-social-summary/socialsum-venv/bin/activate

# === Run the Python script ===
python /gscratch/stf/mx727/multilingual-social-summary/src/data_test/split_by_index.py
