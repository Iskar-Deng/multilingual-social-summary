#!/bin/bash
#SBATCH --job-name=split_data
#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --time=04:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=/mmfs1/home/mx727/multilingual-social-summary/src/data_test/split_data_%j.out
#SBATCH --error=/mmfs1/home/mx727/multilingual-social-summary/src/data_test/split_data_%j.err

# Activate your virtual environment
source /mmfs1/home/mx727/multilingual-social-summary/socialsum-venv/bin/activate

# Run your script
python3 /mmfs1/home/mx727/multilingual-social-summary/src/data_test/split_aligned_random_data.py
