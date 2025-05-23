#!/bin/bash
#SBATCH --job-name=gen_indices
#SBATCH --account=stf               # Your account name
#SBATCH --partition=ckpt            # Partition (e.g., ckpt or compute)
#SBATCH --time=00:10:00             # Time limit (10 minutes)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G                    # Adjust as needed
#SBATCH --output=logs/gen_indices_%j.out
#SBATCH --error=logs/gen_indices_%j.err

# Load your environment
source /mmfs1/home/mx727/.bashrc
conda activate socialsum-venv  # Or source your venv directly

# Run the script
python /gscratch/stf/mx727/multilingual-social-summary/src/data_split/generate_indices.py \
  --tldr_path /gscratch/stf/mx727/multilingual-social-summary/data/corpus-webis-tldr-17.json \
  --translated_sent_path /gscratch/stf/mx727/multilingual-social-summary/data/combined_translated_sentences.jsonl \
  --translated_20_path /gscratch/stf/mx727/multilingual-social-summary/data/translate_20_reordered.jsonl \
  --translated_nouns_path /gscratch/stf/mx727/multilingual-social-summary/data/translate_nouns_reordered.jsonl \
  --output_dir /gscratch/stf/mx727/multilingual-social-summary/data
