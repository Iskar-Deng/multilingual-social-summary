#!/bin/bash
#SBATCH --job-name=tldr_clean
#SBATCH --account=stf
#SBATCH --partition=ckpt           
#SBATCH --gres=gpu:1              
#SBATCH --qos=ckpt-gpu
#SBATCH --constraint=a100 
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=136G
#SBATCH --output=/gscratch/scrubbed/dinuoz/augmentation/fine-tune/logs/tldr_clean_train_%j.out
#SBATCH --error=/gscratch/scrubbed/dinuoz/augmentation/fine-tune/logs/tldr_clean_train_%j.err
#SBATCH --requeue

# === Activate your virtual environment ===
source /gscratch/scrubbed/dinuoz/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/scrubbed/dinuoz/loraenv
which python
python -c "import sys; print(sys.prefix)"
python -c "import torch; print('torch version:', torch.__version__); print('cuda:', torch.cuda.is_available())"


# === Run training ===
python /gscratch/scrubbed/dinuoz/augmentation/fine-tune/lora_base.py \
  --tokenized_path /gscratch/scrubbed/dinuoz/augmentation/fine-tune/tldr_clean_100k \
  --output_dir /gscratch/scrubbed/dinuoz/augmentation/fine-tune/output_tldr_clean_100k\
   --batch_size 4\
  --grad_accum_steps 4 \
  --num_epochs 2 \
  --log_steps 100 \
  --num_workers 4 