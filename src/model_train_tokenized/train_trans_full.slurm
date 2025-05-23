#!/bin/bash
#SBATCH --job-name=aug_train
#SBATCH --account=stf
#SBATCH --partition=ckpt            # ✅ Stay on ckpt partition
#SBATCH --gres=gpu:1                # ✅ Request 1 GPU
#SBATCH --qos=ckpt-gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --output=/gscratch/stf/mx727/multilingual-social-summary/logs/aug_full_train_%j.out
#SBATCH --error=/gscratch/stf/mx727/multilingual-social-summary/logs/aug_full_train_%j.err
#SBATCH --requeue

# === Activate your virtual environment ===
source /gscratch/stf/mx727/multilingual-social-summary/socialsum-venv/bin/activate
echo "VIRTUAL_ENV is: $VIRTUAL_ENV"
which python
python -c "import sys; print(sys.prefix)"
python -c "import torch; print('torch version:', torch.__version__); print('cuda:', torch.cuda.is_available())"


# === Run training ===
python3 /Users/apple/Desktop/573_0522/multilingual-social-summary/src/model_train_tokenized/train_trans_full.py \
  --tokenized_path /gscratch/stf/mx727/multilingual-social-summary/data/splits/tokenized_train/trans_full \
  --output_dir /gscratch/stf/mx727/multilingual-social-summary/checkpoints/trans_full \
  --batch_size 4 \
  --grad_accum_steps 4 \
  --num_epochs 2 \
  --log_steps 100 \
  --num_workers 4 
