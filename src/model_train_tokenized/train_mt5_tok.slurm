#!/bin/bash
#SBATCH --job-name=actual_train_mt5
#SBATCH --account=stf
#SBATCH --partition=ckpt            # ✅ Stay on ckpt partition
#SBATCH --gres=gpu:1                # ✅ Request 1 GPU
#SBATCH --qos=ckpt-gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --output=/gscratch/stf/mx727/multilingual-social-summary/logs/base_train_mt5_%j.out
#SBATCH --error=/gscratch/stf/mx727/multilingual-social-summary/logs/base_train_mt5_%j.err
#SBATCH --requeue

# === Activate your virtual environment ===
source /gscratch/stf/mx727/multilingual-social-summary/socialsum-venv/bin/activate
echo "VIRTUAL_ENV is: $VIRTUAL_ENV"
which python
python -c "import sys; print(sys.prefix)"
python -c "import torch; print('torch version:', torch.__version__); print('cuda:', torch.cuda.is_available())"


# === Run training ===
python3 /gscratch/stf/mx727/multilingual-social-summary/src/model_train_tokenized/train_mt5_tok.py \
  --tokenized_path /gscratch/stf/mx727/multilingual-social-summary/data/splits/tokenized_train_clean \
  --output_dir /gscratch/stf/mx727/multilingual-social-summary/checkpoints/mt5_debug \
  --batch_size 4 \
  --grad_accum_steps 4 \
  --num_epochs 2 \
  --log_steps 100 \
  --num_workers 4 
