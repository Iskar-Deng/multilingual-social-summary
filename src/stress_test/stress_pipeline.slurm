#!/bin/bash
#SBATCH --job-name=stress_train_mt5
#SBATCH --account=stf
#SBATCH --partition=ckpt            # ✅ Stay on ckpt partition
#SBATCH --gres=gpu:1                # ✅ Request 1 GPU
#SBATCH --qos=ckpt-gpu
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/gscratch/stf/mx727/logs/stress_%j.out
#SBATCH --error=/gscratch/stf/mx727/logs/stress_%j.err
#SBATCH --requeue

# activate virtual env
source socialsum-venv/bin/activate

# run the pipeline script
bash src/stress_test/stress_pipeline.sh