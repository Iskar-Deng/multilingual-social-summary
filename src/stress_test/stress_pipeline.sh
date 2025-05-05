#!/bin/bash

# Activate virtual environment
source socialsum-venv/bin/activate

# Create logs directory and set log file name
mkdir -p logs
LOGFILE="logs/stress_pipeline_$(date +%Y%m%d_%H%M%S).log"

# Helper function to run command with timing and resource info
run_with_timer() {
    STEP_NAME=$1
    shift
    echo "===== START $STEP_NAME =====" | tee -a "$LOGFILE"
    START_TIME=$(date +%s)
    
    echo "--- GPU Status Before ---" | tee -a "$LOGFILE"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi | tee -a "$LOGFILE"
    else
        echo "nvidia-smi not found (no GPU or driver not loaded)" | tee -a "$LOGFILE"
    fi
    
    echo "--- CPU Status Before ---" | tee -a "$LOGFILE"
    top -b -n1 | head -20 | tee -a "$LOGFILE"
    
    "$@" 2>&1 | tee -a "$LOGFILE"
    
    echo "--- GPU Status After ---" | tee -a "$LOGFILE"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi | tee -a "$LOGFILE"
    else
        echo "nvidia-smi not found (no GPU or driver not loaded)" | tee -a "$LOGFILE"
    fi
    
    echo "--- CPU Status After ---" | tee -a "$LOGFILE"
    top -b -n1 | head -20 | tee -a "$LOGFILE"
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo "===== END $STEP_NAME (Duration: ${DURATION}s) =====" | tee -a "$LOGFILE"
    echo "" | tee -a "$LOGFILE"
}

# Step 1: Fine-tuning
run_with_timer "Fine-tuning MT5 on TL;DR" \
python src/model_train/train_mt5.py \
    --data_path src/stress_test/tldr_train_3000.jsonl \
    --output_dir checkpoints/mt5_st \
    --batch_size 16 \
    --grad_accum_steps 2 \
    --num_epochs 3 \
    --log_steps 100 \
    --num_workers 2 \
    --fp16

# Step 2: TL;DR Evaluation
run_with_timer "Evaluating TL;DR test set" \
python src/stress_test/evaluate_tldr.py \
    --model_dir checkpoints/mt5_st \
    --test_data src/stress_test/tldr_test_300.jsonl \
    --output_path src/stress_test/results/tldr_test_300_results.jsonl

# Step 3: CodeSwitch Evaluation
run_with_timer "Evaluating CodeSwitch test set" \
python src/stress_test/evaluate_codeswitch.py \
    --model_dir checkpoints/mt5_st \
    --test_data src/stress_test/codeswitch_test_100.jsonl \
    --output_path src/stress_test/results/codeswitch_test_100_results.jsonl \
    --target_lang en

echo "===== ALL TASKS COMPLETED =====" | tee -a "$LOGFILE"
