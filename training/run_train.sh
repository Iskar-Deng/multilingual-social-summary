```bash
#!/bin/bash

echo "Training on base dataset..."
python training/train_model.py --variant base

echo "Training on full translated dataset..."
python training/train_model.py --variant full

echo "Training on sentence-level dataset..."
python training/train_model.py --variant sent

echo "Training on noun-level dataset..."
python training/train_model.py --variant noun

echo "All trainings completed."