"""
eval_summary.py

This script summarizes evaluation results for multiple models using BERTScore and LaSE metrics.

It searches the current directory for files ending in:
1. *_tldr_bertscore.txt: containing BERTScore output
2. *_cs_LaSE.txt: containing LaSE output

Each file is expected to contain a line starting with either "BERTScore:" or "LaSE_scores:", followed by a list of scores.

The script computes the mean and 95% confidence interval of each score list and generates a summary table matching model prefixes (e.g., baseline, noun, sent, full) across both metrics.

Output:
- A file named `eval_summary.txt` with a row per model and columns for BERTScore and LaSE (including confidence intervals).

Example usage:
    python eval_summary.py

Note:
- Evaluation output files must be located in the same directory as this script.
"""

import os
import ast
import numpy as np
from scipy import stats

bertscore_files = [f for f in os.listdir('.') if f.endswith('_tldr_bertscore.txt')]
lase_files = [f for f in os.listdir('.') if f.endswith('_cs_LaSE.txt')]

# for file in bertscore_files + lase_files:
#     with open(file) as f:
#         for line in f:
#             if line.startswith("LaSE_scores:") or line.startswith("BERTScore:"):
#                 print(file)
#                 scores = ast.literal_eval(line.split(":", 1)[1].strip())
                
#                 # Compute average
#                 mean_score = np.mean(scores)
                
#                 # Compute 95% confidence interval
#                 confidence = 0.95
#                 n = len(scores)
#                 stderr = stats.sem(scores)  # standard error of the mean
#                 interval = stats.t.interval(confidence, df=n-1, loc=mean_score, scale=stderr)      
                          
#                 print(f"Average score: {mean_score:.4f}")
#                 print(f"95% confidence interval: ({interval[0]:.4f}, {interval[1]:.4f})\n")

results = []

for file in bertscore_files + lase_files:
    with open(file) as f:
        for line in f:
            if line.startswith("LaSE_scores:") or line.startswith("BERTScore:"):
                scores = ast.literal_eval(line.split(":", 1)[1].strip())
                
                # Compute average
                mean_score = np.mean(scores)
                
                # Compute 95% confidence interval
                confidence = 0.95
                n = len(scores)
                stderr = stats.sem(scores)  # standard error of the mean
                interval = stats.t.interval(confidence, df=n-1, loc=mean_score, scale=stderr)
                
                results.append({
                    'file': file,
                    'metric': 'LaSE' if 'LaSE' in line else 'BERTScore',
                    'mean': mean_score,
                    'ci_lower': interval[0],
                    'ci_upper': interval[1]
                })

# Group results by model prefix
grouped = {}
for r in results:
    model = r['file'].split('_')[0]
    if model not in grouped:
        grouped[model] = {}
    grouped[model][r['metric']] = r

# Write organized summary
with open("eval_summary.txt", "w") as out_f:
    out_f.write("Evaluation Summary\n\n")
    out_f.write(f"{'Model':<10} {'BERTScore (Mean, CI)':<35} {'LaSE (Mean, CI)':<35}\n")
    out_f.write("="*80 + "\n")

    for model in ['baseline', 'noun', 'sent', 'full']:
        b = grouped.get(model, {}).get('BERTScore')
        l = grouped.get(model, {}).get('LaSE')
        b_str = f"{b['mean']:.4f} ({b['ci_lower']:.4f}–{b['ci_upper']:.4f})" if b else "N/A"
        l_str = f"{l['mean']:.4f} ({l['ci_lower']:.4f}–{l['ci_upper']:.4f})" if l else "N/A"
        out_f.write(f"{model:<10} {b_str:<35} {l_str:<35}\n")