import numpy as np
import pandas as pd

# Load 50,000 ground truth rows
gt_df = pd.read_csv("student_resource/dataset/train/train_ground_truth.tsv", sep="\t", nrows=50000, keep_default_na=False)
gt_matches = []
for m in gt_df["matched_entity_ids"]:
    if m:
        gt_matches.append(set(m.split(",")))
    else:
        gt_matches.append(set())

# 1. Simulation from earlier (Perfect Recall, varying extra false positives):
def sim_f05(extra_fp_per_entity, recall=1.0):
    scores = []
    np.random.seed(42)
    for true_set in gt_matches:
        n_true = len(true_set)
        if n_true == 0:
            # Singleton: if model also predicts 0 matches, F0.5 = 1.0; if false positive, F0.5 = 0.0
            fp = np.random.poisson(extra_fp_per_entity)
            scores.append(1.0 if fp == 0 else 0.0)
            continue
        
        # True positives captured
        captured = int(np.round(n_true * recall))
        if captured == 0 and n_true > 0:
            captured = 1 if np.random.rand() < recall else 0
        
        # False positives added
        fp = np.random.poisson(extra_fp_per_entity)
        pred_len = captured + fp
        
        if pred_len == 0:
            scores.append(0.0)
            continue
        
        prec = captured / pred_len
        rec = captured / n_true
        
        denom = 0.25 * prec + rec
        if denom == 0:
            scores.append(0.0)
        else:
            f05 = (1.25 * prec * rec) / denom
            scores.append(f05)
    return np.mean(scores)

print(f"Extra FP = 2.50 (OLD Submission, 100% Rec) --> Score: {sim_f05(2.5, 1.0):.4f}")
print(f"Extra FP = 2.50 (OLD Submission, 93.6% Rec) --> Score: {sim_f05(2.5, 0.9365):.4f}  <-- EXACTLY 0.633!")
print(f"Extra FP = 0.036 (NEW Submission, 100% Rec) --> Score: {sim_f05(0.036, 1.0):.4f}")
print(f"Extra FP = 0.036 (NEW Submission, 93.6% Rec) --> Score: {sim_f05(0.036, 0.9365):.4f}")
print(f"Extra FP = 0.036 (NEW Submission, 91.0% Rec) --> Score: {sim_f05(0.036, 0.9100):.4f}")
