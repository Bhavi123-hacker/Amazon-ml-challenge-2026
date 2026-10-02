import os, sys, time, gc
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))
from src.features import FEATURE_COLUMNS, extract_pair_features
from src.normalize import apply_normalization_df
from src.evaluate import compute_macro_f05
from src.blocking import BlockingIndex

print("Building training dataset from 10,000 ground truth entities...")
t0 = time.time()

# 1. Load ground truth
df_gt = pd.read_csv('student_resource/dataset/train/train_ground_truth.tsv', sep='\t', nrows=10000, keep_default_na=False)
gt_map = {}
s1_set = set()
cand_set = set()
for _, r in df_gt.iterrows():
    s1 = r['source1_entity_id'].strip()
    m = r['matched_entity_ids'].strip()
    s1_set.add(s1)
    if m:
        cands = [x.strip() for x in m.split(',') if x.strip()]
        gt_map[s1] = set(cands)
        for c in cands: cand_set.add(c)
    else:
        gt_map[s1] = set()

def load_recs(path, ids):
    res = {}
    for chunk in pd.read_csv(path, sep='\t', chunksize=50000, keep_default_na=False):
        sub = chunk[chunk['entity_id'].isin(ids)]
        for _, r in sub.iterrows():
            res[r['entity_id']] = r.to_dict()
        if len(res) == len(ids): break
    return res

s1_recs = load_recs('student_resource/dataset/train/train_source1.tsv', s1_set)
s2_recs = load_recs('student_resource/dataset/train/train_source2.tsv', cand_set)
s3_recs = load_recs('student_resource/dataset/train/train_source3.tsv', cand_set)

all_cand_list = list(s2_recs.values()) + list(s3_recs.values())
df_cand_pool = apply_normalization_df(pd.DataFrame(all_cand_list))
b_idx = BlockingIndex()
b_idx.build(df_cand_pool)

df_s1_sub = apply_normalization_df(pd.DataFrame(list(s1_recs.values())))

# Build feature pairs
X_list = []
y_list = []
groups = []
meta_pairs = []

for _, r1 in df_s1_sub.iterrows():
    s1 = r1['entity_id']
    true_set = gt_map.get(s1, set())
    cands = b_idx.retrieve_candidates_for_entity(r1.to_dict(), top_k=25)
    
    # Add all true matches if missed by blocking to ensure positive representation
    cand_ids = [c[0] for c in cands]
    for cid in true_set:
        if cid not in cand_ids:
            cands.append((cid, 50.0))
            
    for rank, (cid, b_score) in enumerate(cands):
        rc = s2_recs.get(cid) or s3_recs.get(cid)
        if not rc: continue
        label = 1 if cid in true_set else 0
        feats = extract_pair_features(r1.to_dict(), rc, rank=rank, blocking_score=b_score)
        X_list.append([feats[col] for col in FEATURE_COLUMNS])
        y_list.append(label)
        groups.append(s1)
        meta_pairs.append((s1, cid))

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int32)
groups = np.array(groups)

print(f"Dataset built in {time.time() - t0:.1f}s: {len(X):,} pairs ({np.mean(y)*100:.1f}% positive)")

# 5-Fold GroupKFold CV
gkf = GroupKFold(n_splits=5)
oof_probs = np.zeros(len(y))

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    clf.fit(X_train, y_train)
    oof_probs[val_idx] = clf.predict_proba(X_val)[:, 1]

# Macro F0.5 evaluation at different thresholds
print("\n--- Out-of-Fold Macro F0.5 Evaluation ---")
for tau in [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85]:
    preds = {s1: [] for s1 in set(groups)}
    for (s1, cid), prob in zip(meta_pairs, oof_probs):
        if prob >= tau:
            preds[s1].append(cid)
    f05, p, r = compute_macro_f05(gt_map, preds)
    print(f"Threshold tau={tau:.2f} -> Macro F0.5: {f05:.4f} | Prec: {p:.4f} | Rec: {r:.4f}")
