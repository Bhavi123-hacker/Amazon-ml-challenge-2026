import os, sys, pickle
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))
from src.features import FEATURE_COLUMNS, extract_pair_features
from src.normalize import apply_normalization_df
from rapidfuzz import fuzz

# Load model
with open("models/best_model_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

# Load ground truth sample
df_gt = pd.read_csv('student_resource/dataset/train/train_ground_truth.tsv', sep='\t', nrows=200, keep_default_na=False)

s1_needed = set()
cand_needed = set()
pairs = []

for _, r in df_gt.iterrows():
    s1 = r['source1_entity_id'].strip()
    m_str = r['matched_entity_ids'].strip()
    if not m_str: continue
    s1_needed.add(s1)
    for cid in m_str.split(','):
        cid = cid.strip()
        cand_needed.add(cid)
        pairs.append((s1, cid))

print(f"Loaded {len(pairs)} true match pairs.")

# Load records
def load_subset(path, ids):
    recs = {}
    for chunk in pd.read_csv(path, sep='\t', chunksize=50000, keep_default_na=False):
        sub = chunk[chunk['entity_id'].isin(ids)]
        for _, row in sub.iterrows():
            recs[row['entity_id']] = row.to_dict()
        if len(recs) == len(ids): break
    return recs

s1_recs = load_subset('student_resource/dataset/train/train_source1.tsv', s1_needed)
s2_recs = load_subset('student_resource/dataset/train/train_source2.tsv', cand_needed)
s3_recs = load_subset('student_resource/dataset/train/train_source3.tsv', cand_needed)

# Normalize
df_s1 = apply_normalization_df(pd.DataFrame(list(s1_recs.values()))).set_index('entity_id')
cand_all = list(s2_recs.values()) + list(s3_recs.values())
df_cand = apply_normalization_df(pd.DataFrame(cand_all)).set_index('entity_id')

feat_rows = []
valid_pairs = []
for s1, cid in pairs:
    if s1 in df_s1.index and cid in df_cand.index:
        r1 = df_s1.loc[s1].to_dict()
        rc = df_cand.loc[cid].to_dict()
        s_name = fuzz.token_set_ratio(r1["name_norm"], rc["name_norm"])
        s_addr = fuzz.token_set_ratio(r1["addr_norm"], rc["addr_norm"])
        b_score = max(s_name, s_addr) + 0.3 * min(s_name, s_addr)
        f = extract_pair_features(r1, rc, rank=0, blocking_score=b_score)
        feat_rows.append([f[c] for c in FEATURE_COLUMNS])
        valid_pairs.append((s1, cid))

X = np.array(feat_rows, dtype=np.float32)
probs = pipeline.predict_proba(X)[:, 1]

print(f"\nTested on {len(valid_pairs)} TRUE ground truth matches:")
print(f"Mean prob: {np.mean(probs):.4f}")
print(f"Median prob: {np.median(probs):.4f}")
print(f"Min prob: {np.min(probs):.4f}")
print(f"Max prob: {np.max(probs):.4f}")
print(f"Prob >= 0.50: {np.mean(probs >= 0.50)*100:.1f}%")
print(f"Prob >= 0.80: {np.mean(probs >= 0.80)*100:.1f}%")
print(f"Prob >= 0.90: {np.mean(probs >= 0.90)*100:.1f}% (Our US/India threshold!)")
print(f"Prob >= 0.95: {np.mean(probs >= 0.95)*100:.1f}%")
print(f"Prob >= 0.995: {np.mean(probs >= 0.995)*100:.1f}% (Our France threshold!)")
