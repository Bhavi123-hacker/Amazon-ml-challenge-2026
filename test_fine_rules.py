import os, sys
import pandas as pd
import numpy as np
from rapidfuzz import fuzz

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))
from src.evaluate import compute_macro_f05
from src.blocking import BlockingIndex
from src.normalize import apply_normalization_df

print("Loading train ground truth (2000 entities)...")
df_gt = pd.read_csv('student_resource/dataset/train/train_ground_truth.tsv', sep='\t', nrows=2000, keep_default_na=False)
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
df_cand_pool = pd.DataFrame(all_cand_list)
df_cand_pool = apply_normalization_df(df_cand_pool)

b_idx = BlockingIndex()
b_idx.build(df_cand_pool)

df_s1_sub = pd.DataFrame(list(s1_recs.values()))
df_s1_sub = apply_normalization_df(df_s1_sub)

# Pre-retrieve candidates for all s1
s1_candidates = {}
for _, r1 in df_s1_sub.iterrows():
    s1 = r1['entity_id']
    cands = b_idx.retrieve_candidates_for_entity(r1.to_dict(), top_k=25)
    cand_details = []
    for cid, b_score in cands:
        rc = s2_recs.get(cid) or s3_recs.get(cid)
        if not rc: continue
        sn = fuzz.token_set_ratio(r1['name_norm'], rc['business_name'].lower())
        sa = fuzz.token_set_ratio(r1['addr_norm'], rc['business_address'].lower())
        cand_details.append((cid, b_score, sn, sa))
    s1_candidates[s1] = cand_details

# Fine sweep
for cut in [95, 98, 100, 102, 104, 105, 106, 108, 110]:
    preds = {}
    for s1, cands in s1_candidates.items():
        matched = [cid for cid, score, sn, sa in cands if score >= cut]
        preds[s1] = matched
    f05, p, r = compute_macro_f05(gt_map, preds)
    print(f"Cutoff score >= {cut:3d} -> Macro F0.5: {f05:.4f} | Prec: {p:.4f} | Rec: {r:.4f}")

# Also test smart compound rules
for name_min, addr_min, score_min in [
    (50, 50, 100),
    (60, 50, 100),
    (50, 60, 100),
    (60, 60, 100),
    (70, 70, 95),
    (75, 50, 98),
    (50, 75, 98),
]:
    preds = {}
    for s1, cands in s1_candidates.items():
        matched = []
        for cid, score, sn, sa in cands:
            # high confidence in either name or addr, plus composite score
            if score >= score_min and (sn >= name_min or sa >= addr_min):
                matched.append(cid)
        preds[s1] = matched
    f05, p, r = compute_macro_f05(gt_map, preds)
    print(f"Rule (name_min={name_min}, addr_min={addr_min}, score>={score_min}) -> Macro F0.5: {f05:.4f} | Prec: {p:.4f} | Rec: {r:.4f}")
