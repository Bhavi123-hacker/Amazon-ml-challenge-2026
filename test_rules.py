import os, sys
import pandas as pd
import numpy as np
from rapidfuzz import fuzz

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))
from src.evaluate import compute_macro_f05
from src.blocking import BlockingIndex
from src.normalize import apply_normalization_df

# Let's test on 1,000 training entities
print("Loading train ground truth (1000 entities)...")
df_gt = pd.read_csv('student_resource/dataset/train/train_ground_truth.tsv', sep='\t', nrows=1000, keep_default_na=False)
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

# Let's evaluate our candidates that were in the candidate pool for these 1000 entities
# Build candidate pool for these 1000 entities
all_cand_list = list(s2_recs.values()) + list(s3_recs.values())
df_cand_pool = pd.DataFrame(all_cand_list)
df_cand_pool = apply_normalization_df(df_cand_pool)

b_idx = BlockingIndex()
b_idx.build(df_cand_pool)

df_s1_sub = pd.DataFrame(list(s1_recs.values()))
df_s1_sub = apply_normalization_df(df_s1_sub)

# Test multiple decision rules on these 1000 entities
for thresh_rule in [
    ("Current MLP at tau=0.90", "mlp_90"),
    ("Rule: name>=75 and addr>=60", "rule_1"),
    ("Rule: name>=80 or (name>=60 and addr>=75) or addr>=88", "rule_2"),
    ("Rule: score >= 110", "rule_score_110"),
    ("Rule: score >= 100", "rule_score_100"),
    ("Rule: score >= 90", "rule_score_90"),
]:
    preds = {}
    for _, r1 in df_s1_sub.iterrows():
        s1 = r1['entity_id']
        cands = b_idx.retrieve_candidates_for_entity(r1.to_dict(), top_k=25)
        matched = []
        for cid, b_score in cands:
            rc = s2_recs.get(cid) or s3_recs.get(cid)
            if not rc: continue
            sn = fuzz.token_set_ratio(r1['name_norm'], rc['business_name'].lower())
            sa = fuzz.token_set_ratio(r1['addr_norm'], rc['business_address'].lower())
            
            # Apply rule
            rule_id = thresh_rule[1]
            is_match = False
            if rule_id == "rule_1":
                if sn >= 75 and sa >= 60: is_match = True
            elif rule_id == "rule_2":
                if sn >= 80 or (sn >= 60 and sa >= 75) or sa >= 88: is_match = True
            elif rule_id == "rule_score_110":
                if b_score >= 110: is_match = True
            elif rule_id == "rule_score_100":
                if b_score >= 100: is_match = True
            elif rule_id == "rule_score_90":
                if b_score >= 90: is_match = True
            if is_match:
                matched.append(cid)
        preds[s1] = matched

    f05, p, r = compute_macro_f05(gt_map, preds)
    print(f"[{thresh_rule[0]}] -> Macro F0.5: {f05:.4f} | Prec: {p:.4f} | Rec: {r:.4f}")
