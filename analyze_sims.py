import os, sys
import pandas as pd
import numpy as np
from rapidfuzz import fuzz

# Let's test on 1,000 training entities
print("Loading train ground truth sample...")
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

print(f"Loading records for {len(s1_set)} S1 and {len(cand_set)} matches...")
s1_recs = load_recs('student_resource/dataset/train/train_source1.tsv', s1_set)
s2_recs = load_recs('student_resource/dataset/train/train_source2.tsv', cand_set)
s3_recs = load_recs('student_resource/dataset/train/train_source3.tsv', cand_set)

# Let's inspect the similarities of true matches!
name_sims = []
addr_sims = []
max_sims = []

for s1, true_cands in gt_map.items():
    if s1 not in s1_recs: continue
    r1 = s1_recs[s1]
    n1 = r1['business_name'].lower()
    a1 = r1['business_address'].lower()
    for cid in true_cands:
        rc = s2_recs.get(cid) or s3_recs.get(cid)
        if not rc: continue
        nc = rc['business_name'].lower()
        ac = rc['business_address'].lower()
        
        sn = fuzz.token_set_ratio(n1, nc)
        sa = fuzz.token_set_ratio(a1, ac)
        name_sims.append(sn)
        addr_sims.append(sa)
        max_sims.append(max(sn, sa))

name_sims = np.array(name_sims)
addr_sims = np.array(addr_sims)
max_sims = np.array(max_sims)

print(f"\nAcross {len(name_sims)} TRUE matches:")
print(f"name_sim >= 70: {np.mean(name_sims >= 70)*100:.1f}%")
print(f"addr_sim >= 70: {np.mean(addr_sims >= 70)*100:.1f}%")
print(f"max(name_sim, addr_sim) >= 70: {np.mean(max_sims >= 70)*100:.1f}%")
print(f"max(name_sim, addr_sim) >= 80: {np.mean(max_sims >= 80)*100:.1f}%")
print(f"max(name_sim, addr_sim) >= 85: {np.mean(max_sims >= 85)*100:.1f}%")
print(f"max(name_sim, addr_sim) >= 90: {np.mean(max_sims >= 90)*100:.1f}%")
