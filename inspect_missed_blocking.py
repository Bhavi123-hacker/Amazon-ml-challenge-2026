import os, sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))
from src.blocking import BlockingIndex
from src.normalize import apply_normalization_df

df_gt = pd.read_csv('student_resource/dataset/train/train_ground_truth.tsv', sep='\t', nrows=500, keep_default_na=False)
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

missed_blocking = []

for _, r1 in df_s1_sub.iterrows():
    s1 = r1['entity_id']
    true_set = gt_map.get(s1, set())
    if not true_set: continue
    cands25 = set(c[0] for c in b_idx.retrieve_candidates_for_entity(r1.to_dict(), top_k=25))
    for cid in true_set:
        if cid not in cands25:
            rc = s2_recs.get(cid) or s3_recs.get(cid)
            if rc:
                missed_blocking.append((r1, rc))

print(f"Total missed in top 25 blocking: {len(missed_blocking)}")
for r1, rc in missed_blocking[:5]:
    print(f"S1: {r1['entity_id']} ({r1['country']}) | Name: '{r1['business_name']}' | Addr: '{r1['business_address']}'")
    print(f"Cand: {rc['entity_id']} | Name: '{rc['business_name']}' | Addr: '{rc['business_address']}'\n")
