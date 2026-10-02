import os, sys
import pandas as pd
import numpy as np
from rapidfuzz import fuzz

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))
from src.blocking import BlockingIndex
from src.normalize import apply_normalization_df

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

all_cand_list = list(s2_recs.values()) + list(s3_recs.values())
df_cand_pool = apply_normalization_df(pd.DataFrame(all_cand_list))
b_idx = BlockingIndex()
b_idx.build(df_cand_pool)

df_s1_sub = apply_normalization_df(pd.DataFrame(list(s1_recs.values())))

fps = []
for _, r1 in df_s1_sub.iterrows():
    s1 = r1['entity_id']
    true_set = gt_map.get(s1, set())
    cands = b_idx.retrieve_candidates_for_entity(r1.to_dict(), top_k=25)
    for cid, b_score in cands:
        rc = s2_recs.get(cid) or s3_recs.get(cid)
        if not rc: continue
        sn = fuzz.token_set_ratio(r1['name_norm'], rc['business_name'].lower())
        sa = fuzz.token_set_ratio(r1['addr_norm'], rc['business_address'].lower())
        
        c1 = r1.get('city_extracted', '')
        c2 = rc.get('city_extracted', '')
        city_conflict = bool(c1 and c2 and c1 != c2 and fuzz.ratio(c1, c2) < 70)
        p1 = r1.get('pin_extracted', '')
        p2 = rc.get('pin_extracted', '')
        pin_conflict = bool(p1 and p2 and p1 != p2 and p1[:3] != p2[:3])
        conflict = city_conflict or pin_conflict
        score = max(sn, sa) + 0.3 * min(sn, sa)
        
        is_match = False
        if sn >= 65 and sa >= 60 and not conflict:
            is_match = True
        elif sa >= 72 and not conflict:
            is_match = True
        elif sn >= 88 and (not rc['business_address'] or not r1['business_address'] or (sa >= 40 and not conflict)):
            is_match = True
        elif score >= 105 and not conflict:
            is_match = True
            
        if is_match and cid not in true_set:
            fps.append((r1, rc, sn, sa, score))

print(f"Total False Positives: {len(fps)}")
for r1, rc, sn, sa, score in fps[:10]:
    print(f"FP S1: {r1['entity_id']} | '{r1['business_name']}' | '{r1['business_address']}'")
    print(f"   Cand: {rc['entity_id']} | '{rc['business_name']}' | '{rc['business_address']}'")
    print(f"   sn={sn}, sa={sa}, score={score}\n")
