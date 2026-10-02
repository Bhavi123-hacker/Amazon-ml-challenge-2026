import os, sys
import pandas as pd
import numpy as np
from rapidfuzz import fuzz

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))
from src.evaluate import compute_macro_f05
from src.blocking import BlockingIndex
from src.normalize import apply_normalization_df

print("Loading 5,000 training ground truth entities...")
df_gt = pd.read_csv('student_resource/dataset/train/train_ground_truth.tsv', sep='\t', nrows=5000, keep_default_na=False)
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

# Precompute candidate features for fast parameter search
entity_candidates = {}
for _, r1 in df_s1_sub.iterrows():
    s1 = r1['entity_id']
    cands = b_idx.retrieve_candidates_for_entity(r1.to_dict(), top_k=30)
    cand_list = []
    for cid, b_score in cands:
        rc = s2_recs.get(cid) or s3_recs.get(cid)
        if not rc: continue
        sn = fuzz.token_set_ratio(r1['name_norm'], rc['business_name'].lower())
        sa = fuzz.token_set_ratio(r1['addr_norm'], rc['business_address'].lower())
        
        # State & City check
        c1 = r1.get('city_extracted', '')
        c2 = rc.get('city_extracted', '')
        city_conflict = bool(c1 and c2 and c1 != c2 and fuzz.ratio(c1, c2) < 70)
        
        p1 = r1.get('pin_extracted', '')
        p2 = rc.get('pin_extracted', '')
        pin_conflict = bool(p1 and p2 and p1 != p2 and p1[:3] != p2[:3])
        
        cand_list.append({
            'cid': cid,
            'sn': sn,
            'sa': sa,
            'city_conflict': city_conflict,
            'pin_conflict': pin_conflict,
            'empty_addr': bool(not rc['business_address'] or not r1['business_address']),
            'empty_name': bool(not rc['business_name'] or not r1['business_name']),
        })
    entity_candidates[s1] = cand_list

print(f"Precomputed features for {len(entity_candidates)} entities.")

best_f05 = 0.0
best_params = None

# Grid search optimal matching rules
for sn_dual in [65, 70, 75]:
    for sa_dual in [55, 60, 65]:
        for sa_alias in [72, 75, 78, 80]:
            for sn_solo in [82, 85, 88]:
                for comp_thresh in [98, 100, 102, 105]:
                    preds = {}
                    for s1, cands in entity_candidates.items():
                        matched = []
                        for c in cands:
                            conflict = c['city_conflict'] or c['pin_conflict']
                            sn = c['sn']
                            sa = c['sa']
                            score = max(sn, sa) + 0.3 * min(sn, sa)
                            
                            is_match = False
                            # Case 1: Dual match
                            if sn >= sn_dual and sa >= sa_dual:
                                is_match = True
                            # Case 2: Exact address match (synthetic alias/DBA or transliteration)
                            elif sa >= sa_alias and not conflict:
                                is_match = True
                            # Case 3: Name match with empty address or partial
                            elif sn >= sn_solo and (c['empty_addr'] or (sa >= 40 and not conflict)):
                                is_match = True
                            # Case 4: High composite score
                            elif score >= comp_thresh and not conflict:
                                is_match = True
                                
                            if is_match:
                                matched.append(c['cid'])
                        preds[s1] = matched
                        
                    f05, p, r = compute_macro_f05(gt_map, preds)
                    if f05 > best_f05:
                        best_f05 = f05
                        best_params = (sn_dual, sa_dual, sa_alias, sn_solo, comp_thresh, p, r)
                        print(f"NEW BEST -> Macro F0.5: {f05:.4f} | Prec: {p:.4f} | Rec: {r:.4f} | params: {best_params}")

print(f"\nFinal Best F0.5: {best_f05:.4f} with params: {best_params}")
