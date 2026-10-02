import os, sys
import pandas as pd
import numpy as np
from rapidfuzz import fuzz

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))
from src.evaluate import compute_macro_f05
from src.blocking import BlockingIndex
from src.normalize import apply_normalization_df

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
df_cand_pool = apply_normalization_df(pd.DataFrame(all_cand_list))
b_idx = BlockingIndex()
b_idx.build(df_cand_pool)

df_s1_sub = apply_normalization_df(pd.DataFrame(list(s1_recs.values())))

for min_sort in [65, 70, 75]:
    for min_sa_alias in [72, 75, 78]:
        preds = {}
        for _, r1 in df_s1_sub.iterrows():
            s1 = r1['entity_id']
            cands = b_idx.retrieve_candidates_for_entity(r1.to_dict(), top_k=25)
            matched = []
            for cid, b_score in cands:
                rc = s2_recs.get(cid) or s3_recs.get(cid)
                if not rc: continue
                
                n1 = r1['name_norm']
                nc = rc['business_name'].lower()
                a1 = r1['addr_norm']
                ac = rc['business_address'].lower()
                
                sn_set = fuzz.token_set_ratio(n1, nc)
                sn_sort = fuzz.token_sort_ratio(n1, nc)
                sa_set = fuzz.token_set_ratio(a1, ac)
                
                c1 = r1.get('city_extracted', '')
                c2 = rc.get('city_extracted', '')
                city_conflict = bool(c1 and c2 and c1 != c2 and fuzz.ratio(c1, c2) < 70)
                
                p1 = r1.get('pin_extracted', '')
                p2 = rc.get('pin_extracted', '')
                pin_conflict = bool(p1 and p2 and p1 != p2 and p1[:3] != p2[:3])
                conflict = city_conflict or pin_conflict
                
                is_match = False
                # Case 1: Both name and address have solid match
                if sn_set >= 70 and sa_set >= 60 and not conflict:
                    is_match = True
                # Case 2: Exact address match (synthetic alias or transliteration)
                elif sa_set >= min_sa_alias and not conflict:
                    is_match = True
                # Case 3: High token_sort name match (avoids single-token subset bug)
                elif sn_sort >= min_sort and (not ac or not a1 or (sa_set >= 45 and not conflict)):
                    is_match = True
                # Case 4: High name + address composite with token_sort protection
                elif sn_sort >= 60 and sa_set >= 50 and (max(sn_set, sa_set) + 0.3 * min(sn_set, sa_set)) >= 105 and not conflict:
                    is_match = True
                    
                if is_match:
                    matched.append(cid)
            preds[s1] = matched

        f05, p, r = compute_macro_f05(gt_map, preds)
        print(f"min_sort={min_sort}, min_sa={min_sa_alias} -> Macro F0.5: {f05:.4f} | Prec: {p:.4f} | Rec: {r:.4f}")
