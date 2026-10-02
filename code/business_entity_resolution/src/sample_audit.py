"""Sample and extract raw records for consolidated test-set audit: France (60), US (40), India (40).
"""

import os
import sys
import json
import random
import pickle
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from src.normalize import apply_normalization_df
from src.features import FEATURE_COLUMNS, extract_pair_features

def main():
    random.seed(42)
    temp_dir = "d:/amazolml/output/temp_work"
    test_dir = "d:/amazolml/student_resource/dataset/test"
    
    configs = [
        ("France", os.path.join(temp_dir, "results_France.tsv"), 60, 0.9990),
        ("US", os.path.join(temp_dir, "results_US.tsv"), 40, 0.860),
        ("India", os.path.join(temp_dir, "results_India.tsv"), 40, 0.860),
    ]
    
    sampled_pairs = {}
    needed_s1 = set()
    needed_s2 = set()
    needed_s3 = set()
    
    for country, path, n_samp, tau in configs:
        pairs = []
        with open(path, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                s1_id = parts[0]
                m_str = parts[2] if len(parts) > 2 else ""
                if m_str:
                    cands = parts[1].split(",") if len(parts) > 1 and parts[1] else []
                    cand_rank = {cid: rank for rank, cid in enumerate(cands)}
                    for cid in m_str.split(","):
                        cid = cid.strip()
                        if cid:
                            pairs.append((s1_id, cid, cand_rank.get(cid, 0)))
        
        samp = random.sample(pairs, min(n_samp, len(pairs)))
        sampled_pairs[country] = samp
        for s1_id, cid, _ in samp:
            needed_s1.add(s1_id)
            if cid.startswith("S2-"):
                needed_s2.add(cid)
            elif cid.startswith("S3-"):
                needed_s3.add(cid)
        print(f"Sampled {len(samp)} pairs from {country} (total matched pairs: {len(pairs):,})")

    print(f"Total needed S1: {len(needed_s1)}, S2: {len(needed_s2)}, S3: {len(needed_s3)}")
    
    raw_records = {}
    
    def extract_from_file(filepath, needed_set):
        print(f"Scanning {filepath} for {len(needed_set)} IDs...", flush=True)
        found = 0
        with open(filepath, "r", encoding="utf-8") as f:
            header = next(f).rstrip("\n").split("\t")
            id_idx = header.index("entity_id")
            name_idx = header.index("business_name")
            addr_idx = header.index("business_address")
            country_idx = header.index("country")
            for line in f:
                parts = line.rstrip("\n").split("\t")
                eid = parts[id_idx]
                if eid in needed_set:
                    raw_records[eid] = {
                        "entity_id": eid,
                        "business_name": parts[name_idx] if len(parts) > name_idx else "",
                        "business_address": parts[addr_idx] if len(parts) > addr_idx else "",
                        "country": parts[country_idx] if len(parts) > country_idx else "",
                    }
                    found += 1
                    if found == len(needed_set):
                        break
        print(f"  Found {found}/{len(needed_set)} records.")

    extract_from_file(os.path.join(test_dir, "test_source1.tsv"), needed_s1)
    extract_from_file(os.path.join(test_dir, "test_source2.tsv"), needed_s2)
    extract_from_file(os.path.join(test_dir, "test_source3.tsv"), needed_s3)
    
    # Normalize needed records to compute exact LightGBM probability
    print("Computing model probabilities for sampled pairs...", flush=True)
    df_raw = pd.DataFrame(list(raw_records.values()))
    df_norm = apply_normalization_df(df_raw)
    norm_records = {r["entity_id"]: r for r in df_norm.to_dict("records")}
    
    with open("models/best_model.pkl", "rb") as f:
        model = pickle.load(f)
        
    audit_data = []
    
    for country, _, _, tau in configs:
        samp = sampled_pairs[country]
        for idx, (s1_id, cid, rank) in enumerate(samp, 1):
            s1_raw = raw_records.get(s1_id, {})
            c_raw = raw_records.get(cid, {})
            s1_norm = norm_records.get(s1_id, {})
            c_norm = norm_records.get(cid, {})
            
            s_name = fuzz.token_set_ratio(s1_norm.get("name_norm", ""), c_norm.get("name_norm", ""))
            s_addr = fuzz.token_set_ratio(s1_norm.get("addr_norm", ""), c_norm.get("addr_norm", ""))
            b_score = max(s_name, s_addr) + 0.3 * min(s_name, s_addr)
            feats = extract_pair_features(s1_norm, c_norm, rank=rank, blocking_score=b_score)
            feat_vec = np.array([[feats[col] for col in FEATURE_COLUMNS]], dtype=np.float32)
            prob = float(model.predict_proba(feat_vec)[0, 1])
            
            audit_data.append({
                "country": country,
                "pair_num": idx,
                "s1_id": s1_id,
                "cand_id": cid,
                "rank": rank,
                "prob": prob,
                "tau": tau,
                "s1_name": s1_raw.get("business_name", ""),
                "s1_address": s1_raw.get("business_address", ""),
                "cand_name": c_raw.get("business_name", ""),
                "cand_address": c_raw.get("business_address", ""),
            })
            
    out_json = "audit_consolidated_sample.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(audit_data)} audit pairs to {out_json}")
    
    out_txt = "audit_consolidated_sample.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        cur_country = None
        for item in audit_data:
            if item["country"] != cur_country:
                cur_country = item["country"]
                f.write(f"\n=======================================================\n")
                f.write(f"COUNTRY: {cur_country} (tau={item['tau']})\n")
                f.write(f"=======================================================\n\n")
            f.write(f"[{item['country']} #{item['pair_num']}] Prob: {item['prob']:.6f} (Rank: {item['rank']})\n")
            f.write(f"  S1 ({item['s1_id']}):\n")
            f.write(f"    Name:    {item['s1_name']}\n")
            f.write(f"    Address: {item['s1_address']}\n")
            f.write(f"  Candidate ({item['cand_id']}):\n")
            f.write(f"    Name:    {item['cand_name']}\n")
            f.write(f"    Address: {item['cand_address']}\n\n")
    print(f"Saved human-readable audit text to {out_txt}")

if __name__ == "__main__":
    main()
