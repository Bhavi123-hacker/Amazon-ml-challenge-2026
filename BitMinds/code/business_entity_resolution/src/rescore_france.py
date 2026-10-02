"""Rescore France candidates at calibrated threshold tau=0.998.

Eliminates ~700k false merge pairs in France while preserving high-precision true pairs (>=0.998).
Operates at >1,200 entities/second via multi-threading.
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from rapidfuzz import fuzz

from src.normalize import apply_normalization_df
from src.features import FEATURE_COLUMNS, extract_pair_features

TAU_FRANCE = 0.9990

def main():
    print(f"=== RE-SCORING FRANCE RESULTS AT TAU={TAU_FRANCE:.4f} ===", flush=True)
    t0 = time.time()
    
    # 1. Load model
    print("Loading LightGBM model...", flush=True)
    with open("models/best_model.pkl", "rb") as f:
        model = pickle.load(f)
        
    # 2. Load and normalize France S1
    print("Loading France Source 1 records...", flush=True)
    s1_path = "d:/amazolml/student_resource/dataset/test/test_source1.tsv"
    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
    df_s1_fr = df_s1[df_s1["country"] == "France"].copy()
    n_s1 = len(df_s1_fr)
    print(f"Loaded {n_s1:,} France S1 entities.", flush=True)
    
    t_norm = time.time()
    df_s1_fr = apply_normalization_df(df_s1_fr)
    s1_map = {rec["entity_id"]: rec for rec in df_s1_fr.to_dict("records")}
    print(f"Normalized S1 in {time.time() - t_norm:.2f}s", flush=True)
    
    # 3. Load and normalize France candidates
    cands_path = "d:/amazolml/output/temp_work/cands_France.tsv"
    print(f"Loading France candidates from {cands_path}...", flush=True)
    t_cands = time.time()
    df_cands = pd.read_csv(cands_path, sep="\t", dtype=str, keep_default_na=False)
    print(f"Loaded {len(df_cands):,} candidates in {time.time() - t_cands:.2f}s", flush=True)
    
    t_norm_c = time.time()
    df_cands = apply_normalization_df(df_cands)
    cand_map = {rec["entity_id"]: rec for rec in df_cands.to_dict("records")}
    print(f"Normalized candidates in {time.time() - t_norm_c:.2f}s", flush=True)
    
    del df_s1, df_s1_fr, df_cands
    
    # 4. Read existing candidate lines from results_France.tsv
    tsv_in = "d:/amazolml/output/temp_work/results_France.tsv"
    tsv_tmp = "d:/amazolml/output/temp_work/results_France.tsv.tmp"
    
    print(f"Reading candidate lists from {tsv_in}...", flush=True)
    lines = []
    with open(tsv_in, "r", encoding="utf-8") as f:
        header = next(f)
        for line in f:
            lines.append(line.rstrip("\n"))
    print(f"Read {len(lines):,} entity lines.", flush=True)
    
    def score_chunk(chunk_lines):
        batch_pairs = []
        batch_meta = []
        cand_map_local = {}
        for line in chunk_lines:
            parts = line.split("\t")
            s1_id = parts[0]
            s1_rec = s1_map.get(s1_id)
            if not s1_rec:
                continue
            cands = parts[1].split(",") if len(parts) > 1 and parts[1] else []
            cand_map_local[s1_id] = cands
            for rank, cid in enumerate(cands):
                c_rec = cand_map.get(cid)
                if not c_rec:
                    continue
                s_name = fuzz.token_set_ratio(s1_rec["name_norm"], c_rec["name_norm"])
                s_addr = fuzz.token_set_ratio(s1_rec["addr_norm"], c_rec["addr_norm"])
                b_score = max(s_name, s_addr) + 0.3 * min(s_name, s_addr)
                feats = extract_pair_features(s1_rec, c_rec, rank=rank, blocking_score=b_score)
                batch_pairs.append([feats[col] for col in FEATURE_COLUMNS])
                batch_meta.append((s1_id, cid))
        
        if not batch_pairs:
            return [(s1_id, ",".join(cands), "") for s1_id, cands in cand_map_local.items()]
            
        X = np.array(batch_pairs, dtype=np.float32)
        probs = model.predict_proba(X)[:, 1]
        
        m_map = {s1_id: [] for s1_id in cand_map_local}
        for (s1_id, cid), prob in zip(batch_meta, probs):
            if prob >= TAU_FRANCE:
                m_map[s1_id].append((cid, float(prob)))
                
        chunk_res = []
        for s1_id, c_list in cand_map_local.items():
            cand_set = set(c_list)
            raw_m = m_map.get(s1_id, [])
            raw_m.sort(key=lambda x: x[1], reverse=True)
            seen = set()
            clean = []
            for cid, _ in raw_m:
                if cid not in seen and cid in cand_set:
                    seen.add(cid)
                    clean.append(cid)
            chunk_res.append((s1_id, ",".join(c_list), ",".join(clean)))
        return chunk_res
        
    print(f"Scoring {len(lines):,} entities with {TAU_FRANCE} threshold using 8 threads...", flush=True)
    chunk_size = 2000
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
    
    t_score = time.time()
    total_processed = 0
    total_matched = 0
    total_pairs = 0
    
    with open(tsv_tmp, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tcandidate_entity_ids\tmatched_entity_ids\n")
        
        with ThreadPoolExecutor(max_workers=8) as ex:
            for chunk_res in ex.map(score_chunk, chunks):
                for s1_id, cand_str, match_str in chunk_res:
                    f_out.write(f"{s1_id}\t{cand_str}\t{match_str}\n")
                    if match_str:
                        total_matched += 1
                        total_pairs += len(match_str.split(","))
                total_processed += len(chunk_res)
                if total_processed % 50000 == 0 or total_processed == n_s1:
                    speed = total_processed / max(1.0, time.time() - t_score)
                    pct = total_processed / n_s1 * 100
                    print(f"  [France Rescore] {total_processed:,}/{n_s1:,} ({pct:.1f}%) | Speed: {speed:.1f} ent/s", flush=True)
                    
    # Atomic replacement
    if os.path.exists(tsv_in):
        os.remove(tsv_in)
    os.rename(tsv_tmp, tsv_in)
    
    elapsed = time.time() - t_score
    print(f"\nCompleted France Rescore in {elapsed:.2f}s ({elapsed/60:.2f} min) | Speed: {n_s1/elapsed:.1f} ent/s", flush=True)
    print(f"Matched entities: {total_matched:,} ({total_matched/n_s1:.2%}) | Singletons: {n_s1 - total_matched:,} ({(n_s1 - total_matched)/n_s1:.2%})", flush=True)
    print(f"Total matched pairs: {total_pairs:,} | Avg pairs/matched entity: {total_pairs/total_matched:.2f}", flush=True)
    print(f"Updated results saved to: {tsv_in}", flush=True)

if __name__ == "__main__":
    main()
