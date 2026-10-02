"""Benchmark heuristic precision audit on a timed slice of 5,000 matched pairs (Step 0).
"""

import time
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from src.normalize import apply_normalization_df

def main():
    print("=== BENCHMARKING HEURISTIC PRECISION AUDIT ON 5,000 PAIRS (STEP 0) ===", flush=True)
    
    # 1. Load first 2,000 entities from results_France.tsv to get ~5,000 pairs
    pairs = []
    needed_s1 = set()
    needed_cand = set()
    with open("d:/amazolml/output/temp_work/results_France.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            s1_id = parts[0]
            if len(parts) > 2 and parts[2].strip():
                for cid in parts[2].split(","):
                    pairs.append((s1_id, cid.strip()))
                    needed_s1.add(s1_id)
                    needed_cand.add(cid.strip())
                    if len(pairs) >= 5000:
                        break
            if len(pairs) >= 5000:
                break
                
    print(f"Collected {len(pairs):,} test pairs across {len(needed_s1):,} S1 and {len(needed_cand):,} Candidates.", flush=True)
    
    # 2. Load and normalize the required entities
    s1_path = "d:/amazolml/student_resource/dataset/test/test_source1.tsv"
    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
    df_s1 = df_s1[df_s1["entity_id"].isin(needed_s1)]
    df_s1 = apply_normalization_df(df_s1)
    s1_map = {r["entity_id"]: r for r in df_s1.to_dict("records")}
    
    cands_path = "d:/amazolml/output/temp_work/cands_France.tsv"
    df_cands = []
    for chunk in pd.read_csv(cands_path, sep="\t", dtype=str, keep_default_na=False, chunksize=200000):
        sub = chunk[chunk["entity_id"].isin(needed_cand)]
        if len(sub) > 0:
            df_cands.append(sub)
    df_cands = pd.concat(df_cands, ignore_index=True)
    df_cands = apply_normalization_df(df_cands)
    cand_map = {r["entity_id"]: r for r in df_cands.to_dict("records")}
    
    print(f"Entities ready. Starting timed heuristic calculation on 5,000 pairs...", flush=True)
    
    t0 = time.perf_counter()
    
    suspicious_count = 0
    reverse_suspicious_count = 0
    name_sims = []
    addr_sims = []
    
    for s1_id, cid in pairs:
        s1 = s1_map.get(s1_id)
        c = cand_map.get(cid)
        if not s1 or not c:
            continue
            
        s1_name = s1["name_norm"]
        c_name = c["name_norm"]
        s1_addr = s1["addr_norm"]
        c_addr = c["addr_norm"]
        
        # 1. Name-only similarity
        name_jw = JaroWinkler.similarity(s1_name, c_name)
        name_tsr = fuzz.token_set_ratio(s1_name, c_name) / 100.0
        name_sim = max(name_jw, name_tsr)
        name_sims.append(name_sim)
        
        # 2. Address-only similarity
        addr_jw = JaroWinkler.similarity(s1_addr, c_addr)
        addr_tsr = fuzz.token_set_ratio(s1_addr, c_addr) / 100.0
        addr_sim = max(addr_jw, addr_tsr)
        addr_sims.append(addr_sim)
        
        # 3. Suspicious pattern: address >= 0.90 AND name < 0.60
        if addr_sim >= 0.90 and name_sim < 0.60:
            suspicious_count += 1
            
        # 4. Reverse suspicious pattern: name >= 0.85 AND addr < 0.40 AND no shared city/pin
        s1_pin = s1["pin_extracted"]
        c_pin = c["pin_extracted"]
        s1_city = s1["city_extracted"]
        c_city = c["city_extracted"]
        
        same_geo = (s1_pin and c_pin and s1_pin == c_pin) or (s1_city and c_city and s1_city == c_city)
        if name_sim >= 0.85 and addr_sim < 0.40 and not same_geo:
            reverse_suspicious_count += 1
            
    elapsed = time.perf_counter() - t0
    throughput = len(pairs) / elapsed
    
    print(f"\n--- BENCHMARK RESULTS (5,000 PAIRS) ---")
    print(f"Elapsed time: {elapsed:.4f} seconds")
    print(f"Single-threaded throughput: {throughput:,.1f} pairs/second")
    print(f"Suspicious pairs flagged: {suspicious_count:,} ({suspicious_count/len(pairs):.2%})")
    print(f"Reverse-suspicious pairs flagged: {reverse_suspicious_count:,} ({reverse_suspicious_count/len(pairs):.2%})")
    print(f"Average name similarity: {sum(name_sims)/len(name_sims):.4f}")
    print(f"Average address similarity: {sum(addr_sims)/len(addr_sims):.4f}")
    
    # Linear extrapolation
    total_pairs = 6970229
    fr_pairs = 1074942
    us_pairs = 3136237
    in_pairs = 2759050
    
    print(f"\n--- LINEAR EXTRAPOLATION TO FULL POPULATION ({total_pairs:,} PAIRS) ---")
    # Single-thread estimate
    sec_1t = total_pairs / throughput
    print(f"Single-core estimated time: {sec_1t:.1f}s ({sec_1t/60:.2f} min / {sec_1t/3600:.2f} hours)")
    
    # 8-worker thread estimate (scale factor ~6.5x)
    tp_8t = throughput * 6.5
    sec_8t = total_pairs / tp_8t
    print(f"8-worker multi-threaded throughput: {tp_8t:,.1f} pairs/second")
    print(f"8-worker estimated time: {sec_8t:.1f}s ({sec_8t/60:.2f} min / {sec_8t/3600:.2f} hours)")
    print(f"  - France ({fr_pairs:,} pairs): {fr_pairs / tp_8t:.1f}s ({fr_pairs / tp_8t / 60:.2f} min)")
    print(f"  - US ({us_pairs:,} pairs):     {us_pairs / tp_8t:.1f}s ({us_pairs / tp_8t / 60:.2f} min)")
    print(f"  - India ({in_pairs:,} pairs):  {in_pairs / tp_8t:.1f}s ({in_pairs / tp_8t / 60:.2f} min)")

if __name__ == "__main__":
    main()
