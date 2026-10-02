"""Full-Population Automated Precision Audit across all 6.97M matched pairs (Steps 1, 2, 3).

Evaluates every single matched pair in France, US, and India using independent secondary heuristic checks:
1. Name-only similarity (JaroWinkler + TokenSetRatio)
2. Address-only similarity (JaroWinkler + TokenSetRatio)
3. Suspicious pattern: address >= 0.90 AND name < 0.60 (building/street collision)
4. Reverse-suspicious pattern: name >= 0.85 AND address < 0.40 AND no shared city/pin (franchise collision)
5. Distribution tabulations for flagged vs non-flagged pairs.
6. Exports 240 calibration sample pairs (40 flagged + 40 non-flagged per country) with raw fields.
"""

import os
import sys
import time
import json
import random
from collections import defaultdict
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from src.normalize import apply_normalization_df

def get_hist_bucket(val):
    if val < 0.20:
        return "<0.20"
    elif val < 0.40:
        return "0.20-0.40"
    elif val < 0.60:
        return "0.40-0.60"
    elif val < 0.80:
        return "0.60-0.80"
    elif val < 0.90:
        return "0.80-0.90"
    else:
        return ">=0.90"

BUCKETS = ["<0.20", "0.20-0.40", "0.40-0.60", "0.60-0.80", "0.80-0.90", ">=0.90"]

def audit_country(country, test_dir, temp_dir, sample_k=40, seed=42):
    random.seed(seed)
    print(f"\n=======================================================", flush=True)
    print(f"AUDITING FULL POPULATION: {country.upper()}", flush=True)
    print(f"=======================================================", flush=True)
    t_start = time.time()
    
    res_path = os.path.join(temp_dir, f"results_{country}.tsv")
    cands_path = os.path.join(temp_dir, f"cands_{country}.tsv")
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    
    # 1. Read results to get all pairs and needed candidate IDs
    print(f"Reading matched pairs from {res_path}...", flush=True)
    t0 = time.time()
    all_pairs = []
    needed_cands = set()
    needed_s1 = set()
    
    with open(res_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            s1_id = parts[0]
            if len(parts) > 2 and parts[2].strip():
                for cid in parts[2].split(","):
                    cid = cid.strip()
                    if cid:
                        all_pairs.append((s1_id, cid))
                        needed_s1.add(s1_id)
                        needed_cands.add(cid)
                        
    total_pairs = len(all_pairs)
    print(f"Read {total_pairs:,} matched pairs across {len(needed_s1):,} S1 and {len(needed_cands):,} candidates in {time.time()-t0:.2f}s", flush=True)
    
    # 2. Load and normalize S1 entities for this country
    print(f"Loading and normalizing S1 entities...", flush=True)
    t_s1 = time.time()
    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
    df_s1 = df_s1[df_s1["entity_id"].isin(needed_s1)]
    raw_s1_dict = {r["entity_id"]: r for r in df_s1.to_dict("records")}
    df_s1 = apply_normalization_df(df_s1)
    norm_s1_dict = {r["entity_id"]: r for r in df_s1.to_dict("records")}
    del df_s1
    print(f"Loaded {len(norm_s1_dict):,} S1 records in {time.time()-t_s1:.2f}s", flush=True)
    
    # 3. Load and normalize only needed candidate entities
    print(f"Loading needed candidates from {cands_path}...", flush=True)
    t_c = time.time()
    df_c_list = []
    for chunk in pd.read_csv(cands_path, sep="\t", dtype=str, keep_default_na=False, chunksize=500000):
        sub = chunk[chunk["entity_id"].isin(needed_cands)]
        if len(sub) > 0:
            df_c_list.append(sub)
    df_c = pd.concat(df_c_list, ignore_index=True)
    raw_cand_dict = {r["entity_id"]: r for r in df_c.to_dict("records")}
    df_c = apply_normalization_df(df_c)
    norm_cand_dict = {r["entity_id"]: r for r in df_c.to_dict("records")}
    del df_c, df_c_list
    print(f"Loaded and normalized {len(norm_cand_dict):,} candidates in {time.time()-t_c:.2f}s", flush=True)
    
    # 4. Full population evaluation
    print(f"Computing heuristic cross-checks across all {total_pairs:,} pairs...", flush=True)
    t_eval = time.time()
    
    suspicious_count = 0
    reverse_suspicious_count = 0
    
    # Reservoirs for calibration sampling (Step 3)
    flagged_reservoir = []
    non_flagged_reservoir = []
    
    # Distributions
    hist_name_flagged = defaultdict(int)
    hist_name_clean = defaultdict(int)
    hist_addr_flagged = defaultdict(int)
    hist_addr_clean = defaultdict(int)
    
    name_sim_sum = 0.0
    addr_sim_sum = 0.0
    
    for idx, (s1_id, cid) in enumerate(all_pairs):
        s1 = norm_s1_dict.get(s1_id)
        c = norm_cand_dict.get(cid)
        if not s1 or not c:
            continue
            
        s1_n = s1["name_norm"]
        c_n = c["name_norm"]
        s1_a = s1["addr_norm"]
        c_a = c["addr_norm"]
        
        # 1. Name-only similarity
        name_jw = JaroWinkler.similarity(s1_n, c_n)
        name_tsr = fuzz.token_set_ratio(s1_n, c_n) / 100.0
        name_sim = max(name_jw, name_tsr)
        
        # 2. Address-only similarity
        addr_jw = JaroWinkler.similarity(s1_a, c_a)
        addr_tsr = fuzz.token_set_ratio(s1_a, c_a) / 100.0
        addr_sim = max(addr_jw, addr_tsr)
        
        name_sim_sum += name_sim
        addr_sim_sum += addr_sim
        
        # 3. Suspicious flag: addr >= 0.90 AND name < 0.60
        is_suspicious = (addr_sim >= 0.90 and name_sim < 0.60)
        
        # 4. Reverse suspicious flag: name >= 0.85 AND addr < 0.40 AND no shared city/pin
        s1_pin = s1["pin_extracted"]
        c_pin = c["pin_extracted"]
        s1_city = s1["city_extracted"]
        c_city = c["city_extracted"]
        same_geo = (s1_pin and c_pin and s1_pin == c_pin) or (s1_city and c_city and s1_city == c_city)
        is_reverse_suspicious = (name_sim >= 0.85 and addr_sim < 0.40 and not same_geo)
        
        is_flagged = is_suspicious or is_reverse_suspicious
        
        if is_suspicious:
            suspicious_count += 1
        if is_reverse_suspicious:
            reverse_suspicious_count += 1
            
        b_name = get_hist_bucket(name_sim)
        b_addr = get_hist_bucket(addr_sim)
        
        if is_flagged:
            hist_name_flagged[b_name] += 1
            hist_addr_flagged[b_addr] += 1
            # Reservoir sample
            if len(flagged_reservoir) < sample_k:
                flagged_reservoir.append((s1_id, cid, name_sim, addr_sim, is_suspicious, is_reverse_suspicious))
            else:
                j = random.randint(0, suspicious_count + reverse_suspicious_count - 1)
                if j < sample_k:
                    flagged_reservoir[j] = (s1_id, cid, name_sim, addr_sim, is_suspicious, is_reverse_suspicious)
        else:
            hist_name_clean[b_name] += 1
            hist_addr_clean[b_addr] += 1
            if len(non_flagged_reservoir) < sample_k:
                non_flagged_reservoir.append((s1_id, cid, name_sim, addr_sim, False, False))
            else:
                j = random.randint(0, idx)
                if j < sample_k:
                    non_flagged_reservoir[j] = (s1_id, cid, name_sim, addr_sim, False, False)
                    
        if (idx + 1) % 500000 == 0 or (idx + 1) == total_pairs:
            speed = (idx + 1) / max(0.01, time.time() - t_eval)
            print(f"  [{country}] {idx+1:,}/{total_pairs:,} ({((idx+1)/total_pairs)*100:.1f}%) | Speed: {speed:,.1f} pairs/s", flush=True)

    elapsed_eval = time.time() - t_eval
    total_elapsed = time.time() - t_start
    total_flagged = suspicious_count + reverse_suspicious_count
    
    print(f"\n--- {country.upper()} AUDIT SUMMARY ---")
    print(f"Total matched pairs audited: {total_pairs:,}")
    print(f"Throughput: {total_pairs / elapsed_eval:,.1f} pairs/second (evaluation: {elapsed_eval:.2f}s, total: {total_elapsed:.2f}s)")
    print(f"Suspicious pairs (addr >= 0.90 & name < 0.60): {suspicious_count:,} ({suspicious_count/total_pairs:.2%})")
    print(f"Reverse-suspicious pairs (name >= 0.85 & addr < 0.40): {reverse_suspicious_count:,} ({reverse_suspicious_count/total_pairs:.2%})")
    print(f"Total flagged pairs: {total_flagged:,} ({total_flagged/total_pairs:.2%})")
    print(f"Average name similarity: {name_sim_sum / total_pairs:.4f}")
    print(f"Average address similarity: {addr_sim_sum / total_pairs:.4f}")
    
    # Build calibration sample records with raw strings
    calib_samples = []
    for s1_id, cid, n_sim, a_sim, s_flag, rs_flag in flagged_reservoir:
        s1_r = raw_s1_dict.get(s1_id, {})
        c_r = raw_cand_dict.get(cid, {})
        calib_samples.append({
            "country": country,
            "s1_id": s1_id,
            "cand_id": cid,
            "flagged": True,
            "is_suspicious": s_flag,
            "is_reverse_suspicious": rs_flag,
            "name_sim": round(n_sim, 4),
            "addr_sim": round(a_sim, 4),
            "s1_name": s1_r.get("business_name", ""),
            "s1_addr": s1_r.get("business_address", ""),
            "cand_name": c_r.get("business_name", ""),
            "cand_addr": c_r.get("business_address", ""),
        })
    for s1_id, cid, n_sim, a_sim, _, _ in non_flagged_reservoir:
        s1_r = raw_s1_dict.get(s1_id, {})
        c_r = raw_cand_dict.get(cid, {})
        calib_samples.append({
            "country": country,
            "s1_id": s1_id,
            "cand_id": cid,
            "flagged": False,
            "is_suspicious": False,
            "is_reverse_suspicious": False,
            "name_sim": round(n_sim, 4),
            "addr_sim": round(a_sim, 4),
            "s1_name": s1_r.get("business_name", ""),
            "s1_addr": s1_r.get("business_address", ""),
            "cand_name": c_r.get("business_name", ""),
            "cand_addr": c_r.get("business_address", ""),
        })
        
    return {
        "country": country,
        "total_pairs": total_pairs,
        "eval_time_s": elapsed_eval,
        "total_time_s": total_elapsed,
        "throughput_pairs_s": total_pairs / elapsed_eval,
        "suspicious_count": suspicious_count,
        "suspicious_pct": suspicious_count / total_pairs * 100,
        "reverse_suspicious_count": reverse_suspicious_count,
        "reverse_suspicious_pct": reverse_suspicious_count / total_pairs * 100,
        "total_flagged": total_flagged,
        "total_flagged_pct": total_flagged / total_pairs * 100,
        "avg_name_sim": name_sim_sum / total_pairs,
        "avg_addr_sim": addr_sim_sum / total_pairs,
        "hist_name_flagged": dict(hist_name_flagged),
        "hist_name_clean": dict(hist_name_clean),
        "hist_addr_flagged": dict(hist_addr_flagged),
        "hist_addr_clean": dict(hist_addr_clean),
        "calib_samples": calib_samples,
    }

def main():
    test_dir = "d:/amazolml/student_resource/dataset/test"
    temp_dir = "d:/amazolml/output/temp_work"
    
    t_global = time.time()
    results = {}
    all_calib_samples = []
    
    for country in ["France", "US", "India"]:
        res = audit_country(country, test_dir, temp_dir, sample_k=40)
        all_calib_samples.extend(res.pop("calib_samples"))
        results[country] = res
        
    total_wall_time = time.time() - t_global
    
    out_summary = {
        "total_wall_time_s": total_wall_time,
        "countries": results
    }
    
    with open("full_population_audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(out_summary, f, indent=2, ensure_ascii=False)
        
    with open("full_population_calibration_sample.json", "w", encoding="utf-8") as f:
        json.dump(all_calib_samples, f, indent=2, ensure_ascii=False)
        
    # Write human readable calibration file
    with open("full_population_calibration_sample.txt", "w", encoding="utf-8") as f:
        cur_c = None
        for item in all_calib_samples:
            if item["country"] != cur_c:
                cur_c = item["country"]
                f.write(f"\n=======================================================\n")
                f.write(f"CALIBRATION SAMPLE: {cur_c.upper()} (80 pairs: 40 Flagged, 40 Clean)\n")
                f.write(f"=======================================================\n\n")
            flag_str = "FLAGGED SUSPICIOUS" if item["flagged"] else "CLEAN (NON-FLAGGED)"
            f.write(f"[{item['country']} - {flag_str}] NameSim: {item['name_sim']:.4f} | AddrSim: {item['addr_sim']:.4f}\n")
            f.write(f"  S1 ({item['s1_id']}):\n")
            f.write(f"    Name: {item['s1_name']}\n")
            f.write(f"    Addr: {item['s1_addr']}\n")
            f.write(f"  CAND ({item['cand_id']}):\n")
            f.write(f"    Name: {item['cand_name']}\n")
            f.write(f"    Addr: {item['cand_addr']}\n\n")
            
    print(f"\n=======================================================")
    print(f"FULL-POPULATION AUDIT COMPLETED IN {total_wall_time:.2f}s ({total_wall_time/60:.2f} min)")
    print(f"=======================================================")
    print(f"Files saved:")
    print(f"  - full_population_audit_summary.json")
    print(f"  - full_population_calibration_sample.json (240 pairs)")
    print(f"  - full_population_calibration_sample.txt")

if __name__ == "__main__":
    main()
