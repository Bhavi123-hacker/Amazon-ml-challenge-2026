"""Task 3: Full-Population 5-Mode Error Audit on Final Post-K=12 Matching Results.

Evaluates every matched pair in output/matching_results.tsv across France, US, and India,
quantifying the exact distribution across all 5 failure modes:
  Mode 1: Co-located distinct businesses (same address, different names)
  Mode 2: Suffix/entity-type inflation (generic stopwords inflating token similarity)
  Mode 3: Chain/franchise over-merging across locations (same brand, different cities)
  Mode 4: Street-level address mismatch (same building number, different street)
  Mode 5: Low mutual token overlap / spurious match
"""

import json
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))

import pandas as pd

from src.audit_heuristics import evaluate_pair_audit
from src.reoptimize_thresholds import P_FALSE_GIVEN_CLEAN, P_FALSE_GIVEN_FLAGGED


def audit_final_results():
    print("=== Task 3: Full-Population 5-Mode Error Audit (Post-K=12) ===", flush=True)
    t_start = time.time()

    matching_file = "output/matching_results.tsv"
    test_dir = "student_resource/dataset/test"
    temp_dir = "output/temp_work"

    # Step 1: Fast Vectorized Load of S1 info (name, addr, country)
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    print(f"Loading Source 1 records from {s1_path} ...", flush=True)
    t0 = time.time()
    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False,
                        usecols=["entity_id", "business_name", "business_address", "country"])
    s1_info = {
        eid: {"name": n, "addr": a, "country": c}
        for eid, n, a, c in zip(df_s1["entity_id"], df_s1["business_name"], df_s1["business_address"], df_s1["country"])
    }
    del df_s1
    print(f"Loaded {len(s1_info):,} S1 entities in {time.time()-t0:.2f}s.", flush=True)

    # Step 2: Read matching_results.tsv and group matches by country
    print(f"Reading final matches from {matching_file} ...", flush=True)
    country_pairs = {"France": [], "US": [], "India": []}
    
    total_entities = 0
    matched_entities = 0
    
    with open(matching_file, "r", encoding="utf-8") as f:
        header = next(f)
        for line in f:
            total_entities += 1
            parts = line.rstrip("\n").split("\t")
            s1_id = parts[0]
            m_str = parts[1] if len(parts) > 1 else ""
            if m_str.strip():
                matched_entities += 1
                c = s1_info.get(s1_id, {}).get("country", "")
                if c in country_pairs:
                    for cid in m_str.split(","):
                        cid = cid.strip()
                        if cid:
                            country_pairs[c].append((s1_id, cid))

    total_pairs = sum(len(v) for v in country_pairs.values())
    print(f"Total S1 entities: {total_entities:,} | Matched entities: {matched_entities:,} ({matched_entities/total_entities*100:.2f}%)")
    print(f"Total matched pairs across all countries: {total_pairs:,}")
    for c, pairs in country_pairs.items():
        print(f"  - {c}: {len(pairs):,} matched pairs")

    audit_summary = {}

    # Step 3: Audit each country
    for country in ["France", "US", "India"]:
        pairs = country_pairs[country]
        print(f"\nAuditing {country} ({len(pairs):,} pairs) ...", flush=True)
        t_country = time.time()

        # Load candidate records for this country (fast vectorized)
        cands_path = os.path.join(temp_dir, f"cands_{country}.tsv")
        print(f"  Loading candidates from {cands_path} ...", flush=True)
        t_c = time.time()
        df_cand = pd.read_csv(cands_path, sep="\t", dtype=str, keep_default_na=False,
                              usecols=["entity_id", "business_name", "business_address"])
        cand_map = {
            eid: {"name": n, "addr": a}
            for eid, n, a in zip(df_cand["entity_id"], df_cand["business_name"], df_cand["business_address"])
        }
        del df_cand
        print(f"  Loaded {len(cand_map):,} candidates in {time.time()-t_c:.2f}s. Beginning heuristic scan...", flush=True)

        n_pairs = len(pairs)
        n_flagged = 0
        mode1_cnt = 0
        mode2_cnt = 0
        mode3_cnt = 0
        mode4_cnt = 0
        mode5_cnt = 0

        t_scan = time.time()
        for idx, (s1_id, cid) in enumerate(pairs):
            s1_rec = s1_info.get(s1_id)
            c_rec = cand_map.get(cid)
            if not s1_rec or not c_rec:
                continue

            h = evaluate_pair_audit(s1_rec["name"], s1_rec["addr"], c_rec["name"], c_rec["addr"])
            if h["expanded_flagged"]:
                n_flagged += 1
            if h["mode1_colocation"]: mode1_cnt += 1
            if h["mode2_suffix_inflation"]: mode2_cnt += 1
            if h["mode3_franchise"]: mode3_cnt += 1
            if h["mode4_street_mismatch"]: mode4_cnt += 1
            if h["mode5_low_mutual"]: mode5_cnt += 1

            if (idx + 1) % 500000 == 0 or (idx + 1) == n_pairs:
                scan_rate = (idx + 1) / (time.time() - t_scan)
                print(f"  Processed {idx+1:,}/{n_pairs:,} ({(idx+1)/n_pairs*100:.1f}%) | Flagged: {n_flagged:,} ({n_flagged/(idx+1)*100:.2f}%) | Rate: {scan_rate:,.0f} pairs/s", flush=True)

        scan_time = time.time() - t_scan
        pct_flagged = n_flagged / n_pairs if n_pairs > 0 else 0.0
        n_clean = n_pairs - n_flagged
        pct_clean = n_clean / n_pairs if n_pairs > 0 else 0.0

        # Calibrated precision via Law of Total Probability
        p_f_flag = P_FALSE_GIVEN_FLAGGED[country]
        p_f_clean = P_FALSE_GIVEN_CLEAN[country]
        pop_false_s = pct_flagged * p_f_flag["strict"] + pct_clean * p_f_clean["strict"]
        pop_prec_s = 1.0 - pop_false_s
        pop_false_c = pct_flagged * p_f_flag["conserv"] + pct_clean * p_f_clean["conserv"]
        pop_prec_c = 1.0 - pop_false_c

        country_data = {
            "country": country,
            "total_pairs": n_pairs,
            "flagged_pairs": n_flagged,
            "clean_pairs": n_clean,
            "flagged_pct": pct_flagged,
            "clean_pct": pct_clean,
            "mode1_colocation": mode1_cnt,
            "mode1_pct_of_flagged": (mode1_cnt / n_flagged * 100.0) if n_flagged > 0 else 0.0,
            "mode2_suffix_inflation": mode2_cnt,
            "mode2_pct_of_flagged": (mode2_cnt / n_flagged * 100.0) if n_flagged > 0 else 0.0,
            "mode3_franchise": mode3_cnt,
            "mode3_pct_of_flagged": (mode3_cnt / n_flagged * 100.0) if n_flagged > 0 else 0.0,
            "mode4_street_mismatch": mode4_cnt,
            "mode4_pct_of_flagged": (mode4_cnt / n_flagged * 100.0) if n_flagged > 0 else 0.0,
            "mode5_low_mutual": mode5_cnt,
            "mode5_pct_of_flagged": (mode5_cnt / n_flagged * 100.0) if n_flagged > 0 else 0.0,
            "calibrated_precision_strict": pop_prec_s,
            "calibrated_precision_conserv": pop_prec_c,
            "scan_time_sec": scan_time,
        }
        audit_summary[country] = country_data
        print(f"  {country} completed in {scan_time:.1f}s. Calibrated Precision: Strict={pop_prec_s*100:.2f}%, Conserv={pop_prec_c*100:.2f}%", flush=True)

        # Free candidate memory
        del cand_map

    # Free S1 info
    del s1_info

    # Print Summary Table
    print("\n" + "=" * 125)
    print("TASK 3: FULL-POPULATION 5-MODE ERROR BREAKDOWN ACROSS FINAL POST-K=12 MATCHES")
    print("=" * 125)
    print(f"{'Country':<8} | {'Total Pairs':<12} | {'Flagged (%)':<15} | {'Mode 1 (Co-Loc)':<17} | {'Mode 2 (Suffix)':<17} | {'Mode 3 (Franchise)':<18} | {'Mode 4 (Street)':<16} | {'Mode 5 (Low Mut)':<16}")
    print("-" * 125)
    for c in ["France", "US", "India"]:
        d = audit_summary[c]
        print(
            f"{c:<8} | {d['total_pairs']:<12,} | {d['flagged_pairs']:,} ({d['flagged_pct']*100:.1f}%) | "
            f"{d['mode1_colocation']:,} ({d['mode1_pct_of_flagged']:.1f}%) | "
            f"{d['mode2_suffix_inflation']:,} ({d['mode2_pct_of_flagged']:.1f}%) | "
            f"{d['mode3_franchise']:,} ({d['mode3_pct_of_flagged']:.1f}%) | "
            f"{d['mode4_street_mismatch']:,} ({d['mode4_pct_of_flagged']:.1f}%) | "
            f"{d['mode5_low_mutual']:,} ({d['mode5_pct_of_flagged']:.1f}%)"
        )
    print("=" * 125)

    with open("models/post_k12_error_mode_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)
    print(f"\nSaved audit summary to models/post_k12_error_mode_audit.json in {time.time()-t_start:.1f}s total.")


if __name__ == "__main__":
    audit_final_results()
