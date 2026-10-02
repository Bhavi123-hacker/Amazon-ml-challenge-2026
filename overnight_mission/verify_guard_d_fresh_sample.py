"""Step 1: Fresh Independent Verification of Guard D on France Predictions.

Draws a fresh random sample of 60 pairs using a completely new seed (seed=987654).
Audits each pair against RAW unnormalized fields.
Reports real strict, effective, and false match precision.
"""

import os
import sys
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

def main():
    print("=" * 90)
    print("STEP 1: INDEPENDENT VERIFICATION OF GUARD D ON FRESH SAMPLE (SEED=987654)")
    print("=" * 90)

    france_res_path = "output/temp_work/champion_results_France_guard_d.tsv"
    if not os.path.exists(france_res_path):
        print(f"Error: {france_res_path} does not exist yet.", flush=True)
        return

    # 1. Load Raw S1 records
    print("Loading raw France S1 records...", flush=True)
    df_s1 = pd.read_csv("student_resource/dataset/test/test_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    df_s1_fr = df_s1[df_s1["country"] == "France"]
    s1_raw_map = {r["entity_id"]: r for r in df_s1_fr.to_dict("records")}
    del df_s1, df_s1_fr

    # 2. Load Raw Candidate records
    print("Loading raw France candidate records...", flush=True)
    df_cands = pd.read_csv("output/temp_work/cands_France.tsv", sep="\t", dtype=str, keep_default_na=False)
    cand_raw_map = {r["entity_id"]: r for r in df_cands.to_dict("records")}
    del df_cands

    # 3. Read all accepted pairs from champion_results_France_guard_d.tsv
    print("Reading accepted pairs from Guard D output...", flush=True)
    all_pairs = []
    with open(france_res_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            sid = parts[0]
            if len(parts) > 2 and parts[2].strip():
                matches = [m.strip() for m in parts[2].split(",") if m.strip()]
                for m in matches:
                    all_pairs.append((sid, m))

    print(f"Total accepted match pairs in full France test set: {len(all_pairs):,}", flush=True)

    # 4. Fresh Random Sample of 60 pairs
    np.random.seed(987654)
    sampled_indices = np.random.choice(len(all_pairs), size=60, replace=False)
    sample_pairs = [all_pairs[i] for i in sampled_indices]

    strict_matches = 0
    effective_matches = 0
    false_matches = 0

    print("\n" + "=" * 90)
    print("AUDIT OF 60 FRESH PAIRS AGAINST RAW UNNORMALIZED FIELDS")
    print("=" * 90)

    for idx, (sid, cid) in enumerate(sample_pairs, 1):
        r1 = s1_raw_map.get(sid, {})
        rc = cand_raw_map.get(cid, {})

        s1_name = r1.get("business_name", "")
        s1_addr = r1.get("business_address", "")
        c_name = rc.get("business_name", "")
        c_addr = rc.get("business_address", "")

        n_sim = fuzz.token_sort_ratio(s1_name.lower(), c_name.lower())
        a_sim = fuzz.token_sort_ratio(s1_addr.lower(), c_addr.lower())

        if n_sim >= 70 or (n_sim >= 50 and a_sim >= 60):
            classification = "GENUINE (Strict Match)"
            strict_matches += 1
            effective_matches += 1
        elif n_sim >= 40 and a_sim >= 75:
            classification = "AMBIGUOUS (Effective/Alias)"
            effective_matches += 1
        else:
            classification = "FALSE MATCH"
            false_matches += 1

        print(f"[{idx:>2}/60] {classification:<26} | N_Sim: {n_sim:>3.0f}% | A_Sim: {a_sim:>3.0f}%")
        print(f"     S1   : [{sid}] '{s1_name}' | Addr: '{s1_addr}'")
        print(f"     Cand : [{cid}] '{c_name}' | Addr: '{c_addr}'")
        print("-" * 90)

    strict_p = strict_matches / len(sample_pairs) * 100
    eff_p = effective_matches / len(sample_pairs) * 100
    false_p = false_matches / len(sample_pairs) * 100

    print("\n" + "=" * 90)
    print("STEP 1: INDEPENDENT VERIFICATION RESULTS (FRESH SEED=987654)")
    print("=" * 90)
    print(f"Strict Precision   : {strict_matches}/60 ({strict_p:.1f}%)")
    print(f"Effective Precision: {effective_matches}/60 ({eff_p:.1f}%)")
    print(f"False Matches      : {false_matches}/60 ({false_p:.1f}%)")
    print("=" * 90, flush=True)

if __name__ == "__main__":
    main()
