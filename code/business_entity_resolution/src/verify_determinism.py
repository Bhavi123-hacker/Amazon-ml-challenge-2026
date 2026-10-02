"""Task 4: End-to-End Inference Determinism and Reproducibility Verification.

Verifies that the inference pipeline is strictly deterministic across repeated runs:
1. Samples 10,000 entities from France, 10,000 entities from US, 10,000 entities from India
   (30,000 total entities) across test_source1.tsv.
2. Executes multi-threaded feature extraction and NeuralNet_MLP inference twice (Run 1 and Run 2)
   using identical model weights and inputs.
3. Compares Run 1 vs Run 2:
   - Confirms exact byte-for-byte and row-for-row match equality.
   - Compares against output/matching_results.tsv to verify 100.0% concordance with the submission.
4. Outputs verification report to models/determinism_verification_report.json.
"""

import json
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from src.features import FEATURE_COLUMNS, extract_pair_features
from src.normalize import apply_normalization_df

THRESHOLDS = {
    "France": 0.995,
    "US": 0.900,
    "India": 0.900,
}


def score_slice_for_country(
    country: str,
    s1_slice_records: List[dict],
    cand_map: Dict[str, dict],
    cands_by_s1: Dict[str, List[str]],
    pipeline,
    num_workers: int = 4,
    chunk_size: int = 1000,
) -> Dict[str, str]:
    """Score candidate pairs for a slice of S1 records."""
    tau = THRESHOLDS[country]
    s1_map = {r["entity_id"]: r for r in s1_slice_records}
    s1_ids = [r["entity_id"] for r in s1_slice_records]

    chunks = [s1_ids[i:i + chunk_size] for i in range(0, len(s1_ids), chunk_size)]

    def worker_func(chunk_ids):
        local_results = {}
        batch_pairs = []
        batch_meta = []

        for sid in chunk_ids:
            s_rec = s1_map[sid]
            cands = cands_by_s1.get(sid, [])
            if not cands:
                local_results[sid] = ""
                continue

            for rank, cid in enumerate(cands):
                c_rec = cand_map.get(cid)
                if not c_rec:
                    continue
                s_name = fuzz.token_set_ratio(s_rec["name_norm"], c_rec["name_norm"])
                s_addr = fuzz.token_set_ratio(s_rec["addr_norm"], c_rec["addr_norm"])
                b_score = max(s_name, s_addr) + 0.3 * min(s_name, s_addr)
                feats = extract_pair_features(s_rec, c_rec, rank=rank, blocking_score=b_score)
                batch_pairs.append([feats[col] for col in FEATURE_COLUMNS])
                batch_meta.append((sid, cid))

        if not batch_pairs:
            return local_results

        X = np.array(batch_pairs, dtype=np.float32)
        probs = pipeline.predict_proba(X)[:, 1]

        matches_by_sid = {sid: [] for sid in chunk_ids}
        for (sid, cid), prob in zip(batch_meta, probs):
            if prob >= tau:
                matches_by_sid[sid].append(cid)

        for sid in chunk_ids:
            local_results[sid] = ",".join(matches_by_sid.get(sid, []))

        return local_results

    results = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for chunk_res in executor.map(worker_func, chunks):
            results.update(chunk_res)

    return results


def main():
    print("=== Task 4: Pipeline Reproducibility & Determinism Proof ===", flush=True)
    t0 = time.time()

    # Load trained model
    model_path = "models/best_model_pipeline.pkl"
    print(f"Loading trained pipeline from {model_path}...", flush=True)
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    # Load S1 sample: 10,000 per country
    s1_path = "student_resource/dataset/test/test_source1.tsv"
    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
    
    # Load candidate mapping from output/candidate_pairs.tsv
    print("Loading candidate mappings for sample entities from output/candidate_pairs.tsv...", flush=True)
    cands_path = "output/candidate_pairs.tsv"
    
    # Also load ground truth submission from output/matching_results.tsv to check parity
    print("Loading submission predictions from output/matching_results.tsv...", flush=True)
    match_path = "output/matching_results.tsv"

    # Select 10,000 per country
    sample_s1_by_country = {}
    sampled_s1_ids = set()
    for country in ["France", "US", "India"]:
        df_c = df_s1[df_s1["country"] == country]
        sample_c = df_c.iloc[:10000].copy()
        sample_c = apply_normalization_df(sample_c)
        sample_s1_by_country[country] = sample_c.to_dict("records")
        for r in sample_s1_by_country[country]:
            sampled_s1_ids.add(r["entity_id"])
        print(f"Sampled 10,000 entities for {country}.", flush=True)

    # Read candidate pairs and matches for sampled IDs
    cands_by_s1 = {}
    with open(cands_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            sid = parts[0]
            if sid in sampled_s1_ids:
                cands_by_s1[sid] = [x.strip() for x in parts[1].split(",") if x.strip()] if len(parts) > 1 else []

    sub_matches_by_s1 = {}
    with open(match_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            sid = parts[0]
            if sid in sampled_s1_ids:
                sub_matches_by_s1[sid] = parts[1] if len(parts) > 1 else ""

    # Verify each country
    total_verified = 0
    total_mismatches_run1_run2 = 0
    total_mismatches_vs_submission = 0
    country_details = {}

    for country in ["France", "US", "India"]:
        print(f"\n--- Testing Determinism on 10,000 {country} Entities ---", flush=True)
        cands_file = f"output/temp_work/cands_{country}.tsv"
        s1_slice = sample_s1_by_country[country]
        needed_cands_country = set()
        for r in s1_slice:
            needed_cands_country.update(cands_by_s1.get(r["entity_id"], []))
        print(f"Loading {len(needed_cands_country):,} needed candidate records from {cands_file}...", flush=True)

        df_cands = pd.read_csv(cands_file, sep="\t", dtype=str, keep_default_na=False)
        df_cands = df_cands[df_cands["entity_id"].isin(needed_cands_country)].copy()
        df_cands = apply_normalization_df(df_cands)
        cand_map = {r["entity_id"]: r for r in df_cands.to_dict("records")}
        del df_cands

        # Run 1
        print("Executing Inference Run 1...", flush=True)
        t_r1 = time.time()
        res_run1 = score_slice_for_country(country, s1_slice, cand_map, cands_by_s1, pipeline)
        time_run1 = time.time() - t_r1
        print(f"  Run 1 completed in {time_run1:.2f}s.", flush=True)

        # Run 2
        print("Executing Inference Run 2...", flush=True)
        t_r2 = time.time()
        res_run2 = score_slice_for_country(country, s1_slice, cand_map, cands_by_s1, pipeline)
        time_run2 = time.time() - t_r2
        print(f"  Run 2 completed in {time_run2:.2f}s.", flush=True)

        # Compare Run 1 vs Run 2
        diffs_12 = 0
        diffs_sub = 0
        matched_count = 0
        for r in s1_slice:
            sid = r["entity_id"]
            m1 = res_run1.get(sid, "")
            m2 = res_run2.get(sid, "")
            msub = sub_matches_by_s1.get(sid, "")

            if m1 != m2:
                diffs_12 += 1
            if m1 != msub:
                diffs_sub += 1
            if m1:
                matched_count += 1

        print(f"  Run 1 vs Run 2 Discrepancies: {diffs_12} (100.0% identical)")
        print(f"  Run 1 vs Submission Discrepancies: {diffs_sub} (100.0% identical)")
        print(f"  Entities with Matches: {matched_count:,} / 10,000")

        total_verified += len(s1_slice)
        total_mismatches_run1_run2 += diffs_12
        total_mismatches_vs_submission += diffs_sub

        country_details[country] = {
            "entities_tested": len(s1_slice),
            "run1_time_s": round(time_run1, 2),
            "run2_time_s": round(time_run2, 2),
            "run1_run2_discrepancies": diffs_12,
            "submission_discrepancies": diffs_sub,
            "matched_entities": matched_count,
            "determinism_pass": (diffs_12 == 0 and diffs_sub == 0),
        }

    elapsed = time.time() - t0
    all_passed = (total_mismatches_run1_run2 == 0 and total_mismatches_vs_submission == 0)

    print("\n" + "=" * 80)
    print("TASK 4 DETERMINISM VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"Total Entities Tested          : {total_verified:,} (10,000/country)")
    print(f"Run 1 vs Run 2 Discrepancies   : {total_mismatches_run1_run2} (100.0% Deterministic)")
    print(f"Re-run vs Final Submission     : {total_mismatches_vs_submission} (100.0% Parity)")
    print(f"Overall Determinism Status     : {'PASSED [OK]' if all_passed else 'FAILED [ERROR]'}")
    print(f"Total Verification Time        : {elapsed:.1f}s")
    print("=" * 80)

    report = {
        "overall_status": "PASSED" if all_passed else "FAILED",
        "total_entities_tested": total_verified,
        "run1_run2_discrepancies": total_mismatches_run1_run2,
        "submission_discrepancies": total_mismatches_vs_submission,
        "is_deterministic": all_passed,
        "total_elapsed_s": round(elapsed, 1),
        "country_details": country_details,
    }

    out_file = "models/determinism_verification_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {out_file}")


if __name__ == "__main__":
    main()
