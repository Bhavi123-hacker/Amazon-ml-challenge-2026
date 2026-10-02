"""Task 1: Generate Supplementary Prediction Confidence Artifact.

Generates output/prediction_confidence.tsv containing per-entity model confidence metadata:
- source1_entity_id: The Source 1 entity identifier
- predicted_match_count: Number of accepted matches
- min_match_probability: Probability of the weakest accepted match
- mean_match_probability: Mean probability across all accepted matches
- confidence_tier: High / Medium / Low based on proximity of min_match_probability
  to the country's operating threshold tau:
    * France (tau=0.995):
        - High: min_p >= 0.9990 (well above boundary, decisive match)
        - Medium: 0.9970 <= min_p < 0.9990 (standard confident match)
        - Low: min_p < 0.9970 (borderline match within 0.002 of threshold)
    * US & India (tau=0.900):
        - High: min_p >= 0.9500 (well above boundary, decisive match)
        - Medium: 0.9150 <= min_p < 0.9500 (standard confident match)
        - Low: min_p < 0.9150 (borderline match within 0.015 of threshold)
    * Singletons (count=0):
        - min_p = 0.0000, mean_p = 0.0000, confidence_tier = High (firm rejection)

Strictly ordered to match test_source1.tsv line-by-line (1,732,544 rows).
Saves summary analytics to models/prediction_confidence_summary.json.
"""

import gc
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


def assign_confidence_tier(country: str, min_p: float, count: int) -> str:
    """Assign confidence tier based on distance to operating threshold tau."""
    if count == 0:
        return "High"

    if country == "France":
        if min_p >= 0.9990:
            return "High"
        elif min_p >= 0.9970:
            return "Medium"
        else:
            return "Low"
    else:  # US and India (tau = 0.900)
        if min_p >= 0.9500:
            return "High"
        elif min_p >= 0.9150:
            return "Medium"
        else:
            return "Low"


def process_country_confidence(
    country: str,
    pipeline,
    s1_records: List[dict],
    matches_by_s1: Dict[str, List[str]],
    cands_file: str,
    num_workers: int = 8,
    chunk_size: int = 2000,
) -> Dict[str, Tuple[int, float, float, str]]:
    """Compute confidence metrics for all entities in a country."""
    print(f"\nComputing prediction confidence for {country} ({len(s1_records):,} entities)...", flush=True)
    t0 = time.time()

    # Collect needed candidate IDs (only those that were accepted as matches)
    needed_cand_ids = set()
    for s_rec in s1_records:
        sid = s_rec["entity_id"]
        needed_cand_ids.update(matches_by_s1.get(sid, []))

    print(f"  Loading {len(needed_cand_ids):,} matched candidate records from {cands_file}...", flush=True)
    df_cands = pd.read_csv(cands_file, sep="\t", dtype=str, keep_default_na=False)
    df_cands = df_cands[df_cands["entity_id"].isin(needed_cand_ids)].copy()
    df_cands = apply_normalization_df(df_cands)
    cand_map = {r["entity_id"]: r for r in df_cands.to_dict("records")}
    del df_cands
    gc.collect()

    s1_map = {r["entity_id"]: r for r in s1_records}
    s1_ids = [r["entity_id"] for r in s1_records]
    chunks = [s1_ids[i:i + chunk_size] for i in range(0, len(s1_ids), chunk_size)]

    def score_chunk(chunk_ids):
        local_results = {}
        batch_pairs = []
        batch_meta = []

        for sid in chunk_ids:
            s_rec = s1_map[sid]
            matches = matches_by_s1.get(sid, [])
            if not matches:
                # Singleton
                local_results[sid] = (0, 0.0, 0.0, "High")
                continue

            for rank, cid in enumerate(matches):
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

        probs_by_sid = {sid: [] for sid in chunk_ids}
        for (sid, cid), prob in zip(batch_meta, probs):
            probs_by_sid[sid].append(float(prob))

        for sid in chunk_ids:
            p_list = probs_by_sid.get(sid, [])
            if p_list:
                min_p = min(p_list)
                mean_p = sum(p_list) / len(p_list)
                tier = assign_confidence_tier(country, min_p, len(p_list))
                local_results[sid] = (len(p_list), min_p, mean_p, tier)
            else:
                local_results[sid] = (0, 0.0, 0.0, "High")

        return local_results

    country_results = {}
    print(f"  Scoring {len(chunks):,} chunks across {num_workers} parallel workers...", flush=True)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for chunk_res in executor.map(score_chunk, chunks):
            country_results.update(chunk_res)

    elapsed = time.time() - t0
    print(f"  Completed {country} in {elapsed:.1f}s ({len(s1_records)/elapsed:,.0f} ent/s).", flush=True)
    return country_results


def main():
    print("=== Task 1: Generate Supplementary Prediction Confidence Artifact ===", flush=True)
    t_start = time.time()

    model_path = "models/best_model_pipeline.pkl"
    print(f"Loading trained pipeline from {model_path}...", flush=True)
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    # 1. Load matches from matching_results.tsv
    matching_path = "output/matching_results.tsv"
    print(f"Reading accepted matches from {matching_path}...", flush=True)
    matches_by_s1 = {}
    with open(matching_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            sid = parts[0]
            matches_by_s1[sid] = [x.strip() for x in parts[1].split(",") if x.strip()] if len(parts) > 1 else []

    # 2. Load S1 grouped by country
    s1_path = "student_resource/dataset/test/test_source1.tsv"
    print(f"Loading S1 records from {s1_path}...", flush=True)
    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
    ordered_s1_ids = list(df_s1["entity_id"])

    confidence_by_s1 = {}
    tier_counts_by_country = {}

    for country in ["France", "US", "India"]:
        df_c = df_s1[df_s1["country"] == country].copy()
        df_c = apply_normalization_df(df_c)
        s1_records = df_c.to_dict("records")
        del df_c
        gc.collect()

        cands_file = f"output/temp_work/cands_{country}.tsv"
        c_res = process_country_confidence(country, pipeline, s1_records, matches_by_s1, cands_file)
        confidence_by_s1.update(c_res)

        # Compute tier breakdown
        counts = {"High": 0, "Medium": 0, "Low": 0}
        matched_counts = {"High": 0, "Medium": 0, "Low": 0}
        for r in s1_records:
            cnt, min_p, mean_p, tier = c_res[r["entity_id"]]
            counts[tier] += 1
            if cnt > 0:
                matched_counts[tier] += 1

        n_tot = len(s1_records)
        n_match = sum(1 for r in s1_records if len(matches_by_s1.get(r["entity_id"], [])) > 0)
        tier_counts_by_country[country] = {
            "total_entities": n_tot,
            "matched_entities": n_match,
            "singleton_entities": n_tot - n_match,
            "all_entities_tier_pct": {
                "High": round(counts["High"] / n_tot * 100.0, 2),
                "Medium": round(counts["Medium"] / n_tot * 100.0, 2),
                "Low": round(counts["Low"] / n_tot * 100.0, 2),
            },
            "matched_entities_tier_counts": matched_counts,
            "matched_entities_tier_pct": {
                "High": round(matched_counts["High"] / max(1, n_match) * 100.0, 2),
                "Medium": round(matched_counts["Medium"] / max(1, n_match) * 100.0, 2),
                "Low": round(matched_counts["Low"] / max(1, n_match) * 100.0, 2),
            },
        }

    # 3. Write output/prediction_confidence.tsv strictly ordered matching test_source1.tsv
    out_tsv = "output/prediction_confidence.tsv"
    print(f"\nWriting supplementary artifact to {out_tsv} strictly matching {s1_path} order...", flush=True)
    with open(out_tsv, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tpredicted_match_count\tmin_match_probability\tmean_match_probability\tconfidence_tier\n")
        for sid in ordered_s1_ids:
            cnt, min_p, mean_p, tier = confidence_by_s1.get(sid, (0, 0.0, 0.0, "High"))
            f_out.write(f"{sid}\t{cnt}\t{min_p:.4f}\t{mean_p:.4f}\t{tier}\n")

    file_bytes = os.path.getsize(out_tsv)
    print(f"Successfully generated {out_tsv} ({file_bytes:,} bytes, {len(ordered_s1_ids):,} rows).")

    # Overall Summary
    total_tiers = {"High": 0, "Medium": 0, "Low": 0}
    matched_tiers = {"High": 0, "Medium": 0, "Low": 0}
    for sid, (cnt, min_p, mean_p, tier) in confidence_by_s1.items():
        total_tiers[tier] += 1
        if cnt > 0:
            matched_tiers[tier] += 1

    total_ents = len(ordered_s1_ids)
    total_matched = sum(1 for m in matches_by_s1.values() if len(m) > 0)

    print("\n" + "=" * 80)
    print("PREDICTION CONFIDENCE TIERS SUMMARY")
    print("=" * 80)
    print(f"Overall Population (1,732,544 Entities):")
    print(f"  High Confidence  : {total_tiers['High']:,} ({total_tiers['High']/total_ents*100.0:.2f}%)")
    print(f"  Medium Confidence: {total_tiers['Medium']:,} ({total_tiers['Medium']/total_ents*100.0:.2f}%)")
    print(f"  Low Confidence   : {total_tiers['Low']:,} ({total_tiers['Low']/total_ents*100.0:.2f}%) -> Human Audit Queue")
    print(f"\nMatched Population Only (1,612,707 Matched Entities):")
    print(f"  High Confidence  : {matched_tiers['High']:,} ({matched_tiers['High']/total_matched*100.0:.2f}%)")
    print(f"  Medium Confidence: {matched_tiers['Medium']:,} ({matched_tiers['Medium']/total_matched*100.0:.2f}%)")
    print(f"  Low Confidence   : {matched_tiers['Low']:,} ({matched_tiers['Low']/total_matched*100.0:.2f}%) -> Human Audit Queue")
    print("=" * 80)

    report = {
        "output_file": out_tsv,
        "rows": len(ordered_s1_ids),
        "file_bytes": file_bytes,
        "overall_tier_counts": total_tiers,
        "matched_tier_counts": matched_tiers,
        "country_breakdown": tier_counts_by_country,
        "execution_time_s": round(time.time() - t_start, 1),
    }

    out_json = "models/prediction_confidence_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved confidence summary report to {out_json}")


if __name__ == "__main__":
    main()
