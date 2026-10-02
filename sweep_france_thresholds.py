"""Checkpoint 1: France Coarse Threshold Sweep at tau in [0.75, 0.85, 0.90, 0.95].

Performs simultaneous multi-threshold evaluation in a single scoring pass:
1. Confirms Column_N to FEATURE_COLUMNS mapping.
2. Extracts features and evaluates LightGBM probabilities on all France test pairs.
3. Evaluates exact match distributions, match rates, singleton rates, and cap pileup for:
   - tau = 0.75
   - tau = 0.85
   - tau = 0.90
   - tau = 0.95
4. Writes exact candidate and matching results files for each tau for immediate downstream use.
"""

import gc
import os
import pickle
import sys
import time
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))
from src.features import FEATURE_COLUMNS, extract_pair_features
from src.normalize import apply_normalization_df


def main():
    t_start = time.time()
    print("=" * 80, flush=True)
    print("CHECKPOINT 1 — FRANCE COARSE THRESHOLD SWEEP", flush=True)
    print("=" * 80, flush=True)

    # 1. Load model and verify feature names mapping
    model_path = "models/lgbm_champion.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print("\n--- Model Feature Mapping and Importances ---", flush=True)
    print("clf.feature_name_:", model.feature_name_, flush=True)
    print(f"\n{'Col Name':<12} | {'Real Feature Name':<35} | {'Importance':<10}", flush=True)
    print("-" * 65, flush=True)
    for col_idx, col_name in enumerate(model.feature_name_):
        real_name = FEATURE_COLUMNS[col_idx]
        imp_val = model.feature_importances_[col_idx]
        print(f"{col_name:<12} | {real_name:<35} | {imp_val:>10,d}", flush=True)

    # 2. Load France S1
    test_dir = "student_resource/dataset/test"
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    print(f"\nLoading France Source 1 records from {s1_path}...", flush=True)
    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
    df_s1_fr = df_s1[df_s1["country"] == "France"].copy()
    n_s1 = len(df_s1_fr)
    print(f"Loaded {n_s1:,} S1 entities. Normalizing...", flush=True)
    df_s1_fr = apply_normalization_df(df_s1_fr)
    s1_map = {r["entity_id"]: r for r in df_s1_fr.to_dict("records")}
    del df_s1, df_s1_fr
    gc.collect()

    # 3. Load France candidates
    temp_dir = "output/temp_work"
    cands_path = os.path.join(temp_dir, "cands_France.tsv")
    print(f"Loading candidate records from {cands_path}...", flush=True)
    df_cands = pd.read_csv(cands_path, sep="\t", dtype=str, keep_default_na=False)
    print(f"Loaded {len(df_cands):,} candidate records. Normalizing...", flush=True)
    df_cands = apply_normalization_df(df_cands)
    cand_map = {r["entity_id"]: r for r in df_cands.to_dict("records")}
    del df_cands
    gc.collect()

    # 4. Stream and score across thresholds
    thresholds = [0.75, 0.85, 0.90, 0.95]
    max_k = 12
    max_matches = 10

    stats = {
        tau: {
            "dist": {k: 0 for k in range(max_matches + 1)},
            "total_matches": 0,
            "matched_entities": 0,
            "singletons": 0,
        }
        for tau in thresholds
    }

    in_results_path = os.path.join(temp_dir, "results_France.tsv")
    file_handles = {
        tau: open(os.path.join(temp_dir, f"lgbm_clean_results_France_tau_{int(tau*100)}.tsv"), "w", encoding="utf-8")
        for tau in thresholds
    }
    for tau, fh in file_handles.items():
        fh.write("source1_entity_id\tcandidate_entity_ids\tmatched_entity_ids\n")

    print(f"\nScoring {n_s1:,} entities across tau in {thresholds}...", flush=True)

    chunk_size = 8000
    current_chunk = []
    total_processed = 0
    t_score = time.time()

    def process_chunk(chunk):
        feat_matrix = []
        pair_meta = []  # (chunk_idx, cid)

        for idx, (sid, clist) in enumerate(chunk):
            r1 = s1_map.get(sid)
            if not r1:
                continue
            for cid in clist:
                rc = cand_map.get(cid)
                if not rc:
                    continue
                feats = extract_pair_features(r1, rc)
                feat_matrix.append([feats[col] for col in FEATURE_COLUMNS])
                pair_meta.append((idx, cid))

        if feat_matrix:
            X_chunk = np.array(feat_matrix, dtype=np.float32)
            probs = model.predict_proba(X_chunk)[:, 1]
        else:
            probs = np.array([])

        cand_scored_map = {i: [] for i in range(len(chunk))}
        for (idx, cid), prob in zip(pair_meta, probs):
            cand_scored_map[idx].append((cid, prob))

        for idx, (sid, clist) in enumerate(chunk):
            pairs_scored = cand_scored_map[idx]
            pairs_scored.sort(key=lambda x: x[1], reverse=True)

            for tau in thresholds:
                matched = [cid for cid, prob in pairs_scored if prob >= tau][:max_matches]
                matched_set = set(matched)

                opt_cands = list(matched)
                for cid, prob in pairs_scored:
                    if cid not in matched_set:
                        opt_cands.append(cid)
                    if len(opt_cands) >= max_k:
                        break

                c_str = ",".join(opt_cands)
                m_str = ",".join(matched)
                file_handles[tau].write(f"{sid}\t{c_str}\t{m_str}\n")

                n_m = len(matched)
                stats[tau]["dist"][n_m] += 1
                if n_m > 0:
                    stats[tau]["matched_entities"] += 1
                    stats[tau]["total_matches"] += n_m
                else:
                    stats[tau]["singletons"] += 1

    with open(in_results_path, "r", encoding="utf-8") as in_f:
        header = next(in_f)
        for line in in_f:
            parts = line.rstrip("\n").split("\t")
            s1_id = parts[0]
            cands = [c.strip() for c in parts[1].split(",") if c.strip()] if len(parts) > 1 and parts[1] else []
            current_chunk.append((s1_id, cands))

            if len(current_chunk) >= chunk_size:
                process_chunk(current_chunk)
                total_processed += len(current_chunk)
                current_chunk = []
                if total_processed % 50000 == 0 or total_processed >= n_s1:
                    pct = total_processed / n_s1 * 100.0
                    rate = total_processed / max(0.01, time.time() - t_score)
                    print(f"  Processed {total_processed:,}/{n_s1:,} ({pct:.1f}%) | {rate:,.0f} ent/s", flush=True)

        if current_chunk:
            process_chunk(current_chunk)
            total_processed += len(current_chunk)

    for fh in file_handles.values():
        fh.close()

    del s1_map, cand_map
    gc.collect()

    print(f"\nScoring completed in {time.time() - t_score:.1f}s.", flush=True)

    # 5. Print comprehensive measured comparison table
    print("\n" + "=" * 80, flush=True)
    print("FRANCE COARSE THRESHOLD SWEEP RESULTS (TOTAL ENTITIES = 259,452)", flush=True)
    print("=" * 80, flush=True)

    summary_rows = []
    for tau in thresholds:
        st = stats[tau]
        matched_pct = st["matched_entities"] / n_s1 * 100.0
        sing_pct = st["singletons"] / n_s1 * 100.0
        avg_per_matched = st["total_matches"] / st["matched_entities"] if st["matched_entities"] > 0 else 0.0
        bin_10_count = st["dist"][10]
        bin_10_pct = bin_10_count / n_s1 * 100.0
        summary_rows.append({
            "tau": f"{tau:.2f}",
            "match_rate": f"{matched_pct:.2f}%",
            "singleton_rate": f"{sing_pct:.2f}%",
            "avg_matches_per_matched": f"{avg_per_matched:.2f}",
            "cap_bin_10_entities": f"{bin_10_count:,} ({bin_10_pct:.2f}%)",
        })

    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False), flush=True)

    print("\n" + "=" * 80, flush=True)
    print("FULL MATCH COUNT DISTRIBUTIONS PER THRESHOLD", flush=True)
    print("=" * 80, flush=True)

    dist_table = []
    for k in range(max_matches + 1):
        row = {"matches_k": k}
        for tau in thresholds:
            cnt = stats[tau]["dist"][k]
            pct = cnt / n_s1 * 100.0
            row[f"tau_{tau:.2f}_cnt"] = f"{cnt:>7,d}"
            row[f"tau_{tau:.2f}_pct"] = f"{pct:>5.2f}%"
        dist_table.append(row)

    df_dist = pd.DataFrame(dist_table)
    print(df_dist.to_string(index=False), flush=True)
    print("\n" + "=" * 80, flush=True)


if __name__ == "__main__":
    main()
