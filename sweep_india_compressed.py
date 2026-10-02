"""India Compressed Threshold Sweep (tau = 0.90 and tau = 0.98).

Evaluates both thresholds simultaneously in a single pass:
- Match rate (%) and singleton rate (%)
- Full match-count distribution (bins 0 through 10)
- Average matches per matched entity
- Cap bin 10 count and %
- Writes:
    output/temp_work/lgbm_clean_results_India_tau_90.tsv
    output/temp_work/lgbm_clean_results_India_tau_98.tsv
"""

import gc
import os
import pickle
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))
from src.features import FEATURE_COLUMNS, extract_pair_features
from src.normalize import apply_normalization_df


def main():
    t_start = time.time()
    print("=" * 80, flush=True)
    print("INDIA COMPRESSED THRESHOLD EVALUATION (TAU = 0.90 AND TAU = 0.98)", flush=True)
    print("=" * 80, flush=True)

    # 1. Load LightGBM Champion model
    model_path = "models/lgbm_champion.pkl"
    print(f"Loading champion model from {model_path}...", flush=True)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # 2. Load India S1 records
    test_dir = "student_resource/dataset/test"
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    print(f"Loading India Source 1 records from {s1_path}...", flush=True)
    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
    df_s1_in = df_s1[df_s1["country"] == "India"].copy()
    n_s1 = len(df_s1_in)
    print(f"Loaded {n_s1:,} India S1 entities. Normalizing...", flush=True)
    del df_s1
    gc.collect()

    df_s1_in = apply_normalization_df(df_s1_in)

    # Pack into lightweight dict of dicts for feature extraction
    s1_map = {}
    for r in df_s1_in.to_dict("records"):
        s1_map[r["entity_id"]] = {
            "entity_id": r["entity_id"],
            "name_norm": r.get("name_norm", ""),
            "addr_norm": r.get("addr_norm", ""),
            "legal_suffix": r.get("legal_suffix", ""),
            "pin_extracted": r.get("pin_extracted", ""),
            "street_number": r.get("street_number", ""),
            "city_extracted": r.get("city_extracted", ""),
            "country": "India"
        }
    del df_s1_in
    gc.collect()
    print(f"S1 records normalized and packed ({len(s1_map):,} entities).", flush=True)

    # 3. Load candidate pool in chunks
    temp_dir = "output/temp_work"
    cands_path = os.path.join(temp_dir, "cands_India.tsv")
    print(f"Loading candidate records in chunks from {cands_path}...", flush=True)

    cand_map = {}
    chunk_reader = pd.read_csv(cands_path, sep="\t", dtype=str, keep_default_na=False, chunksize=600000)
    total_cands_loaded = 0

    t_cands = time.time()
    for chunk_idx, df_c_chunk in enumerate(chunk_reader):
        df_c_chunk = apply_normalization_df(df_c_chunk)
        for r in df_c_chunk.to_dict("records"):
            cand_map[r["entity_id"]] = {
                "entity_id": r["entity_id"],
                "name_norm": r.get("name_norm", ""),
                "addr_norm": r.get("addr_norm", ""),
                "legal_suffix": r.get("legal_suffix", ""),
                "pin_extracted": r.get("pin_extracted", ""),
                "street_number": r.get("street_number", ""),
                "city_extracted": r.get("city_extracted", ""),
                "country": "India"
            }
        total_cands_loaded += len(df_c_chunk)
        del df_c_chunk
        gc.collect()
        print(f"  Loaded candidate chunk {chunk_idx + 1} | {total_cands_loaded:,} total candidates in {time.time() - t_cands:.1f}s", flush=True)

    print(f"Candidate loading complete. {len(cand_map):,} unique candidates in memory.", flush=True)

    # 4. Setup output files and metrics tracking
    max_k = 12
    max_matches = 10

    out_tsv_90 = os.path.join(temp_dir, "lgbm_clean_results_India_tau_90.tsv")
    out_tsv_98 = os.path.join(temp_dir, "lgbm_clean_results_India_tau_98.tsv")

    f_out_90 = open(out_tsv_90, "w", encoding="utf-8")
    f_out_90.write("source1_entity_id\tcandidate_entity_ids\tmatched_entity_ids\n")

    f_out_98 = open(out_tsv_98, "w", encoding="utf-8")
    f_out_98.write("source1_entity_id\tcandidate_entity_ids\tmatched_entity_ids\n")

    dist_90 = {k: 0 for k in range(max_matches + 1)}
    total_m_90 = 0
    matched_ent_90 = 0
    sing_90 = 0

    dist_98 = {k: 0 for k in range(max_matches + 1)}
    total_m_98 = 0
    matched_ent_98 = 0
    sing_98 = 0

    in_results_path = os.path.join(temp_dir, "results_India.tsv")
    print(f"\nScoring {n_s1:,} India entities against candidates at tau in [0.90, 0.98]...", flush=True)

    batch_size = 8000
    current_chunk = []
    total_processed = 0
    t_score = time.time()

    def process_chunk(chunk):
        nonlocal total_m_90, matched_ent_90, sing_90
        nonlocal total_m_98, matched_ent_98, sing_98

        feat_matrix = []
        pair_meta = []

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

            # Evaluate tau = 0.90
            matched_90 = [cid for cid, prob in pairs_scored if prob >= 0.90][:max_matches]
            m_set_90 = set(matched_90)
            cands_90 = list(matched_90)
            for cid, prob in pairs_scored:
                if cid not in m_set_90:
                    cands_90.append(cid)
                if len(cands_90) >= max_k:
                    break
            f_out_90.write(f"{sid}\t{','.join(cands_90)}\t{','.join(matched_90)}\n")

            n90 = len(matched_90)
            dist_90[n90] += 1
            if n90 > 0:
                matched_ent_90 += 1
                total_m_90 += n90
            else:
                sing_90 += 1

            # Evaluate tau = 0.98
            matched_98 = [cid for cid, prob in pairs_scored if prob >= 0.98][:max_matches]
            m_set_98 = set(matched_98)
            cands_98 = list(matched_98)
            for cid, prob in pairs_scored:
                if cid not in m_set_98:
                    cands_98.append(cid)
                if len(cands_98) >= max_k:
                    break
            f_out_98.write(f"{sid}\t{','.join(cands_98)}\t{','.join(matched_98)}\n")

            n98 = len(matched_98)
            dist_98[n98] += 1
            if n98 > 0:
                matched_ent_98 += 1
                total_m_98 += n98
            else:
                sing_98 += 1

    with open(in_results_path, "r", encoding="utf-8") as in_f:
        header = next(in_f)
        for line in in_f:
            parts = line.rstrip("\n").split("\t")
            s1_id = parts[0]
            cands = [c.strip() for c in parts[1].split(",") if c.strip()] if len(parts) > 1 and parts[1] else []
            current_chunk.append((s1_id, cands))

            if len(current_chunk) >= batch_size:
                process_chunk(current_chunk)
                total_processed += len(current_chunk)
                current_chunk = []
                if total_processed % 50000 < batch_size:
                    pct = total_processed / n_s1 * 100.0
                    rate = total_processed / max(0.01, time.time() - t_score)
                    print(f"  Processed {total_processed:,}/{n_s1:,} ({pct:.1f}%) | {rate:,.0f} ent/s", flush=True)

        if current_chunk:
            process_chunk(current_chunk)
            total_processed += len(current_chunk)

    f_out_90.close()
    f_out_98.close()

    print(f"\nScoring completed in {time.time() - t_score:.1f}s.", flush=True)

    # 5. Print reports
    print("\n" + "=" * 80, flush=True)
    print("INDIA THRESHOLD SWEEP SUMMARY (TOTAL ENTITIES = 809,986)", flush=True)
    print("=" * 80, flush=True)

    summary_rows = []
    for tau, matched_ent, sing, total_m, dist in [
        (0.90, matched_ent_90, sing_90, total_m_90, dist_90),
        (0.98, matched_ent_98, sing_98, total_m_98, dist_98)
    ]:
        m_pct = matched_ent / n_s1 * 100.0
        s_pct = sing / n_s1 * 100.0
        avg_m = total_m / matched_ent if matched_ent > 0 else 0.0
        b10_cnt = dist[10]
        b10_pct = b10_cnt / n_s1 * 100.0
        summary_rows.append({
            "tau": f"{tau:.2f}",
            "match_rate": f"{m_pct:.2f}%",
            "singleton_rate": f"{s_pct:.2f}%",
            "avg_matches_per_matched": f"{avg_m:.2f}",
            "cap_bin_10_entities": f"{b10_cnt:,} ({b10_pct:.2f}%)"
        })

    df_sum = pd.DataFrame(summary_rows)
    print(df_sum.to_string(index=False), flush=True)

    print("\n" + "=" * 80, flush=True)
    print("INDIA FULL MATCH COUNT DISTRIBUTIONS", flush=True)
    print("=" * 80, flush=True)

    dist_rows = []
    for k in range(max_matches + 1):
        c90 = dist_90[k]
        p90 = c90 / n_s1 * 100.0
        c98 = dist_98[k]
        p98 = c98 / n_s1 * 100.0
        dist_rows.append({
            "matches_k": k,
            "tau_0.90_count": f"{c90:>7,d}",
            "tau_0.90_pct": f"{p90:>5.2f}%",
            "tau_0.98_count": f"{c98:>7,d}",
            "tau_0.98_pct": f"{p98:>5.2f}%"
        })
    df_dist = pd.DataFrame(dist_rows)
    print(df_dist.to_string(index=False), flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
