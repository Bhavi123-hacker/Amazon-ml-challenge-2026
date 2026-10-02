"""Generate full France predictions with Champion LightGBM + Guard D + Cluster Disambiguation.

France Settings:
- Base threshold: tau = 0.75
- Multi-tenant Guard D: reject if name_token_sort < 45 and addr_token_sort > 70
- Cluster Disambiguation: suppress if sim(top_name, cand_name) < 40 and p1 - p > 0.08
- Max candidates: K <= 12
- Max matches: <= 10
- 100% containment: matches <= candidates
"""

import gc
import os
import pickle
import sys
import time
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))
from src.features import FEATURE_COLUMNS, extract_pair_features
from src.normalize import apply_normalization_df

def main():
    t_start = time.time()
    print("=" * 80, flush=True)
    print("GENERATING FRANCE PREDICTIONS (CHAMPION LIGHTGBM + GUARD D + A1 DISAMBIG)", flush=True)
    print("=" * 80, flush=True)

    # 1. Load Champion Model
    model_path = "overnight_mission/models/champion_step3_model.pkl"
    print(f"Loading champion model from {model_path}...", flush=True)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # 2. Load France S1
    test_dir = "student_resource/dataset/test"
    temp_dir = "output/temp_work"
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    print(f"Loading France S1 records from {s1_path}...", flush=True)
    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
    df_s1_fr = df_s1[df_s1["country"] == "France"].copy()
    n_s1 = len(df_s1_fr)
    print(f"Loaded {n_s1:,} France S1 entities. Normalizing...", flush=True)
    df_s1_fr = apply_normalization_df(df_s1_fr)
    s1_map = {r["entity_id"]: r for r in df_s1_fr.to_dict("records")}
    del df_s1, df_s1_fr
    gc.collect()

    # 3. Load France Candidates
    cands_path = os.path.join(temp_dir, "cands_France.tsv")
    print(f"Loading candidate records from {cands_path}...", flush=True)
    df_cands = pd.read_csv(cands_path, sep="\t", dtype=str, keep_default_na=False)
    print(f"Loaded {len(df_cands):,} candidate records. Normalizing...", flush=True)
    df_cands = apply_normalization_df(df_cands)
    cand_map = {r["entity_id"]: r for r in df_cands.to_dict("records")}
    del df_cands
    gc.collect()

    # 4. Stream & Score
    in_results_path = os.path.join(temp_dir, "results_France.tsv")
    out_tsv = os.path.join(temp_dir, "champion_results_France_guard_d.tsv")
    out_f = open(out_tsv, "w", encoding="utf-8")
    out_f.write("source1_entity_id\tcandidate_entity_ids\tmatched_entity_ids\n")

    max_k = 12
    max_matches = 10
    tau = 0.75

    dist = {k: 0 for k in range(max_matches + 1)}
    total_matches = 0
    matched_entities = 0
    singletons = 0
    total_candidates = 0

    chunk_size = 8000
    current_chunk = []
    total_processed = 0
    t_score = time.time()

    def process_chunk(chunk):
        nonlocal total_matches, matched_entities, singletons, total_candidates
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

        cand_scored = {i: [] for i in range(len(chunk))}
        for (idx, cid), prob in zip(pair_meta, probs):
            cand_scored[idx].append((cid, float(prob)))

        for idx, (sid, clist) in enumerate(chunk):
            r1 = s1_map.get(sid, {})
            s1_name = r1.get("name_norm", "")
            s1_addr = r1.get("addr_norm", "")

            pairs = cand_scored[idx]
            pairs.sort(key=lambda x: x[1], reverse=True)

            top_cand_name = ""
            p1 = 0.0
            if pairs:
                rc0 = cand_map.get(pairs[0][0], {})
                top_cand_name = rc0.get("name_norm", "")
                p1 = pairs[0][1]

            accepted_matches = []
            for rank, (cid, p) in enumerate(pairs):
                if p < tau:
                    continue
                rc = cand_map.get(cid, {})
                c_name = rc.get("name_norm", "")
                c_addr = rc.get("addr_norm", "")

                ns = fuzz.token_sort_ratio(s1_name, c_name)
                asim = fuzz.token_sort_ratio(s1_addr, c_addr)

                # Guard D: reject multi-tenant collision
                if ns < 45 and asim > 70:
                    continue

                # Cluster Disambiguation
                if rank > 0:
                    sim_top = fuzz.token_sort_ratio(top_cand_name, c_name)
                    if sim_top < 40 and (p1 - p > 0.08):
                        continue

                accepted_matches.append(cid)
                if len(accepted_matches) >= max_matches:
                    break

            matched_set = set(accepted_matches)
            final_cands = list(accepted_matches)
            for cid, _ in pairs:
                if cid not in matched_set:
                    final_cands.append(cid)
                if len(final_cands) >= max_k:
                    break

            c_str = ",".join(final_cands)
            m_str = ",".join(accepted_matches)
            out_f.write(f"{sid}\t{c_str}\t{m_str}\n")

            n_m = len(accepted_matches)
            dist[n_m] = dist.get(n_m, 0) + 1
            total_matches += n_m
            total_candidates += len(final_cands)
            if n_m > 0:
                matched_entities += 1
            else:
                singletons += 1

    print(f"Scoring {n_s1:,} France entities...", flush=True)
    with open(in_results_path, "r", encoding="utf-8") as in_f:
        next(in_f)
        for line in in_f:
            parts = line.rstrip("\n").split("\t")
            sid = parts[0]
            cands = [c.strip() for c in parts[1].split(",") if c.strip()] if len(parts) > 1 and parts[1] else []
            current_chunk.append((sid, cands))

            if len(current_chunk) >= chunk_size:
                process_chunk(current_chunk)
                total_processed += len(current_chunk)
                current_chunk = []
                if total_processed % 40000 == 0 or total_processed >= n_s1:
                    rate = total_processed / max(0.01, time.time() - t_score)
                    print(f"  Processed {total_processed:,}/{n_s1:,} ({total_processed/n_s1*100:.1f}%) | "
                          f"{rate:,.0f} ent/s | Matches: {total_matches:,} (avg {total_matches/total_processed:.2f}) | "
                          f"Singletons: {singletons/total_processed*100:.2f}%", flush=True)

        if current_chunk:
            process_chunk(current_chunk)
            total_processed += len(current_chunk)

    out_f.close()
    del s1_map, cand_map
    gc.collect()

    print("\n" + "=" * 80, flush=True)
    print("FRANCE GENERATION COMPLETE", flush=True)
    print(f"Total processed: {total_processed:,}")
    print(f"Entities with matches: {matched_entities:,} ({matched_entities/total_processed*100:.2f}%)")
    print(f"Singletons: {singletons:,} ({singletons/total_processed*100:.2f}%)")
    print(f"Total matches: {total_matches:,} (avg {total_matches/total_processed:.2f}/entity)")
    print(f"Total candidates: {total_candidates:,} (avg {total_candidates/total_processed:.2f}/entity)")
    print(f"Time taken: {time.time() - t_start:.1f}s")
    print(f"Output saved to: {out_tsv}")
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
