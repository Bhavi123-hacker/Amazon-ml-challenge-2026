"""Unified High-Speed 2-Stage Dynamic Clustering Pipeline across France, US, India.

Implements:
1. Champion LightGBM model (49 clean features, transliteration, missing indicators).
2. Dynamic Probability Margin: p1 >= 0.75 and (p1 - p <= 0.08), eliminating trailing distractors.
3. France Guard D: rejects co-located multi-tenant collisions (N_sim < 45 & A_sim > 70).
4. Natural Cluster Distribution: max matches = 6 (matches ground truth tail).
5. Pareto Candidate Cutoff: K <= 12 (51.55% distractor reduction).
6. 100% Containment: matches <= candidates.
7. Strict Byte Synchronization between output/ and BitMinds/output/.
8. Official Submission Validator Execution.
9. Packaging BitMinds_submission.zip.
"""

import gc
import hashlib
import os
import pickle
import shutil
import subprocess
import sys
import time
import zipfile
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))
from src.features import FEATURE_COLUMNS, extract_pair_features
from src.normalize import apply_normalization_df

def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def process_country(country: str, model, test_dir: str, temp_dir: str, min_p: float = 0.75, delta_p: float = 0.08, max_matches: int = 6, max_k: int = 12):
    print("\n" + "=" * 85, flush=True)
    print(f"PROCESSING {country.upper()} WITH 2-STAGE DYNAMIC CLUSTERING (min_p={min_p}, delta_p={delta_p})", flush=True)
    print("=" * 85, flush=True)
    t0 = time.time()

    s1_path = os.path.join(test_dir, "test_source1.tsv")
    in_results_path = os.path.join(temp_dir, f"results_{country}.tsv")
    out_tsv = os.path.join(temp_dir, f"dynamic_results_{country}.tsv")

    # 1. Load S1 for country
    print(f"Loading {country} S1 records...", flush=True)
    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
    df_s1_country = df_s1[df_s1["country"] == country].copy()
    n_s1 = len(df_s1_country)
    print(f"Loaded {n_s1:,} S1 entities. Normalizing...", flush=True)
    df_s1_country = apply_normalization_df(df_s1_country)
    s1_map = {r["entity_id"]: r for r in df_s1_country.to_dict("records")}
    del df_s1, df_s1_country
    gc.collect()

    # 2. Collect referenced candidate IDs from results_{country}.tsv
    print(f"Collecting referenced candidate IDs from {in_results_path}...", flush=True)
    referenced_cands = set()
    with open(in_results_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) > 1 and p[1]:
                for cid in p[1].split(","):
                    cid_s = cid.strip()
                    if cid_s:
                        referenced_cands.add(cid_s)
    print(f"Referenced unique candidates: {len(referenced_cands):,}.", flush=True)

    # 3. Load candidates filtering strictly to referenced IDs
    cands_path = os.path.join(temp_dir, f"cands_{country}.tsv")
    print(f"Loading candidate records from {cands_path}...", flush=True)
    c_chunks = []
    for chunk in pd.read_csv(cands_path, sep="\t", dtype=str, keep_default_na=False, chunksize=300000):
        sub = chunk[chunk["entity_id"].isin(referenced_cands)]
        if len(sub) > 0:
            c_chunks.append(sub)
    df_cands = pd.concat(c_chunks, ignore_index=True).drop_duplicates("entity_id")
    del c_chunks
    gc.collect()

    print(f"Loaded {len(df_cands):,} candidate records. Normalizing...", flush=True)
    df_cands = apply_normalization_df(df_cands)
    cand_map = {r["entity_id"]: r for r in df_cands.to_dict("records")}
    del df_cands
    gc.collect()

    # 4. Stream & Score with Dynamic Margin Clustering
    print(f"Scoring {n_s1:,} entities with Dynamic Margin Clustering...", flush=True)
    out_f = open(out_tsv, "w", encoding="utf-8")
    out_f.write("source1_entity_id\tcandidate_entity_ids\tmatched_entity_ids\n")

    chunk_size = 8000
    current_chunk = []
    total_processed = 0
    total_matches = 0
    singletons = 0
    t_score = time.time()

    def process_chunk(chunk):
        nonlocal total_matches, singletons
        feat_matrix = []
        pair_meta = []  # (idx, cid)

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
            probs = model.predict_proba(np.array(feat_matrix, dtype=np.float32))[:, 1]
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

            accepted_matches = []
            if pairs and pairs[0][1] >= min_p:
                p1 = pairs[0][1]
                top_cid = pairs[0][0]
                rc0 = cand_map.get(top_cid, {})
                top_name = rc0.get("name_norm", "")
                top_addr = rc0.get("addr_norm", "")

                # Check top candidate Guard D (France)
                ns0 = fuzz.token_sort_ratio(s1_name, top_name)
                asim0 = fuzz.token_sort_ratio(s1_addr, top_addr)
                if country == "France" and (ns0 < 45 and asim0 > 70):
                    pass  # Reject top match if multi-tenant collision
                else:
                    accepted_matches.append(top_cid)

                # Check secondary candidates
                for cid, p in pairs[1:]:
                    if p < min_p:
                        break
                    # Dynamic Margin: prune trailing distractors
                    if (p1 - p) > delta_p:
                        break

                    rc = cand_map.get(cid, {})
                    c_name = rc.get("name_norm", "")
                    c_addr = rc.get("addr_norm", "")

                    # France Guard D
                    if country == "France":
                        ns = fuzz.token_sort_ratio(s1_name, c_name)
                        asim = fuzz.token_sort_ratio(s1_addr, c_addr)
                        if ns < 45 and asim > 70:
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
            total_matches += n_m
            if n_m == 0:
                singletons += 1

    with open(in_results_path, "r", encoding="utf-8") as in_f:
        next(in_f)
        for line in in_f:
            p = line.rstrip("\n").split("\t")
            sid = p[0]
            clist = [c.strip() for c in p[1].split(",") if c.strip()] if len(p) > 1 and p[1] else []
            current_chunk.append((sid, clist))

            if len(current_chunk) >= chunk_size:
                process_chunk(current_chunk)
                total_processed += len(current_chunk)
                current_chunk = []
                if total_processed % 50000 == 0 or total_processed >= n_s1:
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

    print(f"\n{country} Complete in {time.time() - t0:.1f}s:", flush=True)
    print(f"  Total processed: {total_processed:,}")
    print(f"  Singletons: {singletons:,} ({singletons/total_processed*100:.2f}%)")
    print(f"  Total matches: {total_matches:,} (avg {total_matches/total_processed:.2f}/entity)")
    print(f"  Output saved: {out_tsv}")
    return out_tsv

def main():
    t_global = time.time()
    print("=" * 85, flush=True)
    print("STARTING UNIFIED 2-STAGE DYNAMIC CLUSTERING PRODUCTION RUN", flush=True)
    print("=" * 85, flush=True)

    test_dir = "student_resource/dataset/test"
    output_dir = "output"
    temp_dir = os.path.join(output_dir, "temp_work")
    os.makedirs(temp_dir, exist_ok=True)

    model_path = "overnight_mission/models/champion_step3_model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("Loaded Champion LightGBM model successfully.", flush=True)

    # Optimal calibrated parameters from empirical grid search
    # min_p = 0.75, delta_p = 0.08, max_matches = 6, max_k = 12
    fr_res = process_country("France", model, test_dir, temp_dir, min_p=0.75, delta_p=0.08, max_matches=6, max_k=12)
    us_res = process_country("US", model, test_dir, temp_dir, min_p=0.75, delta_p=0.08, max_matches=6, max_k=12)
    in_res = process_country("India", model, test_dir, temp_dir, min_p=0.75, delta_p=0.08, max_matches=6, max_k=12)

    # Stitch outputs strictly in test_source1.tsv line order
    print("\n" + "=" * 85, flush=True)
    print("STITCHING FINAL MASTER SUBMISSION OUTPUTS", flush=True)
    print("=" * 85, flush=True)

    s1_path = os.path.join(test_dir, "test_source1.tsv")
    match_file = os.path.join(output_dir, "matching_results.tsv")
    cand_file = os.path.join(output_dir, "candidate_pairs.tsv")

    country_files = [fr_res, us_res, in_res]
    res_map = {}
    for cf in country_files:
        print(f"Reading {cf}...", flush=True)
        with open(cf, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\n").split("\t")
                res_map[p[0]] = (p[1] if len(p) > 1 else "", p[2] if len(p) > 2 else "")

    print(f"Total entities loaded from country files: {len(res_map):,}.", flush=True)

    f_match = open(match_file, "w", encoding="utf-8")
    f_cand = open(cand_file, "w", encoding="utf-8")
    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

    total = 0
    matched = 0
    singletons = 0
    total_matches = 0
    total_cands = 0
    match_counts = {}

    with open(s1_path, "r", encoding="utf-8") as f_s1:
        next(f_s1)
        for line in f_s1:
            total += 1
            sid = line.split("\t", 1)[0].strip()
            c_str, m_str = res_map.get(sid, ("", ""))

            f_match.write(f"{sid}\t{m_str}\n")
            f_cand.write(f"{sid}\t{c_str}\n")

            m_list = [m for m in m_str.split(",") if m.strip()] if m_str else []
            c_list = [c for c in c_str.split(",") if c.strip()] if c_str else []

            n_m = len(m_list)
            match_counts[n_m] = match_counts.get(n_m, 0) + 1
            total_matches += n_m
            total_cands += len(c_list)
            if n_m > 0:
                matched += 1
            else:
                singletons += 1

    f_match.close()
    f_cand.close()

    print(f"\nFinal Master Submission Files Written:")
    print(f"  Total Entities      : {total:,}")
    print(f"  Entities with Match : {matched:,} ({matched/total*100:.2f}%)")
    print(f"  Natural Singletons  : {singletons:,} ({singletons/total*100:.2f}%)")
    print(f"  Total Matched Pairs : {total_matches:,} (avg {total_matches/total:.2f}/entity)")
    print(f"  Total Candidate Pairs: {total_cands:,} (avg {total_cands/total:.2f}/entity)")

    print("\n--- Match Distribution ---")
    for k in sorted(match_counts.keys()):
        print(f"  {k:>2} matches: {match_counts[k]:>8,d} entities ({match_counts[k]/total*100:>5.2f}%)")

    # Byte-Level Synchronization with BitMinds/output/
    print("\n" + "=" * 85, flush=True)
    print("SYNCHRONIZING WITH BitMinds/output/ DIRECTORY", flush=True)
    print("=" * 85, flush=True)
    bm_out_dir = "BitMinds/output"
    os.makedirs(bm_out_dir, exist_ok=True)
    bm_match_file = os.path.join(bm_out_dir, "matching_results.tsv")
    bm_cand_file = os.path.join(bm_out_dir, "candidate_pairs.tsv")

    shutil.copy2(match_file, bm_match_file)
    shutil.copy2(cand_file, bm_cand_file)

    h_m1 = compute_sha256(match_file)
    h_m2 = compute_sha256(bm_match_file)
    h_c1 = compute_sha256(cand_file)
    h_c2 = compute_sha256(bm_cand_file)

    print(f"output/matching_results.tsv           SHA-256: {h_m1}")
    print(f"BitMinds/output/matching_results.tsv  SHA-256: {h_m2}")
    assert h_m1 == h_m2, "MATCHING HASH MISMATCH!"
    print(f"output/candidate_pairs.tsv            SHA-256: {h_c1}")
    print(f"BitMinds/output/candidate_pairs.tsv   SHA-256: {h_c2}")
    assert h_c1 == h_c2, "CANDIDATE HASH MISMATCH!"
    print("  -> Hashes are 100% byte-identical!")

    # Official Validation
    print("\n" + "=" * 85, flush=True)
    print("RUNNING OFFICIAL VALIDATION", flush=True)
    print("=" * 85, flush=True)
    val_cmd = [
        sys.executable,
        "student_resource/utils/validate_submission.py",
        "--matching", match_file,
        "--candidate", cand_file,
        "--test-dir", test_dir
    ]
    res = subprocess.run(val_cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr)
    print(f"Validation return code: {res.returncode}")
    assert res.returncode == 0, "VALIDATION FAILED!"

    # Packaging BitMinds_submission.zip
    print("\n" + "=" * 85, flush=True)
    print("REPACKAGING BitMinds_submission.zip", flush=True)
    print("=" * 85, flush=True)
    zip_path = "BitMinds_submission.zip"
    if os.path.exists(zip_path):
        os.remove(zip_path)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(match_file, "output/matching_results.tsv")
        zipf.write(cand_file, "output/candidate_pairs.tsv")
        zipf.write("Documentation_template.md", "Documentation_template.md")
        for root, dirs, files in os.walk("code/business_entity_resolution"):
            for f in files:
                if f.endswith((".py", ".md", ".txt", ".sh", ".json")):
                    full_p = os.path.join(root, f)
                    rel_p = os.path.relpath(full_p, ".")
                    zipf.write(full_p, rel_p)

    zip_size = os.path.getsize(zip_path)
    zip_sha = compute_sha256(zip_path)
    print(f"Created {zip_path} ({zip_size:,} bytes, {zip_size/(1024*1024):.2f} MB)")
    print(f"SHA-256: {zip_sha}")
    print(f"\nALL DONE IN {time.time() - t_global:.1f}s!", flush=True)

if __name__ == "__main__":
    main()
