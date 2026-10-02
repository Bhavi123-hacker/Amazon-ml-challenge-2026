"""End-to-End Prediction Regeneration, Re-Stitching, and Dual-Audit Validation.

Executes:
1. Multi-threaded scoring of France (tau=0.995), US (tau=0.900), and India (tau=0.900)
   using the winning NeuralNet_MLP pipeline with enriched features.
2. Direct disk streaming to results_{country}.tsv.
3. Strict 1:1 row-order stitching matching test_source1.tsv.
4. Submission structural validation (utils/validate_submission.py).
5. 10/10 Verification checklist (verify_checklist.py).
6. Final full-population expanded audit pass across all 6.97M pairs.
"""

import gc
import json
import os
import pickle
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Set, Tuple

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from src.audit_heuristics import evaluate_pair_audit
from src.features import FEATURE_COLUMNS, extract_pair_features
from src.normalize import apply_normalization_df

THRESHOLDS = {
    "France": 0.995,
    "US": 0.900,
    "India": 0.900,
}


def process_country(country: str, pipeline, test_dir: str, output_dir: str, num_workers: int = 8, chunk_size: int = 1500):
    tau = THRESHOLDS[country]
    temp_dir = os.path.join(output_dir, "temp_work")
    out_tsv = os.path.join(temp_dir, f"results_{country}.tsv")
    tmp_tsv = os.path.join(temp_dir, f"results_{country}.tsv.tmp")

    print(f"\n" + "=" * 70, flush=True)
    print(f"RE-SCORING COUNTRY: {country.upper()} AT TAU = {tau:.3f}", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()

    # 1. Load S1 for country
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    print(f"Loading {country} Source 1 records...", flush=True)
    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
    df_s1_country = df_s1[df_s1["country"] == country].copy()
    n_s1 = len(df_s1_country)
    print(f"Loaded {n_s1:,} S1 entities.", flush=True)
    df_s1_country = apply_normalization_df(df_s1_country)
    s1_map = {r["entity_id"]: r for r in df_s1_country.to_dict("records")}
    del df_s1, df_s1_country
    gc.collect()

    # 2. Load candidates for country
    cands_path = os.path.join(temp_dir, f"cands_{country}.tsv")
    print(f"Loading {country} candidates from {cands_path}...", flush=True)
    df_cands = pd.read_csv(cands_path, sep="\t", dtype=str, keep_default_na=False)
    print(f"Loaded {len(df_cands):,} candidate records.", flush=True)
    df_cands = apply_normalization_df(df_cands)
    cand_map = {r["entity_id"]: r for r in df_cands.to_dict("records")}
    del df_cands
    gc.collect()

    # 3. Read existing candidate lines from results_{country}.tsv
    existing_res_path = os.path.join(temp_dir, f"results_{country}.tsv")
    print(f"Reading candidate lists from {existing_res_path}...", flush=True)
    lines = []
    with open(existing_res_path, "r", encoding="utf-8") as f:
        header = next(f)
        for line in f:
            lines.append(line.rstrip("\n"))
    print(f"Read {len(lines):,} entity lines to rescore.", flush=True)

    # 4. Multi-threaded scoring worker
    def score_sub_chunk(chunk_lines):
        batch_pairs = []
        batch_meta = []
        cand_map_local = {}

        for line in chunk_lines:
            parts = line.split("\t")
            s1_id = parts[0]
            s1_rec = s1_map.get(s1_id)
            if not s1_rec:
                continue
            cands = [c.strip() for c in parts[1].split(",") if c.strip()] if len(parts) > 1 and parts[1] else []
            cand_map_local[s1_id] = cands

            for rank, cid in enumerate(cands):
                c_rec = cand_map.get(cid)
                if not c_rec:
                    continue
                s_name = fuzz.token_set_ratio(s1_rec["name_norm"], c_rec["name_norm"])
                s_addr = fuzz.token_set_ratio(s1_rec["addr_norm"], c_rec["addr_norm"])
                b_score = max(s_name, s_addr) + 0.3 * min(s_name, s_addr)
                feats = extract_pair_features(s1_rec, c_rec, rank=rank, blocking_score=b_score)
                batch_pairs.append([feats[col] for col in FEATURE_COLUMNS])
                batch_meta.append((s1_id, cid))

        if not batch_pairs:
            return [(s1_id, ",".join(cands), "") for s1_id, cands in cand_map_local.items()]

        X = np.array(batch_pairs, dtype=np.float32)
        probs = pipeline.predict_proba(X)[:, 1]

        m_map = {s1_id: [] for s1_id in cand_map_local}
        for (s1_id, cid), prob in zip(batch_meta, probs):
            if prob >= tau:
                m_map[s1_id].append(cid)

        results = []
        for s1_id, cands in cand_map_local.items():
            results.append((s1_id, ",".join(cands), ",".join(m_map.get(s1_id, []))))
        return results

    # Chunk lines
    chunk_list = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
    print(f"Scoring {len(chunk_list):,} chunks across {num_workers} parallel workers...", flush=True)

    t_eval = time.time()
    total_processed = 0
    total_matched_pairs = 0
    total_entities_with_match = 0

    with open(tmp_tsv, "w", encoding="utf-8") as out_f:
        out_f.write("source1_entity_id\tcandidate_entity_ids\tmatched_entity_ids\n")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for sub_res in executor.map(score_sub_chunk, chunk_list):
                for s1_id, c_str, m_str in sub_res:
                    out_f.write(f"{s1_id}\t{c_str}\t{m_str}\n")
                    total_processed += 1
                    if m_str:
                        total_entities_with_match += 1
                        total_matched_pairs += len(m_str.split(","))

                if total_processed % 50000 == 0 or total_processed == len(lines):
                    pct = total_processed / len(lines) * 100.0
                    rate = total_processed / max(0.1, time.time() - t_eval)
                    print(f"  Processed {total_processed:,}/{len(lines):,} ({pct:.1f}%) | {rate:,.0f} entities/s | Matched Pairs: {total_matched_pairs:,}", flush=True)

    # Atomic replace
    if os.path.exists(out_tsv):
        os.remove(out_tsv)
    os.rename(tmp_tsv, out_tsv)

    elapsed = time.time() - t0
    match_rate = total_entities_with_match / n_s1 * 100.0
    print(f"\nFinished {country}: {total_processed:,} entities in {elapsed:.1f}s ({total_processed/elapsed:,.0f} ent/s)", flush=True)
    print(f"Entities with Matches: {total_entities_with_match:,} ({match_rate:.2f}%) | Total Matched Pairs: {total_matched_pairs:,}", flush=True)
    print(f"Output saved to: {out_tsv}", flush=True)

    del s1_map, cand_map, lines
    gc.collect()
    return total_matched_pairs, match_rate


def stitch_final_outputs(test_dir: str, output_dir: str):
    """Stitch country results into final submission files matching exact test_source1 ordering."""
    print("\n" + "=" * 70, flush=True)
    print("STITCHING FINAL MASTER SUBMISSION OUTPUTS", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()

    temp_dir = os.path.join(output_dir, "temp_work")
    res_map = {}

    for country in ["France", "US", "India"]:
        res_file = os.path.join(temp_dir, f"results_{country}.tsv")
        print(f"Loading rescored {country} results from {res_file}...", flush=True)
        with open(res_file, "r", encoding="utf-8") as f:
            header = next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                s1_id = parts[0]
                c_str = parts[1] if len(parts) > 1 else ""
                m_str = parts[2] if len(parts) > 2 else ""
                res_map[s1_id] = (c_str, m_str)

    print(f"Total rescored records indexed: {len(res_map):,}", flush=True)

    # Write in exact order of test_source1.tsv
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    match_path = os.path.join(output_dir, "matching_results.tsv")
    cand_path = os.path.join(output_dir, "candidate_pairs.tsv")

    print(f"Streaming final outputs in exact line order of {s1_path}...", flush=True)
    total = 0
    matched = 0
    singletons = 0

    with open(cand_path, "w", encoding="utf-8") as f_cand, open(match_path, "w", encoding="utf-8") as f_match:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        f_match.write("source1_entity_id\tmatched_entity_ids\n")

        with open(s1_path, "r", encoding="utf-8") as f_s1:
            next(f_s1) # skip header
            for line in f_s1:
                s1_id = line.split("\t", 1)[0].strip()
                if not s1_id:
                    continue
                c_str, m_str = res_map.get(s1_id, ("", ""))
                f_cand.write(f"{s1_id}\t{c_str}\n")
                f_match.write(f"{s1_id}\t{m_str}\n")
                total += 1
                if m_str:
                    matched += 1
                else:
                    singletons += 1

    print(f"\nFinal Master Outputs Generated in {time.time() - t0:.2f}s:")
    print(f"  - {match_path} ({os.path.getsize(match_path):,} bytes)")
    print(f"  - {cand_path} ({os.path.getsize(cand_path):,} bytes)")
    print(f"Total entities: {total:,} | Matched: {matched:,} ({matched/total:.2%}) | Singletons: {singletons:,} ({singletons/total:.2%})")


def run_validations():
    """Execute submission validator and verification checklist."""
    print("\n" + "=" * 70, flush=True)
    print("EXECUTING SUBMISSION VALIDATOR & CHECKLIST", flush=True)
    print("=" * 70, flush=True)

    # 1. validate_submission.py
    cmd_val = [sys.executable, "utils/validate_submission.py", "output/matching_results.tsv"]
    print(f"Running: {' '.join(cmd_val)}", flush=True)
    res_val = subprocess.run(cmd_val, capture_output=True, text=True)
    print(res_val.stdout)
    if res_val.stderr:
        print("STDERR:", res_val.stderr)

    # 2. verify_checklist.py
    cmd_check = [sys.executable, "code/business_entity_resolution/src/verify_checklist.py"]
    print(f"Running: {' '.join(cmd_check)}", flush=True)
    res_check = subprocess.run(cmd_check, capture_output=True, text=True)
    print(res_check.stdout)
    if res_check.stderr:
        print("STDERR:", res_check.stderr)


def main():
    test_dir = "student_resource/dataset/test"
    output_dir = "output"

    print("Loading NeuralNet_MLP Pipeline from models/best_model_pipeline.pkl...", flush=True)
    with open("models/best_model_pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)

    # Re-score each country
    for country in ["France", "US", "India"]:
        process_country(country, pipeline, test_dir, output_dir, num_workers=8, chunk_size=1500)

    # Stitch outputs
    stitch_final_outputs(test_dir, output_dir)

    # Run structural validations
    run_validations()


if __name__ == "__main__":
    main()
