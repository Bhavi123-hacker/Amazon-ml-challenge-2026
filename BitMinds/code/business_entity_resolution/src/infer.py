"""Optimized High-Throughput Inference Engine (Stage 5).

Key architectural optimizations:
1. Frequency-capped multi-channel blocking index (skips noisy lists > 300 candidates).
2. Fallback prefix 4-gram scans (only when full token hits < 5).
3. ThreadPoolExecutor multi-core parallelism (shares memory, 0 IPC pickling overhead).
4. Independent country runner (--country France | US | India | all).
5. Immediate disk streaming to output/temp_work/results_{country}.tsv.
6. Final stitcher producing exact test_source1-ordered output files.
"""

import argparse
import gc
import json
import os
import pickle
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from metaphone import doublemetaphone
from rapidfuzz import fuzz

from src.blocking import BlockingIndex, get_acronym, get_first_two_tokens
from src.features import FEATURE_COLUMNS, extract_pair_features
from src.normalize import apply_normalization_df


def partition_candidate_sources(
    test_dir: str,
    temp_dir: str,
    countries: List[str] = ["France", "US", "India"],
    chunk_size: int = 500000,
) -> Dict[str, str]:
    """Partition test_source2 and test_source3 into per-country TSV files."""
    os.makedirs(temp_dir, exist_ok=True)
    partition_paths = {c: os.path.join(temp_dir, f"cands_{c}.tsv") for c in countries}

    all_exist = all(os.path.isfile(p) and os.path.getsize(p) > 1000 for p in partition_paths.values())
    if all_exist:
        print("Existing per-country candidate files found. Skipping partitioning.", flush=True)
        return partition_paths

    print("Partitioning candidate files by country for memory-safe streaming...", flush=True)
    t0 = time.time()

    writers = {}
    for c, p in partition_paths.items():
        writers[c] = open(p, "w", encoding="utf-8")
        writers[c].write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")

    for src_name in ["test_source2.tsv", "test_source3.tsv"]:
        src_path = os.path.join(test_dir, src_name)
        print(f"Streaming {src_path} in chunks of {chunk_size:,}...", flush=True)
        for chunk in pd.read_csv(src_path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunk_size):
            for c in countries:
                sub = chunk[chunk["country"] == c]
                if len(sub) > 0:
                    sub[["entity_id", "business_name", "business_address", "country"]].to_csv(
                        writers[c], sep="\t", index=False, header=False
                    )

    for f in writers.values():
        f.close()

    print(f"Partitioning completed in {time.time() - t0:.2f}s", flush=True)
    return partition_paths


def process_sub_chunk(
    s1_chunk: List[dict],
    indexer: BlockingIndex,
    model,
    cand_records_map: dict,
    country: str,
    tau: float,
    top_k: int = 25,
) -> List[Tuple[str, str, str]]:
    """Process a chunk of S1 entities using the fast inverted index & model."""
    results = []
    batch_pairs = []
    batch_meta = []
    batch_cand_map = {}

    for s1_rec in s1_chunk:
        s1_id = s1_rec["entity_id"]
        n_norm = s1_rec["name_norm"]
        pin = s1_rec["pin_extracted"]
        st_num = s1_rec["street_number"]
        addr_norm = s1_rec["addr_norm"]
        city = s1_rec["city_extracted"]

        cand_counts = Counter()

        # 1. First 2 tokens
        k_tok2 = get_first_two_tokens(n_norm)
        if k_tok2:
            for idx in indexer.idx_tok2[country].get(k_tok2, []):
                cand_counts[idx] += 12

        # 2. Phonetic
        p1, _ = doublemetaphone(n_norm.split()[0]) if n_norm else ("", "")
        if p1:
            p_list = indexer.idx_phone[country].get(p1, [])
            if len(p_list) <= 300:
                for idx in p_list:
                    cand_counts[idx] += 6

        # 3. Postal + token prefix
        if pin and n_norm:
            for idx in indexer.idx_postal[country].get(f"{pin}_{n_norm.split()[0][:3]}", []):
                cand_counts[idx] += 8

        # 4. Address Street Number + Street tokens / PIN / City
        if st_num:
            addr_tokens = [t for t in addr_norm.split() if t != st_num and len(t) > 2]
            for t in addr_tokens[:3]:
                for idx in indexer.idx_addr[country].get(f"{st_num}_{t}", []):
                    cand_counts[idx] += 10
            if pin:
                for idx in indexer.idx_addr[country].get(f"{pin}_{st_num}", []):
                    cand_counts[idx] += 10

        # 5. Significant tokens & prefix 4-grams (with fallback)
        toks = [t for t in n_norm.split() if len(t) >= 4]
        for tok in toks:
            s_list = indexer.idx_sigtoken[country].get(tok, [])
            if len(s_list) <= 300:
                for idx in s_list:
                    cand_counts[idx] += 5
            if len(s_list) < 5:
                p_list = indexer.idx_prefix[country].get(tok[:4], [])
                if len(p_list) <= 100:
                    for idx in p_list:
                        cand_counts[idx] += 2

        # 6. Acronyms
        if len(toks) >= 2:
            acr = get_acronym(toks)
            if acr:
                for idx in indexer.idx_sigtoken[country].get(acr, []):
                    cand_counts[idx] += 6
        elif len(toks) == 1 and 2 <= len(toks[0]) <= 4:
            for idx in indexer.idx_sigtoken[country].get(toks[0], []):
                cand_counts[idx] += 4

        if not cand_counts:
            batch_cand_map[s1_id] = []
            continue

        pre_selected = [idx for idx, _ in cand_counts.most_common(max(top_k * 2, 50))]
        scored = []
        for idx in pre_selected:
            c_rec = indexer.cand_records[idx]
            s_name = fuzz.token_set_ratio(n_norm, c_rec["name_norm"])
            s_addr = fuzz.token_set_ratio(addr_norm, c_rec["addr_norm"])
            score = max(s_name, s_addr) + 0.3 * min(s_name, s_addr)
            scored.append((c_rec["entity_id"], float(score)))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_cands = scored[:top_k]
        c_ids = [cid for cid, _ in top_cands]
        batch_cand_map[s1_id] = c_ids

        for rank, (cand_id, b_score) in enumerate(top_cands):
            cand_rec = cand_records_map.get(cand_id)
            if not cand_rec:
                continue
            feats = extract_pair_features(s1_rec, cand_rec, rank=rank, blocking_score=b_score)
            batch_pairs.append([feats[col] for col in FEATURE_COLUMNS])
            batch_meta.append((s1_id, cand_id))

    if batch_pairs:
        X_batch = np.array(batch_pairs, dtype=np.float32)
        probs = model.predict_proba(X_batch)[:, 1]

        matches_map = {s1_id: [] for s1_id in batch_cand_map}
        for (s1_id, cand_id), prob in zip(batch_meta, probs):
            if prob >= tau:
                matches_map[s1_id].append((cand_id, float(prob)))

        for s1_id, cand_list in batch_cand_map.items():
            cand_set = set(cand_list)
            raw_m = matches_map.get(s1_id, [])
            raw_m.sort(key=lambda x: x[1], reverse=True)
            seen = set()
            clean = []
            for cid, _ in raw_m:
                if cid not in seen and cid in cand_set:
                    seen.add(cid)
                    clean.append(cid)
            results.append((s1_id, ",".join(cand_list), ",".join(clean)))
    else:
        for s1_id, cand_list in batch_cand_map.items():
            results.append((s1_id, ",".join(cand_list), ""))

    return results


def run_country_inference(
    country: str,
    test_dir: str,
    output_dir: str,
    model_dir: str = "models",
    tau_override: float = None,
    top_k: int = 25,
    num_workers: int = 8,
    chunk_size: int = 1000,
) -> str:
    """Run optimized inference for a single country and write output/temp_work/results_{country}.tsv."""
    temp_dir = os.path.join(output_dir, "temp_work")
    os.makedirs(temp_dir, exist_ok=True)
    out_tsv = os.path.join(temp_dir, f"results_{country}.tsv")

    print(f"\n=======================================================", flush=True)
    print(f"STARTING OPTIMIZED INFERENCE: {country.upper()}", flush=True)
    print(f"=======================================================", flush=True)
    t_start = time.time()

    # 1. Load model & config
    model_path = os.path.join(model_dir, "best_model.pkl")
    config_path = os.path.join(model_dir, "config.json")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(config_path, "r") as f:
        config = json.load(f)
    tau = tau_override if tau_override is not None else config.get("optimal_threshold", 0.86)
    print(f"Using Decision Threshold tau for {country}: {tau:.3f}", flush=True)

    # 2. Load and normalize S1 for this country
    print(f"Loading {country} Source 1 records...", flush=True)
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
    df_s1_country = df_s1[df_s1["country"] == country].copy()
    n_s1 = len(df_s1_country)
    print(f"Loaded {n_s1:,} S1 entities for {country}", flush=True)

    t_norm = time.time()
    df_s1_country = apply_normalization_df(df_s1_country)
    s1_records = df_s1_country.to_dict("records")
    print(f"Normalized S1 records in {time.time() - t_norm:.2f}s", flush=True)

    # 3. Load candidates for this country
    cands_file = os.path.join(temp_dir, f"cands_{country}.tsv")
    print(f"Loading {country} candidates from {cands_file}...", flush=True)
    t_load = time.time()
    df_cands = pd.read_csv(cands_file, sep="\t", dtype=str, keep_default_na=False)
    n_cands = len(df_cands)
    print(f"Loaded {n_cands:,} candidates in {time.time() - t_load:.2f}s", flush=True)

    t_norm = time.time()
    df_cands = apply_normalization_df(df_cands)
    print(f"Normalized candidates in {time.time() - t_norm:.2f}s", flush=True)

    # 4. Build index
    print(f"Building blocking index for {country}...", flush=True)
    t_idx = time.time()
    indexer = BlockingIndex()
    indexer.build(df_cands)
    print(f"Index built in {time.time() - t_idx:.2f}s", flush=True)

    cand_records_map = {rec["entity_id"]: rec for rec in indexer.cand_records}

    # 5. Process in parallel chunks using ThreadPoolExecutor
    chunks = [s1_records[i:i + chunk_size] for i in range(0, n_s1, chunk_size)]
    print(f"Processing {n_s1:,} entities across {len(chunks)} chunks using {num_workers} threads...", flush=True)

    matched_count = 0
    total_processed = 0

    with open(out_tsv, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tcandidate_entity_ids\tmatched_entity_ids\n")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Helper to execute task
            def task(chunk):
                return process_sub_chunk(
                    chunk, indexer, model, cand_records_map, country, tau, top_k=top_k
                )

            for chunk_res in executor.map(task, chunks):
                for s1_id, cand_str, match_str in chunk_res:
                    f_out.write(f"{s1_id}\t{cand_str}\t{match_str}\n")
                    if match_str:
                        matched_count += 1
                total_processed += len(chunk_res)
                if total_processed % 50000 == 0 or total_processed == n_s1:
                    speed = total_processed / max(1.0, time.time() - t_start)
                    pct = total_processed / n_s1 * 100
                    print(f"  [{country}] {total_processed:,}/{n_s1:,} ({pct:.1f}%) | Speed: {speed:.1f} ent/s", flush=True)

    elapsed = time.time() - t_start
    print(f"\nCompleted {country} in {elapsed:.2f}s ({elapsed/60:.2f} min) | Speed: {n_s1/max(1.0, elapsed):.1f} ent/s", flush=True)
    print(f"Matched entities: {matched_count:,} ({matched_count/n_s1:.2%}) | Singletons: {n_s1 - matched_count:,} ({(n_s1 - matched_count)/n_s1:.2%})", flush=True)
    print(f"Results saved to: {out_tsv}", flush=True)

    del df_s1, df_s1_country, df_cands, indexer, cand_records_map
    gc.collect()

    return out_tsv


def stitch_final_outputs(
    test_dir: str,
    output_dir: str,
    countries: List[str] = ["France", "US", "India"],
):
    """Combine results_{country}.tsv into final submission files ordered by test_source1.tsv."""
    temp_dir = os.path.join(output_dir, "temp_work")
    cand_path = os.path.join(output_dir, "candidate_pairs.tsv")
    match_path = os.path.join(output_dir, "matching_results.tsv")

    print("\n--- Stitching Final Submission Output Files ---", flush=True)
    t0 = time.time()

    # Load country results into memory map: s1_id -> (cand_str, match_str)
    res_map = {}
    for country in countries:
        tsv_path = os.path.join(temp_dir, f"results_{country}.tsv")
        print(f"Reading {tsv_path}...", flush=True)
        with open(tsv_path, "r", encoding="utf-8") as f:
            next(f)  # skip header
            for line in f:
                parts = line.rstrip("\n").split("\t")
                s1_id = parts[0]
                c_str = parts[1] if len(parts) > 1 else ""
                m_str = parts[2] if len(parts) > 2 else ""
                res_map[s1_id] = (c_str, m_str)

    # Read original test_source1.tsv order
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    print(f"Streaming final outputs in exact order of {s1_path}...", flush=True)

    total = 0
    matched = 0
    singletons = 0

    with open(cand_path, "w", encoding="utf-8") as f_cand, open(match_path, "w", encoding="utf-8") as f_match:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        f_match.write("source1_entity_id\tmatched_entity_ids\n")

        with open(s1_path, "r", encoding="utf-8") as f_s1:
            next(f_s1)  # skip header
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

    print(f"\nFinal Outputs Generated in {time.time() - t0:.2f}s:")
    print(f"  - {match_path} ({os.path.getsize(match_path):,} bytes)")
    print(f"  - {cand_path} ({os.path.getsize(cand_path):,} bytes)")
    print(f"Total entities: {total:,} | Matched: {matched:,} ({matched/total:.2%}) | Singletons: {singletons:,} ({singletons/total:.2%})")


def main():
    parser = argparse.ArgumentParser(description="Run optimized high-throughput inference.")
    parser.add_argument("--test-dir", default="d:/amazolml/student_resource/dataset/test", help="Test dataset directory")
    parser.add_argument("--output-dir", default="d:/amazolml/output", help="Output directory")
    parser.add_argument("--model-dir", default="models", help="Model directory")
    parser.add_argument("--country", default="all", choices=["all", "France", "US", "India", "stitch"], help="Country to process")
    parser.add_argument("--top-k", type=int, default=25, help="Top K candidates")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker threads")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size per thread")
    parser.add_argument("--tau", type=float, default=None, help="Decision threshold override")
    args = parser.parse_args()

    temp_dir = os.path.join(args.output_dir, "temp_work")
    partition_candidate_sources(args.test_dir, temp_dir)

    if args.country == "stitch":
        stitch_final_outputs(args.test_dir, args.output_dir)
        return

    countries = ["France", "US", "India"] if args.country == "all" else [args.country]

    for c in countries:
        run_country_inference(
            country=c,
            test_dir=args.test_dir,
            output_dir=args.output_dir,
            model_dir=args.model_dir,
            tau_override=args.tau,
            top_k=args.top_k,
            num_workers=args.workers,
            chunk_size=args.chunk_size,
        )

    if args.country == "all":
        stitch_final_outputs(args.test_dir, args.output_dir)


if __name__ == "__main__":
    main()
