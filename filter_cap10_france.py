"""Filter cap-10 France entities with street_name_sim >= 0.70 and audit a fresh sample.
"""

import os
import sys
import random
import time
import pandas as pd
from rapidfuzz import fuzz

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))
from src.features import extract_street_tokens
from src.normalize import apply_normalization_df


def main():
    t0 = time.time()
    results_path = "output/temp_work/lgbm_clean_results_France_tau_98.tsv"
    print(f"Reading cap-10 entities from {results_path}...", flush=True)

    cap10_entities = []  # list of (sid, [cid1, cid2, ...])
    all_needed_cands = set()
    all_needed_s1 = set()

    with open(results_path, "r", encoding="utf-8") as f:
        header = next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[2].strip():
                matches = [m.strip() for m in parts[2].split(",") if m.strip()]
                if len(matches) == 10:
                    sid = parts[0]
                    cap10_entities.append((sid, matches))
                    all_needed_s1.add(sid)
                    for m in matches:
                        all_needed_cands.add(m)

    n_cap = len(cap10_entities)
    print(f"Found {n_cap:,} cap-10 entities ({len(all_needed_cands):,} unique candidates).", flush=True)

    # Load S1 records
    print("Loading S1 records...", flush=True)
    s1_path = "student_resource/dataset/test/test_source1.tsv"
    s1_rows = []
    with open(s1_path, "r", encoding="utf-8") as f:
        header = next(f).rstrip("\n").split("\t")
        col_idx = {col: i for i, col in enumerate(header)}
        for line in f:
            parts = line.rstrip("\n").split("\t")
            sid = parts[col_idx["entity_id"]]
            if sid in all_needed_s1:
                row_dict = {col: parts[i] if i < len(parts) else "" for col, i in col_idx.items()}
                s1_rows.append(row_dict)

    df_s1 = apply_normalization_df(pd.DataFrame(s1_rows))
    s1_map = {r["entity_id"]: r for r in df_s1.to_dict("records")}
    del s1_rows, df_s1

    # Load candidate records
    print("Loading candidate records...", flush=True)
    cands_path = "output/temp_work/cands_France.tsv"
    c_rows = []
    with open(cands_path, "r", encoding="utf-8") as f:
        header = next(f).rstrip("\n").split("\t")
        col_idx = {col: i for i, col in enumerate(header)}
        for line in f:
            parts = line.rstrip("\n").split("\t")
            cid = parts[col_idx["entity_id"]]
            if cid in all_needed_cands:
                row_dict = {col: parts[i] if i < len(parts) else "" for col, i in col_idx.items()}
                c_rows.append(row_dict)

    df_c = apply_normalization_df(pd.DataFrame(c_rows))
    cand_map = {r["entity_id"]: r for r in df_c.to_dict("records")}
    del c_rows, df_c

    print(f"Loaded records in {time.time() - t0:.1f}s. Pre-computing street token strings...", flush=True)

    s1_st_str_map = {}
    for sid, r in s1_map.items():
        addr = r.get("addr_norm", "")
        city = r.get("city_extracted", "")
        toks = extract_street_tokens(addr, city)
        s1_st_str_map[sid] = " ".join(sorted(toks))

    cand_st_str_map = {}
    for cid, r in cand_map.items():
        addr = r.get("addr_norm", "")
        city = r.get("city_extracted", "")
        toks = extract_street_tokens(addr, city)
        cand_st_str_map[cid] = " ".join(sorted(toks))

    print("Evaluating post-filter (street_name_sim >= 0.70)...", flush=True)

    surviving_pairs = []
    total_evaluated = 0
    total_rejected = 0
    new_counts = {k: 0 for k in range(11)}

    for sid, matches in cap10_entities:
        s1_st = s1_st_str_map.get(sid, "")
        accepted = []
        for cid in matches:
            total_evaluated += 1
            c_st = cand_st_str_map.get(cid, "")
            sim = fuzz.token_sort_ratio(s1_st, c_st) / 100.0 if s1_st and c_st else 0.0
            if sim >= 0.70:
                accepted.append(cid)
                surviving_pairs.append((sid, cid, sim))
            else:
                total_rejected += 1
        new_counts[len(accepted)] += 1

    pct_rej = total_rejected / total_evaluated * 100.0
    print("\n" + "=" * 80, flush=True)
    print("CAP-10 POST-FILTER RESULTS (street_name_sim >= 0.70)", flush=True)
    print("=" * 80, flush=True)
    print(f"Total cap-10 entities evaluated : {n_cap:,}", flush=True)
    print(f"Total match pairs evaluated     : {total_evaluated:,}", flush=True)
    print(f"Pairs rejected (< 0.70)         : {total_rejected:,} ({pct_rej:.2f}%)", flush=True)
    print(f"Pairs retained (>= 0.70)        : {len(surviving_pairs):,} ({100 - pct_rej:.2f}%)", flush=True)

    print("\nNew match-count distribution for former cap-10 entities:", flush=True)
    for k in range(11):
        cnt = new_counts[k]
        pct = cnt / n_cap * 100.0
        print(f"  Matches {k:2d}: {cnt:>6,d} entities ({pct:>5.2f}%)", flush=True)

    # Sample fresh 20 pairs
    random.seed(123)
    fresh_sample = random.sample(surviving_pairs, 20)

    out_file = "output/temp_work/france_cap10_filtered_sample20.txt"
    with open(out_file, "w", encoding="utf-8") as out_f:
        header_str = "\n" + "=" * 90 + "\nFRESH 20-PAIR RANDOM SAMPLE OF SURVIVING CAP-10 PAIRS (street_name_sim >= 0.70)\n" + "=" * 90 + "\n"
        print(header_str, flush=True)
        out_f.write(header_str)

        for i, (sid, cid, sim) in enumerate(fresh_sample, 1):
            r1 = s1_map[sid]
            rc = cand_map[cid]
            lines = [
                f"FILTERED PAIR {i:02d}/20 | street_name_sim: {sim:.3f}",
                f"  S1   : [{sid}] '{r1.get('business_name', '')}' | Addr: '{r1.get('business_address', '')}'",
                f"  Cand : [{cid}] '{rc.get('business_name', '')}' | Addr: '{rc.get('business_address', '')}'",
                "-" * 90 + "\n"
            ]
            block = "\n".join(lines)
            print(block, flush=True)
            out_f.write(block)

    print(f"Saved sample to {out_file}", flush=True)


if __name__ == "__main__":
    main()
