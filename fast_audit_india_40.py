"""Fast Audit of 40 predicted match pairs for India at tau = 0.98.
"""

import os
import sys
import pickle
import random
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))
from src.features import FEATURE_COLUMNS, extract_pair_features
from src.normalize import apply_normalization_df


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    random.seed(42)
    np.random.seed(42)

    results_path = "output/temp_work/lgbm_clean_results_India_tau_98.tsv"
    print(f"Reading results from {results_path}...", flush=True)

    buckets = {
        "low (1-2)": [],
        "mid (3-6)": [],
        "high (7-9)": [],
        "cap (10)": []
    }

    with open(results_path, "r", encoding="utf-8") as f:
        header = next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            sid = parts[0]
            if len(parts) < 3 or not parts[2].strip():
                continue
            matches = [m.strip() for m in parts[2].split(",") if m.strip()]
            n_m = len(matches)
            if n_m == 0:
                continue

            for m in matches:
                item = (sid, m, n_m)
                if 1 <= n_m <= 2:
                    buckets["low (1-2)"].append(item)
                elif 3 <= n_m <= 6:
                    buckets["mid (3-6)"].append(item)
                elif 7 <= n_m <= 9:
                    buckets["high (7-9)"].append(item)
                elif n_m == 10:
                    buckets["cap (10)"].append(item)

    print("Bucket pair counts:", flush=True)
    for b, items in buckets.items():
        print(f"  {b}: {len(items):,} pairs", flush=True)

    sample_targets = [
        ("low (1-2)", 10),
        ("mid (3-6)", 15),
        ("high (7-9)", 5),
        ("cap (10)", 10)
    ]

    selected_pairs = []
    for b, count in sample_targets:
        sampled = random.sample(buckets[b], count)
        for sid, cid, n_m in sampled:
            selected_pairs.append((b, sid, cid, n_m))

    random.shuffle(selected_pairs)

    needed_s1 = {sid for _, sid, _, _ in selected_pairs}
    needed_cands = {cid for _, _, cid, _ in selected_pairs}
    print(f"Target sample: {len(selected_pairs)} pairs ({len(needed_s1)} unique S1, {len(needed_cands)} unique Cands)", flush=True)

    # Grab raw S1 records
    s1_path = "student_resource/dataset/test/test_source1.tsv"
    s1_records = []
    with open(s1_path, "r", encoding="utf-8") as f:
        header = next(f).rstrip("\n").split("\t")
        col_idx = {col: i for i, col in enumerate(header)}
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts[col_idx["entity_id"]] in needed_s1:
                row_dict = {col: parts[i] if i < len(parts) else "" for col, i in col_idx.items()}
                s1_records.append(row_dict)

    # Grab raw candidate records
    cands_path = "output/temp_work/cands_India.tsv"
    cand_records = []
    with open(cands_path, "r", encoding="utf-8") as f:
        header = next(f).rstrip("\n").split("\t")
        col_idx = {col: i for i, col in enumerate(header)}
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts[col_idx["entity_id"]] in needed_cands:
                row_dict = {col: parts[i] if i < len(parts) else "" for col, i in col_idx.items()}
                cand_records.append(row_dict)

    print(f"Loaded {len(s1_records)} raw S1 and {len(cand_records)} raw cand records. Normalizing...", flush=True)

    df_s1_raw = pd.DataFrame(s1_records)
    df_s1_norm = apply_normalization_df(df_s1_raw.copy())

    df_cands_raw = pd.DataFrame(cand_records)
    df_cands_norm = apply_normalization_df(df_cands_raw.copy())

    s1_raw_map = {r["entity_id"]: r for r in df_s1_raw.to_dict("records")}
    s1_norm_map = {r["entity_id"]: r for r in df_s1_norm.to_dict("records")}

    cand_raw_map = {r["entity_id"]: r for r in df_cands_raw.to_dict("records")}
    cand_norm_map = {r["entity_id"]: r for r in df_cands_norm.to_dict("records")}

    model_path = "models/lgbm_champion.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    out_file = "output/temp_work/india_audit_40_results.txt"
    with open(out_file, "w", encoding="utf-8") as out_f:
        header_str = "=" * 90 + "\nINDIA 40-PAIR RANDOM AUDIT AT TAU = 0.98\n" + "=" * 90 + "\n"
        print(header_str, flush=True)
        out_f.write(header_str)

        for i, (bucket_name, sid, cid, n_m) in enumerate(selected_pairs, 1):
            r1_raw = s1_raw_map[sid]
            rc_raw = cand_raw_map[cid]

            r1_norm = s1_norm_map[sid]
            rc_norm = cand_norm_map[cid]

            feats = extract_pair_features(r1_norm, rc_norm)
            X = np.array([[feats[col] for col in FEATURE_COLUMNS]], dtype=np.float32)
            prob = model.predict_proba(X)[0, 1]

            lines = [
                f"PAIR {i:02d}/40 | Bucket: {bucket_name} (Entity Matches: {n_m}) | Prob: {prob:.6f}",
                f"  S1   : [{sid}] '{r1_raw.get('business_name', '')}' | Addr: '{r1_raw.get('business_address', '')}' | City: '{r1_norm.get('city_extracted', '')}' | PIN: '{r1_norm.get('pin_extracted', '')}'",
                f"  Cand : [{cid}] '{rc_raw.get('business_name', '')}' | Addr: '{rc_raw.get('business_address', '')}' | City: '{rc_norm.get('city_extracted', '')}' | PIN: '{rc_norm.get('pin_extracted', '')}'",
                "-" * 90 + "\n"
            ]
            block = "\n".join(lines)
            print(block, flush=True)
            out_f.write(block)

    print(f"Results saved to {out_file}", flush=True)


if __name__ == "__main__":
    main()
