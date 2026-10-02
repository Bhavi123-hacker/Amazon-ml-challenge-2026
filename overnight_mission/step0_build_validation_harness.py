"""Step 0: Build the Honest Validation Harness.

Stratified 80/20 train/held-out split of labeled US + India training data.
Stratified by country and match count bin (0, 1, 2, 3, 4, 5, 6+).
Guarantees 0 entity overlap between training and held-out evaluation.
Also exports a rapid-benchmarking 10,000 entity subset of the held-out set
for rapid iteration during Steps 1-3.
"""

import os
import sys
import numpy as np
import pandas as pd

np.random.seed(42)

def main():
    print("=== STEP 0: BUILDING HONEST VALIDATION HARNESS ===", flush=True)

    # 1. Load S1 training records
    s1_path = "student_resource/dataset/train/train_source1.tsv"
    print(f"Loading {s1_path} ...", flush=True)
    df_s1 = pd.read_csv(s1_path, sep="\t", usecols=["entity_id", "country"], keep_default_na=False)
    n_total = len(df_s1)
    print(f"Total Source 1 entities: {n_total:,}", flush=True)

    # 2. Load ground truth labels
    gt_path = "student_resource/dataset/train/train_ground_truth.tsv"
    print(f"Loading {gt_path} ...", flush=True)
    df_gt = pd.read_csv(gt_path, sep="\t", keep_default_na=False)
    gt_map = dict(zip(df_gt["source1_entity_id"], df_gt["matched_entity_ids"]))

    # 3. Compute match counts and stratification bins
    print("Computing match counts and stratification strata...", flush=True)
    match_counts = []
    for sid in df_s1["entity_id"]:
        m_raw = gt_map.get(sid, "").strip()
        cnt = len([x for x in m_raw.split(",") if x.strip()]) if m_raw else 0
        match_counts.append(cnt)

    df_s1["match_count"] = match_counts
    df_s1["strata"] = df_s1["country"] + "_bin_" + df_s1["match_count"].clip(upper=6).astype(str)

    print("\nPopulation Strata Breakdown:")
    for strata, cnt in df_s1["strata"].value_counts().sort_index().items():
        print(f"  {strata:<18}: {cnt:>9,d} ({cnt/n_total*100:>5.2f}%)")

    # 4. Perform Stratified 80/20 Split
    held_out_list = []
    train_list = []

    for strata_name, group in df_s1.groupby("strata"):
        n = len(group)
        n_held = int(round(n * 0.20))
        shuffled_idx = np.random.permutation(group.index)
        held_idx = shuffled_idx[:n_held]
        train_idx = shuffled_idx[n_held:]
        held_out_list.append(df_s1.loc[held_idx])
        train_list.append(df_s1.loc[train_idx])

    df_held = pd.concat(held_out_list, ignore_index=True)
    df_train = pd.concat(train_list, ignore_index=True)

    print("\n" + "=" * 70)
    print("SPLIT SUMMARY")
    print("=" * 70)
    print(f"Train split entities    : {len(df_train):,} ({len(df_train)/n_total*100:.2f}%)")
    print(f"Held-out split entities : {len(df_held):,} ({len(df_held)/n_total*100:.2f}%)")

    # Verify zero entity overlap
    overlap = set(df_train["entity_id"]) & set(df_held["entity_id"])
    print(f"Overlap between Train and Held-out: {len(overlap)} entities")
    assert len(overlap) == 0, f"FATAL: Overlap detected! {len(overlap)} entities shared."

    # 5. Build Fast Benchmark Set (10,000 entities from held-out set)
    val_10k_list = []
    for strata_name, group in df_held.groupby("strata"):
        n_sub = int(round(len(group) / len(df_held) * 10000))
        shuffled_idx = np.random.permutation(group.index)
        val_10k_list.append(df_held.loc[shuffled_idx[:n_sub]])

    df_val_10k = pd.concat(val_10k_list, ignore_index=True)
    if len(df_val_10k) > 10000:
        df_val_10k = df_val_10k.sample(n=10000, random_state=42).reset_index(drop=True)

    val_overlap = set(df_val_10k["entity_id"]) & set(df_train["entity_id"])
    print(f"Rapid Benchmark Val 10k entities : {len(df_val_10k):,}")
    print(f"Overlap Val 10k with Train       : {len(val_overlap)} entities")
    assert len(val_overlap) == 0, "FATAL: Val 10k overlaps with Train!"

    # 6. Save splits
    out_dir = "overnight_mission/eval"
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train_split_ids.parquet")
    held_path = os.path.join(out_dir, "held_out_split_ids.parquet")
    val10k_path = os.path.join(out_dir, "val_10k_benchmark.tsv")

    df_train[["entity_id", "country", "match_count"]].to_parquet(train_path, index=False)
    df_held[["entity_id", "country", "match_count"]].to_parquet(held_path, index=False)
    df_val_10k[["entity_id", "country", "match_count"]].to_csv(val10k_path, sep="\t", index=False)

    print(f"\nSaved splits to:")
    print(f"  - {train_path} ({os.path.getsize(train_path):,} bytes)")
    print(f"  - {held_path} ({os.path.getsize(held_path):,} bytes)")
    print(f"  - {val10k_path} ({os.path.getsize(val10k_path):,} bytes)")
    print("STEP 0 COMPLETE: Honest validation harness locked.", flush=True)

if __name__ == "__main__":
    main()
