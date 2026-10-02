"""Verify Guard D impact on held-out ground truth (US & India)."""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))
from src.features import FEATURE_COLUMNS
from src.evaluate import compute_macro_f05

def main():
    print("=" * 80)
    print("VERIFYING GUARD D ON HELD-OUT 10K BENCHMARK (US & INDIA)")
    print("=" * 80)

    # 1. Load Model
    model_path = "overnight_mission/models/champion_step3_model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # 2. Load 10k benchmark
    val_path = "overnight_mission/eval/val_10k_benchmark.tsv"
    df_val = pd.read_csv(val_path, sep="\t")

    # Group by S1
    s1_groups = df_val.groupby("s1_id")

    # Baseline: tau=0.65 without Guard D
    # With Guard D: tau=0.65, reject if ns < 45 and asim > 70
    
    # We need features and predictions
    X_val = df_val[FEATURE_COLUMNS].values.astype(np.float32)
    probs = model.predict_proba(X_val)[:, 1]
    df_val["prob"] = probs

    for mode in ["Baseline (tau=0.65)", "With Guard D (tau=0.65, reject if ns<45 & asim>70)"]:
        metrics_by_country = {}
        for country in ["US", "India"]:
            c_df = df_val[df_val["country"] == country]
            pred_dict = {}
            gt_dict = {}

            for sid, group in c_df.groupby("s1_id"):
                gt_set = set(group[group["label"] == 1]["cand_id"])
                gt_dict[sid] = gt_set

                # Predict
                sorted_rows = group.sort_values(by="prob", ascending=False)
                top_cand_name = sorted_rows.iloc[0]["cand_name_norm"] if len(sorted_rows) > 0 else ""
                p1 = sorted_rows.iloc[0]["prob"] if len(sorted_rows) > 0 else 0.0

                accepted = []
                for rank, (_, row) in enumerate(sorted_rows.iterrows()):
                    p = row["prob"]
                    if p < 0.65:
                        continue
                    ns = row["name_token_sort"]
                    asim = row["addr_token_sort"]
                    if "Guard D" in mode and (ns < 45 and asim > 70):
                        continue
                    if rank > 0:
                        sim_top = fuzz.token_sort_ratio(top_cand_name, row["cand_name_norm"])
                        if sim_top < 40 and (p1 - p > 0.08):
                            continue
                    accepted.append(row["cand_id"])
                pred_dict[sid] = set(accepted)

            f05, p, r = compute_macro_f05(gt_dict, pred_dict)
            metrics_by_country[country] = (p, r, f05)

        overall_f = 0.5 * (metrics_by_country["US"][2] + metrics_by_country["India"][2])
        print(f"\n{mode}:")
        print(f"  US    -> Prec: {metrics_by_country['US'][0]*100:.2f}%, Rec: {metrics_by_country['US'][1]*100:.2f}%, F0.5: {metrics_by_country['US'][2]:.6f}")
        print(f"  India -> Prec: {metrics_by_country['India'][0]*100:.2f}%, Rec: {metrics_by_country['India'][1]*100:.2f}%, F0.5: {metrics_by_country['India'][2]:.6f}")
        print(f"  Overall Macro F0.5: {overall_f:.6f}")

if __name__ == "__main__":
    main()
