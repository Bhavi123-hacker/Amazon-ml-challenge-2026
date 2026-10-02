"""Evaluate candidate reduction and match retention across the full test set for K in [8, 10, 12, 14, 16, 20, 25].
"""

import os
import sys
import pandas as pd
import numpy as np

def main():
    print("=== Evaluating K Cutoff Impact Across Full Test Set (1.73M Entities) ===", flush=True)
    countries = ["France", "US", "India"]
    k_values = [8, 10, 12, 14, 16, 20, 25]

    for country in countries:
        res_path = f"output/temp_work/results_{country}.tsv"
        print(f"\nProfiling {country} from {res_path}...", flush=True)
        
        # Stats per K
        cand_counts = {k: 0 for k in k_values}
        match_counts = {k: 0 for k in k_values}
        entities_with_match = {k: 0 for k in k_values}
        total_entities = 0

        with open(res_path, "r", encoding="utf-8") as f:
            header = next(f)
            for line in f:
                total_entities += 1
                parts = line.rstrip("\n").split("\t")
                cands = [x.strip() for x in parts[1].split(",") if x.strip()] if len(parts) > 1 else []
                matches = set(x.strip() for x in parts[2].split(",") if x.strip()) if len(parts) > 2 else set()

                for k in k_values:
                    sub_cands = cands[:k]
                    cand_counts[k] += len(sub_cands)
                    # matches that survive cutoff
                    surviving_matches = matches & set(sub_cands)
                    match_counts[k] += len(surviving_matches)
                    if surviving_matches:
                        entities_with_match[k] += 1

        print(f"{country} Total S1 Entities: {total_entities:,}")
        base_cands = cand_counts[25]
        base_matches = match_counts[25]
        base_match_rate = entities_with_match[25] / total_entities * 100.0

        print(f"{'Cutoff (K)':<12} | {'Avg Cands':<10} | {'Cand Reduct %':<14} | {'Match Rate':<12} | {'Matches Retained':<18} | {'Match Retain %':<14}")
        print("-" * 88)
        for k in k_values:
            avg_c = cand_counts[k] / total_entities
            reduct_pct = (1.0 - cand_counts[k] / base_cands) * 100.0
            m_rate = entities_with_match[k] / total_entities * 100.0
            m_cnt = match_counts[k]
            m_pct = m_cnt / base_matches * 100.0
            print(f"Top-{k:<8} | {avg_c:<10.2f} | {reduct_pct:<13.1f}% | {m_rate:<11.2f}% | {m_cnt:<18,} | {m_pct:<13.2f}%")

if __name__ == "__main__":
    main()
