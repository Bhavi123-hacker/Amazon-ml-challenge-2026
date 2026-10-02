"""Step 4: Blocking Recall & Candidate Pairs Sanity Check across 500 test entities."""

import os
import sys
import pandas as pd
import collections

def main():
    print("=" * 80)
    print("STEP 4: BLOCKING RECALL SANITY CHECK ON CANDIDATE_PAIRS.TSV")
    print("=" * 80)

    cand_path = "BitMinds/output/candidate_pairs.tsv"
    match_path = "BitMinds/output/matching_results.tsv"
    s1_path = "student_resource/dataset/test/test_source1.tsv"

    # 1. Check full distribution of candidate_pairs.tsv
    print(f"Reading {cand_path}...", flush=True)
    total_s1 = 0
    empty_cands = 0
    cand_lengths = collections.Counter()
    total_cands = 0

    with open(cand_path, "r", encoding="utf-8") as f:
        header = next(f)
        for line in f:
            total_s1 += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 1 and parts[1].strip():
                clist = parts[1].split(",")
                k = len(clist)
                cand_lengths[k] += 1
                total_cands += k
            else:
                empty_cands += 1
                cand_lengths[0] += 1

    print(f"\nFull candidate_pairs.tsv Statistics:")
    print(f"  Total Entities              : {total_s1:,}")
    print(f"  Entities with 0 candidates  : {empty_cands:,} ({empty_cands/total_s1*100:.4f}%)")
    print(f"  Total Candidate Pairs       : {total_cands:,}")
    print(f"  Average Candidates / Entity : {total_cands/total_s1:.2f}")
    print("\nCandidate Count Distribution (K):")
    for k in sorted(cand_lengths.keys()):
        cnt = cand_lengths[k]
        print(f"  K = {k:2d} candidates: {cnt:9,d} entities ({cnt/total_s1*100:5.2f}%)")

    # 2. Check Containment (matching_results.tsv subset of candidate_pairs.tsv)
    print("\nChecking Containment (matched_entity_ids <= candidate_entity_ids)...", flush=True)
    violations = 0
    checked = 0
    with open(cand_path, "r", encoding="utf-8") as f_cand, open(match_path, "r", encoding="utf-8") as f_match:
        next(f_cand)
        next(f_match)
        for lc, lm in zip(f_cand, f_match):
            checked += 1
            pc = lc.rstrip("\n").split("\t")
            pm = lm.rstrip("\n").split("\t")
            assert pc[0] == pm[0], f"Row mismatch: {pc[0]} vs {pm[0]}"
            c_set = set(pc[1].split(",")) if len(pc) > 1 and pc[1].strip() else set()
            m_set = set(pm[1].split(",")) if len(pm) > 1 and pm[1].strip() else set()
            if not m_set.issubset(c_set):
                violations += 1

    print(f"  Entities checked: {checked:,}")
    print(f"  Containment violations (matches not in candidates): {violations}")

if __name__ == "__main__":
    main()
