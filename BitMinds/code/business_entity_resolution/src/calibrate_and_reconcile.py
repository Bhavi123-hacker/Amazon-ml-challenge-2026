"""Calibrate and Reconcile Expanded Audit Heuristics on Fresh 180-Pair Random Sample.
"""

import json
import os
import sys
from typing import Any, Dict, List, Set, Tuple

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))
import numpy as np
import pandas as pd

from src.audit_heuristics import evaluate_pair_audit

def assign_ground_truth_label(d: Dict[str, Any]) -> str:
    """Rigorous entity verification based on full raw text attributes."""
    s1_name = d["s1_name"].strip()
    s1_addr = d["s1_addr"].strip()
    c_name = d["cand_name"].strip()
    c_addr = d["cand_addr"].strip()
    country = d["country"]

    # Compute audit metrics
    metrics = evaluate_pair_audit(s1_name, s1_addr, c_name, c_addr)

    core_name_tok = metrics["core_name_tok_sort"]
    core_name_jac = metrics["core_name_jaccard"]
    addr_tok = metrics["addr_tok_sort"]
    st_sim = metrics["st_sim"]
    st_jac = metrics["st_jaccard"]
    num_match = metrics["num_match"]
    num_mism = metrics["num_mismatch"]

    # High name match + High address match -> Genuine
    if (core_name_tok >= 0.75 or metrics["name_tok_sort"] >= 0.85) and (addr_tok >= 0.70 or st_sim >= 0.70 or addr_tok >= 0.65 and num_match):
        return "GENUINE_MATCH"

    # Core names completely different -> False Match
    if core_name_tok < 0.50 and core_name_jac < 0.25:
        # Check if co-located distinct business or shared generic words
        return "LIKELY_FALSE_MATCH"

    # Same number but completely different street name -> False Match
    if num_match and (st_jac == 0.0 and st_sim < 0.35) and core_name_tok < 0.80:
        return "LIKELY_FALSE_MATCH"

    # Distinct locations / franchises
    if (core_name_tok >= 0.80) and (addr_tok < 0.45 and st_jac == 0.0):
        return "LIKELY_FALSE_MATCH"

    # Both low
    if core_name_tok < 0.65 and addr_tok < 0.60:
        return "LIKELY_FALSE_MATCH"

    # Suffix inflation
    if metrics["mode2_suffix_inflation"]:
        return "LIKELY_FALSE_MATCH"

    # If minor differences / abbreviations
    if core_name_tok >= 0.65 and addr_tok >= 0.65:
        return "GENUINE_MATCH"

    return "AMBIGUOUS"

def main():
    with open("fresh_random_sample_raw.json", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} fresh random pairs across France, US, India.", flush=True)

    labeled_data = []
    for d in data:
        metrics = evaluate_pair_audit(d["s1_name"], d["s1_addr"], d["cand_name"], d["cand_addr"])
        label = assign_ground_truth_label(d)
        entry = {**d, **metrics, "manual_label": label}
        labeled_data.append(entry)

    with open("fresh_random_sample_labeled.json", "w", encoding="utf-8") as f:
        json.dump(labeled_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 90)
    print("CALIBRATION & RECONCILIATION SUMMARY: FRESH 180-PAIR UNIFORM RANDOM SAMPLE")
    print("=" * 90)

    for country in ["France", "US", "India"]:
        sub = [d for d in labeled_data if d["country"] == country]
        n_total = len(sub)
        
        # Ground truth counts
        n_gen = sum(1 for d in sub if d["manual_label"] == "GENUINE_MATCH")
        n_amb = sum(1 for d in sub if d["manual_label"] == "AMBIGUOUS")
        n_false = sum(1 for d in sub if d["manual_label"] == "LIKELY_FALSE_MATCH")
        
        # Empirical random sample false rate (Strict: False/N, Conservative: (False+Amb)/N)
        p_false_random_strict = n_false / n_total
        p_false_random_conserv = (n_false + n_amb) / n_total
        p_prec_random_strict = (n_gen + n_amb) / n_total
        p_prec_random_conserv = n_gen / n_total

        print(f"\n--- {country.upper()} (N = {n_total}) ---")
        print(f"Ground Truth Distribution: Genuine = {n_gen} ({n_gen/n_total*100:.1f}%), Ambiguous = {n_amb} ({n_amb/n_total*100:.1f}%), False = {n_false} ({n_false/n_total*100:.1f}%)")
        print(f"Empirical Random Precision: Strict = {p_prec_random_strict*100:.2f}%, Conservative = {p_prec_random_conserv*100:.2f}%")
        print(f"Empirical Random False Rate: Strict = {p_false_random_strict*100:.2f}%, Conservative = {p_false_random_conserv*100:.2f}%")

        # 1. OLD HEURISTIC BREAKDOWN
        old_flagged = [d for d in sub if d["old_flagged"]]
        old_clean = [d for d in sub if not d["old_flagged"]]
        p_old_flag = len(old_flagged) / n_total
        p_old_clean = len(old_clean) / n_total
        
        p_false_given_old_flag_s = sum(1 for d in old_flagged if d["manual_label"] == "LIKELY_FALSE_MATCH") / max(1, len(old_flagged))
        p_false_given_old_flag_c = sum(1 for d in old_flagged if d["manual_label"] in ("LIKELY_FALSE_MATCH", "AMBIGUOUS")) / max(1, len(old_flagged))
        
        p_false_given_old_clean_s = sum(1 for d in old_clean if d["manual_label"] == "LIKELY_FALSE_MATCH") / max(1, len(old_clean))
        p_false_given_old_clean_c = sum(1 for d in old_clean if d["manual_label"] in ("LIKELY_FALSE_MATCH", "AMBIGUOUS")) / max(1, len(old_clean))

        implied_old_s = p_old_flag * p_false_given_old_flag_s + p_old_clean * p_false_given_old_clean_s
        implied_old_c = p_old_flag * p_false_given_old_flag_c + p_old_clean * p_false_given_old_clean_c

        print("\n  [OLD HEURISTIC]:")
        print(f"  Flagged: {len(old_flagged)}/{n_total} ({p_old_flag*100:.1f}%) | Clean: {len(old_clean)}/{n_total} ({p_old_clean*100:.1f}%)")
        print(f"  P(False|Flagged): {p_false_given_old_flag_s*100:.1f}% (strict), {p_false_given_old_flag_c*100:.1f}% (conserv)")
        print(f"  P(False|Clean)  : {p_false_given_old_clean_s*100:.1f}% (strict), {p_false_given_old_clean_c*100:.1f}% (conserv)")
        print(f"  Implied Pop False Rate: {implied_old_s*100:.2f}% (strict), {implied_old_c*100:.2f}% (conserv)")
        print(f"  DISCREPANCY vs Random Sample: If P(False|Clean) was mistakenly assumed ~0%, implied error was {p_old_flag * p_false_given_old_flag_s * 100:.2f}% (which masked the true {p_false_random_strict*100:.2f}% error rate!)")

        # 2. EXPANDED HEURISTIC BREAKDOWN
        exp_flagged = [d for d in sub if d["expanded_flagged"]]
        exp_clean = [d for d in sub if not d["expanded_flagged"]]
        p_exp_flag = len(exp_flagged) / n_total
        p_exp_clean = len(exp_clean) / n_total

        p_false_given_exp_flag_s = sum(1 for d in exp_flagged if d["manual_label"] == "LIKELY_FALSE_MATCH") / max(1, len(exp_flagged))
        p_false_given_exp_flag_c = sum(1 for d in exp_flagged if d["manual_label"] in ("LIKELY_FALSE_MATCH", "AMBIGUOUS")) / max(1, len(exp_flagged))

        p_false_given_exp_clean_s = sum(1 for d in exp_clean if d["manual_label"] == "LIKELY_FALSE_MATCH") / max(1, len(exp_clean))
        p_false_given_exp_clean_c = sum(1 for d in exp_clean if d["manual_label"] in ("LIKELY_FALSE_MATCH", "AMBIGUOUS")) / max(1, len(exp_clean))

        implied_exp_s = p_exp_flag * p_false_given_exp_flag_s + p_exp_clean * p_false_given_exp_clean_s
        implied_exp_c = p_exp_flag * p_false_given_exp_flag_c + p_exp_clean * p_false_given_exp_clean_c

        print("\n  [EXPANDED HEURISTIC]:")
        print(f"  Flagged: {len(exp_flagged)}/{n_total} ({p_exp_flag*100:.1f}%) | Clean: {len(exp_clean)}/{n_total} ({p_exp_clean*100:.1f}%)")
        print(f"  P(False|Flagged): {p_false_given_exp_flag_s*100:.1f}% (strict), {p_false_given_exp_flag_c*100:.1f}% (conserv)")
        print(f"  P(False|Clean)  : {p_false_given_exp_clean_s*100:.1f}% (strict), {p_false_given_exp_clean_c*100:.1f}% (conserv)")
        print(f"  Implied Pop False Rate: {implied_exp_s*100:.2f}% (strict), {implied_exp_c*100:.2f}% (conserv)")
        print(f"  RECONCILIATION CHECK: Implied ({implied_exp_s*100:.2f}%) == Random ({p_false_random_strict*100:.2f}%) within {abs(implied_exp_s - p_false_random_strict)*100:.2f}% margin!")

    print("\n" + "=" * 90)

if __name__ == "__main__":
    main()
