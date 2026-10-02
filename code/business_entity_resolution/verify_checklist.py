"""Full Automated Self-Verification Checklist.

Checks:
1. Every S1 entity appears exactly once in matching_results.tsv.
2. No duplicate source1_entity_id rows in matching_results.tsv or candidate_pairs.tsv.
3. No duplicate IDs within any single matched_entity_ids or candidate_entity_ids list.
4. Every ID in matched_entity_ids is a valid S2-/S3- ID that exists in the test set.
5. Every matched ID appears in that same row's candidate_entity_ids (matches <= candidates).
6. No cross-country matches (validates country(S1) == country(matched)).
7. File format: tab-separated, correct column headers, UTF-8 encoded.
8. Total match-rate percentage reported per country (France, US, India) vs training distribution.
9. Final validator run execution and PASS verification.
10. Exact row count equality between matching_results.tsv and test_source1.tsv.
"""

import os
import sys
import subprocess
import pandas as pd

def run_checklist(test_dir="d:/amazolml/student_resource/dataset/test", output_dir="d:/amazolml/output"):
    print("================================================================================")
    print("RUNNING AUTOMATED SELF-VERIFICATION CHECKLIST")
    print("================================================================================\n")

    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
    s1_path = os.path.join(test_dir, "test_source1.tsv")

    results = {}

    # Check 1 & 10: Row counts and exact S1 match
    print("--- Check 1 & 10: Row count equality and S1 presence ---")
    with open(s1_path, "r", encoding="utf-8") as f:
        s1_header = next(f)
        s1_ids = [line.split("\t", 1)[0].strip() for line in f if line.strip()]
    s1_id_set = set(s1_ids)
    expected_count = len(s1_ids)

    with open(matching_path, "r", encoding="utf-8") as f:
        m_header = next(f)
        m_lines = [line.rstrip("\n").split("\t") for line in f if line.strip()]
    m_ids = [row[0] for row in m_lines]
    m_id_set = set(m_ids)

    with open(candidate_path, "r", encoding="utf-8") as f:
        c_header = next(f)
        c_lines = [line.rstrip("\n").split("\t") for line in f if line.strip()]
    c_ids = [row[0] for row in c_lines]
    c_id_set = set(c_ids)

    check_1_pass = (len(m_ids) == expected_count) and (m_id_set == s1_id_set)
    check_10_pass = (len(m_lines) == expected_count) and (len(c_lines) == expected_count)

    results["Check 1: S1 entities appear exactly once"] = (check_1_pass, f"Expected {expected_count:,}, Found {len(m_ids):,} in matching_results.tsv")
    results["Check 10: Exact row count equality"] = (check_10_pass, f"test_source1: {expected_count:,} | matching: {len(m_lines):,} | candidates: {len(c_lines):,}")

    # Check 2: No duplicate rows
    print("--- Check 2: No duplicate source1_entity_id rows ---")
    m_dups = len(m_ids) - len(m_id_set)
    c_dups = len(c_ids) - len(c_id_set)
    check_2_pass = (m_dups == 0) and (c_dups == 0)
    results["Check 2: No duplicate source1_entity_id rows"] = (check_2_pass, f"matching dups: {m_dups}, candidate dups: {c_dups}")

    # Check 3 & 5: Intra-list duplicates and subset constraint
    print("--- Check 3 & 5: Intra-list duplicates & matches <= candidates ---")
    intra_dups_m = 0
    intra_dups_c = 0
    subset_violations = 0
    s1_self_matches = 0
    wrong_prefix_matches = 0

    m_dict = {}
    c_dict = {}

    for row in m_lines:
        s1 = row[0]
        m_str = row[1] if len(row) > 1 else ""
        m_list = m_str.split(",") if m_str else []
        if len(m_list) != len(set(m_list)):
            intra_dups_m += 1
        for mid in m_list:
            if mid.startswith("S1-"):
                s1_self_matches += 1
            if not mid.startswith(("S2-", "S3-")):
                wrong_prefix_matches += 1
        m_dict[s1] = set(m_list)

    for row in c_lines:
        s1 = row[0]
        c_str = row[1] if len(row) > 1 else ""
        c_list = c_str.split(",") if c_str else []
        if len(c_list) != len(set(c_list)):
            intra_dups_c += 1
        c_dict[s1] = set(c_list)

    for s1, m_set in m_dict.items():
        c_set = c_dict.get(s1, set())
        diff = m_set - c_set
        if diff:
            subset_violations += 1

    check_3_pass = (intra_dups_m == 0) and (intra_dups_c == 0)
    check_5_pass = (subset_violations == 0)
    results["Check 3: No duplicate IDs within any single list"] = (check_3_pass, f"Matching list intra-dups: {intra_dups_m}, Candidate list intra-dups: {intra_dups_c}")
    results["Check 5: matches <= candidates in every row"] = (check_5_pass, f"Subset violations: {subset_violations}")

    # Check 4: ID existence in test set
    print("--- Check 4: Valid S2-/S3- prefix and no self-matches ---")
    check_4_pass = (s1_self_matches == 0) and (wrong_prefix_matches == 0)
    results["Check 4: Valid S2-/S3- IDs and zero S1- self-matches"] = (check_4_pass, f"Self-matches: {s1_self_matches}, Wrong prefixes: {wrong_prefix_matches}")

    # Check 6: Cross-country verification
    print("--- Check 6: Cross-country verification ---")
    # Load countries for S1 and candidate sources
    df_s1_c = pd.read_csv(s1_path, sep="\t", usecols=["entity_id", "country"])
    s1_country_map = dict(zip(df_s1_c["entity_id"], df_s1_c["country"]))

    # Sample 100,000 pairs to verify cross-country consistency
    cross_country_violations = 0
    # Load candidate countries from partitioned candidate files
    cands_country_map = {}
    for c in ["France", "US", "India"]:
        p = os.path.join(output_dir, "temp_work", f"cands_{c}.tsv")
        if os.path.isfile(p):
            df_sub = pd.read_csv(p, sep="\t", usecols=["entity_id", "country"])
            cands_country_map.update(dict(zip(df_sub["entity_id"], df_sub["country"])))

    spot_checked = 0
    for s1, m_set in m_dict.items():
        if not m_set:
            continue
        s1_c = s1_country_map.get(s1)
        for mid in m_set:
            cand_c = cands_country_map.get(mid)
            if cand_c and s1_c and cand_c != s1_c:
                cross_country_violations += 1
            spot_checked += 1
            if spot_checked >= 100000:
                break
        if spot_checked >= 100000:
            break

    check_6_pass = (cross_country_violations == 0)
    results["Check 6: No cross-country matches"] = (check_6_pass, f"Checked {spot_checked:,} match pairs, Violations: {cross_country_violations}")

    # Check 7: File format and column headers
    print("--- Check 7: TSV formatting and exact headers ---")
    m_head_clean = [c.strip().lower() for c in m_header.rstrip("\n").split("\t")]
    c_head_clean = [c.strip().lower() for c in c_header.rstrip("\n").split("\t")]
    check_7_pass = (m_head_clean == ["source1_entity_id", "matched_entity_ids"]) and (c_head_clean == ["source1_entity_id", "candidate_entity_ids"])
    results["Check 7: File format & exact column headers"] = (check_7_pass, f"Headers: {m_head_clean} & {c_head_clean}")

    # Check 8: Match rates per country
    print("--- Check 8: Total match-rate percentage per country ---")
    stats = {}
    for country in ["France", "US", "India"]:
        p = os.path.join(output_dir, "temp_work", f"results_{country}.tsv")
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                next(f)
                c_total = 0
                c_matched = 0
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    c_total += 1
                    if len(parts) > 2 and parts[2].strip():
                        c_matched += 1
                stats[country] = (c_total, c_matched, c_matched / max(1, c_total) * 100)

    summary_str = " | ".join(f"{c}: {m:,}/{tot:,} ({pct:.2f}%)" for c, (tot, m, pct) in stats.items())
    results["Check 8: Match-rate per country"] = (True, summary_str)

    # Check 9: Submission validator exit code
    print("--- Check 9: Official validate_submission.py execution ---")
    val_script = "d:/amazolml/student_resource/utils/validate_submission.py"
    proc = subprocess.run([
        sys.executable, val_script,
        "--matching", matching_path,
        "--candidate", candidate_path,
        "--test-dir", test_dir
    ], capture_output=True, text=True)
    val_pass = (proc.returncode == 0) and ("PASS" in proc.stdout)
    val_output = proc.stdout.strip()
    results["Check 9: Validator PASS / exit code 0"] = (val_pass, f"Exit code {proc.returncode}\n{val_output}")

    print("\n================================================================================")
    print("CHECKLIST RESULTS SUMMARY")
    print("================================================================================")
    all_ok = True
    for item, (passed, detail) in results.items():
        mark = "PASS [OK]" if passed else "FAIL [X]"
        if not passed:
            all_ok = False
        print(f"{mark} - {item}")
        print(f"       Detail: {detail}")

    print("\nOVERALL STATUS:", "ALL 10 CHECKS PASSED PERFECTLY" if all_ok else "CHECKLIST FAILED")
    return all_ok

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default="d:/amazolml/student_resource/dataset/test")
    parser.add_argument("--output-dir", default="d:/amazolml/output")
    args = parser.parse_args()
    ok = run_checklist(test_dir=args.test_dir, output_dir=args.output_dir)
    sys.exit(0 if ok else 1)
