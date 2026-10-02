"""Step 2: ID and Join Integrity Check on the actual submitted matching_results.tsv."""

import sys
import pandas as pd

def main():
    print("=" * 80)
    print("STEP 2: FULL ID AND JOIN INTEGRITY AUDIT")
    print("=" * 80)

    # 1. Load all valid S1 IDs from test_source1.tsv
    print("Loading valid S1 IDs from student_resource/dataset/test/test_source1.tsv...", flush=True)
    valid_s1_ids = set()
    s1_whitespace_issues = 0
    with open("student_resource/dataset/test/test_source1.tsv", "r", encoding="utf-8") as f:
        header = next(f)
        for line_no, line in enumerate(f, 2):
            parts = line.split("\t")
            sid = parts[0]
            if sid != sid.strip():
                s1_whitespace_issues += 1
            valid_s1_ids.add(sid.strip())

    print(f"Total valid S1 IDs in test_source1.tsv: {len(valid_s1_ids):,}")
    print(f"S1 IDs with raw whitespace in source: {s1_whitespace_issues}")

    # 2. Load all valid S2 and S3 IDs from test_source2.tsv and test_source3.tsv
    print("\nLoading valid S2 & S3 IDs from test_source2.tsv and test_source3.tsv...", flush=True)
    valid_cand_ids = set()
    cand_whitespace_issues = 0
    for p in ["student_resource/dataset/test/test_source2.tsv", "student_resource/dataset/test/test_source3.tsv"]:
        print(f"  Reading {p}...", flush=True)
        with open(p, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                cid = line.split("\t")[0]
                if cid != cid.strip():
                    cand_whitespace_issues += 1
                valid_cand_ids.add(cid.strip())

    print(f"Total valid S2/S3 IDs in test sources: {len(valid_cand_ids):,}")
    print(f"Candidate IDs with raw whitespace in source: {cand_whitespace_issues}")

    # 3. Check BitMinds/output/matching_results.tsv line by line
    sub_path = "BitMinds/output/matching_results.tsv"
    print(f"\nChecking submitted file: {sub_path}...", flush=True)

    total_rows = 0
    s1_matched_count = 0
    s1_missing_in_source = 0
    s1_whitespace_err = 0
    duplicate_s1_rows = 0
    seen_s1 = set()

    total_predicted_matches = 0
    cand_missing_in_source = 0
    cand_whitespace_err = 0
    duplicate_cands_in_row = 0
    invalid_chars_in_id = 0

    with open(sub_path, "r", encoding="utf-8") as f:
        header = next(f)
        for line_idx, line in enumerate(f, 2):
            total_rows += 1
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) == 0:
                continue
            sid = parts[0]
            
            # Check S1 ID
            if sid != sid.strip():
                s1_whitespace_err += 1
            sid_clean = sid.strip()
            
            if sid_clean in seen_s1:
                duplicate_s1_rows += 1
            seen_s1.add(sid_clean)

            if sid_clean not in valid_s1_ids:
                s1_missing_in_source += 1

            # Check Matched IDs
            if len(parts) > 1 and parts[1].strip():
                s1_matched_count += 1
                cands = parts[1].split(",")
                row_seen_cands = set()
                for c in cands:
                    total_predicted_matches += 1
                    if c != c.strip():
                        cand_whitespace_err += 1
                    c_clean = c.strip()
                    
                    if c_clean in row_seen_cands:
                        duplicate_cands_in_row += 1
                    row_seen_cands.add(c_clean)

                    if c_clean not in valid_cand_ids:
                        cand_missing_in_source += 1

                    # Check for non-ascii or hidden chars
                    if not c_clean.replace("-", "").isalnum():
                        invalid_chars_in_id += 1

    print("\n" + "=" * 80)
    print("ID INTEGRITY AUDIT RESULTS")
    print("=" * 80)
    print(f"Total rows in submitted file           : {total_rows:,}")
    print(f"Required S1 rows in test_source1.tsv   : {len(valid_s1_ids):,}")
    print(f"Rows matching test_source1 row count   : {total_rows == len(valid_s1_ids)}")
    print(f"S1 IDs missing in test_source1.tsv     : {s1_missing_in_source}")
    print(f"S1 IDs with whitespace / padding       : {s1_whitespace_err}")
    print(f"Duplicate S1 rows                      : {duplicate_s1_rows}")
    print(f"Entities with predicted matches        : {s1_matched_count:,} ({s1_matched_count/total_rows*100:.2f}%)")
    print(f"Singletons (0 matches)                 : {total_rows - s1_matched_count:,} ({(total_rows - s1_matched_count)/total_rows*100:.2f}%)")
    print(f"Total predicted candidate matches      : {total_predicted_matches:,}")
    print(f"Matched IDs missing in S2/S3           : {cand_missing_in_source}")
    print(f"Matched IDs with whitespace / padding  : {cand_whitespace_err}")
    print(f"Duplicate candidate IDs within a row   : {duplicate_cands_in_row}")
    print(f"Candidate IDs with abnormal characters : {invalid_chars_in_id}")
    print("=" * 80)

if __name__ == "__main__":
    main()
