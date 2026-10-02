"""Verify all final submission invariants:
1. Exact row count: 1,732,544 rows + 1 header
2. 100% containment: matches <= candidates
3. Candidate count <= 12
4. Match count <= 10
5. No duplicate S1 IDs
6. Exact line order matching test_source1.tsv
"""

import sys

def main():
    s1_path = "student_resource/dataset/test/test_source1.tsv"
    match_path = "output/matching_results.tsv"
    cand_path = "output/candidate_pairs.tsv"

    print("Verifying final submission files invariants...", flush=True)

    with open(s1_path, "r", encoding="utf-8") as fs1, \
         open(match_path, "r", encoding="utf-8") as fm, \
         open(cand_path, "r", encoding="utf-8") as fc:

        h_s1 = next(fs1).strip()
        h_m = next(fm).strip()
        h_c = next(fc).strip()

        assert h_m == "source1_entity_id\tmatched_entity_ids", f"Bad match header: {h_m}"
        assert h_c == "source1_entity_id\tcandidate_entity_ids", f"Bad cand header: {h_c}"

        count = 0
        seen_s1 = set()
        containment_errs = 0
        max_cands = 0
        max_matches = 0

        for line_idx, (ls1, lm, lc) in enumerate(zip(fs1, fm, fc), 1):
            count += 1
            s1_id = ls1.split("\t", 1)[0].strip()
            m_parts = lm.rstrip("\n").split("\t")
            c_parts = lc.rstrip("\n").split("\t")

            m_sid = m_parts[0]
            c_sid = c_parts[0]

            assert s1_id == m_sid == c_sid, f"Line {line_idx} ID mismatch: s1={s1_id}, m={m_sid}, c={c_sid}"
            assert s1_id not in seen_s1, f"Duplicate S1 ID at line {line_idx}: {s1_id}"
            seen_s1.add(s1_id)

            m_list = [x.strip() for x in m_parts[1].split(",") if x.strip()] if len(m_parts) > 1 and m_parts[1] else []
            c_list = [x.strip() for x in c_parts[1].split(",") if x.strip()] if len(c_parts) > 1 and c_parts[1] else []

            max_matches = max(max_matches, len(m_list))
            max_cands = max(max_cands, len(c_list))

            assert len(m_list) <= 10, f"Line {line_idx} exceeds match cap 10: {len(m_list)}"
            assert len(c_list) <= 12, f"Line {line_idx} exceeds cand cap 12: {len(c_list)}"

            c_set = set(c_list)
            for m in m_list:
                if m not in c_set:
                    containment_errs += 1

        print(f"Total Rows Verified: {count:,}")
        print(f"Max Matches per Entity: {max_matches}")
        print(f"Max Candidates per Entity: {max_cands}")
        print(f"Containment Errors: {containment_errs}")
        assert containment_errs == 0, f"Containment errors found: {containment_errs}"
        assert count == 1732544, f"Expected 1,732,544 rows, found {count}"
        print("ALL SUBMISSION INVARIANTS PERFECTLY SATISFIED!")

if __name__ == "__main__":
    main()
