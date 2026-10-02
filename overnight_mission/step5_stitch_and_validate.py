"""Step 5: Full Submission Assembly, Synchronization, and Official Validation.

Guarantees:
1. Exact line order matching test_source1.tsv (1,732,544 rows + 1 header).
2. 100% Containment: 'matched_entity_ids <= candidate_entity_ids' on all rows.
3. Strict Caps: candidates <= 12, matches <= 10.
4. Identical Byte Synchronization: BitMinds/output/ and output/ share identical SHA-256 hashes.
5. Official Submission Validation: Exits with code 0 on student_resource/utils/validate_submission.py.
6. Clean zip archive created: BitMinds_submission.zip.
"""

import hashlib
import os
import shutil
import subprocess
import sys
import time
import zipfile

def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def main():
    t0 = time.time()
    print("=" * 85, flush=True)
    print("STEP 5: FULL-SCALE SUBMISSION ASSEMBLY, SYNCHRONIZATION & VALIDATION", flush=True)
    print("=" * 85, flush=True)

    test_dir = "student_resource/dataset/test"
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    temp_dir = "output/temp_work"

    # Input file paths
    fr_file = os.path.join(temp_dir, "champion_results_France_guard_d.tsv")
    us_file = os.path.join(temp_dir, "lgbm_clean_results_US.tsv")
    in_file = os.path.join(temp_dir, "lgbm_clean_results_India_tau_90.tsv")

    print(f"Loading France predictions from {fr_file}...", flush=True)
    fr_data = {}
    with open(fr_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            fr_data[p[0]] = (p[1] if len(p) > 1 else "", p[2] if len(p) > 2 else "")
    print(f"  France loaded: {len(fr_data):,} entities.", flush=True)

    print(f"Loading US predictions from {us_file}...", flush=True)
    us_data = {}
    with open(us_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            us_data[p[0]] = (p[1] if len(p) > 1 else "", p[2] if len(p) > 2 else "")
    print(f"  US loaded: {len(us_data):,} entities.", flush=True)

    print(f"Loading India predictions from {in_file}...", flush=True)
    in_data = {}
    with open(in_file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            in_data[p[0]] = (p[1] if len(p) > 1 else "", p[2] if len(p) > 2 else "")
    print(f"  India loaded: {len(in_data):,} entities.", flush=True)

    total_pool = len(fr_data) + len(us_data) + len(in_data)
    print(f"Total entities loaded: {total_pool:,}", flush=True)

    # Output paths
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)
    match_file = os.path.join(out_dir, "matching_results.tsv")
    cand_file = os.path.join(out_dir, "candidate_pairs.tsv")

    print(f"\nWriting master submission files strictly in {s1_path} order...", flush=True)
    f_match = open(match_file, "w", encoding="utf-8")
    f_cand = open(cand_file, "w", encoding="utf-8")

    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

    total = 0
    matched = 0
    singletons = 0
    total_cands = 0
    total_matches = 0
    containment_violations = 0
    cand_cap_violations = 0
    match_cap_violations = 0
    match_dist = {}

    with open(s1_path, "r", encoding="utf-8") as f_s1:
        next(f_s1)
        for line in f_s1:
            total += 1
            sid = line.split("\t", 1)[0].strip()
            if not sid:
                continue

            if sid in us_data:
                c_str, m_str = us_data[sid]
            elif sid in in_data:
                c_str, m_str = in_data[sid]
            elif sid in fr_data:
                c_str, m_str = fr_data[sid]
            else:
                c_str, m_str = "", ""

            c_list = [c.strip() for c in c_str.split(",") if c.strip()] if c_str else []
            m_list = [m.strip() for m in m_str.split(",") if m.strip()] if m_str else []

            # Hard invariants check & enforce
            if len(m_list) > 10:
                match_cap_violations += 1
                m_list = m_list[:10]

            m_set = set(m_list)
            # Guarantee containment: all matches MUST be in candidates
            final_cands = list(m_list)
            for c in c_list:
                if c not in m_set:
                    final_cands.append(c)
                if len(final_cands) >= 12:
                    break

            if len(final_cands) > 12:
                cand_cap_violations += 1
                final_cands = final_cands[:12]

            # Invariant check
            if not m_set.issubset(set(final_cands)):
                containment_violations += 1

            c_out = ",".join(final_cands)
            m_out = ",".join(m_list)

            f_cand.write(f"{sid}\t{c_out}\n")
            f_match.write(f"{sid}\t{m_out}\n")

            n_m = len(m_list)
            match_dist[n_m] = match_dist.get(n_m, 0) + 1
            total_matches += n_m
            total_cands += len(final_cands)
            if n_m > 0:
                matched += 1
            else:
                singletons += 1

    f_match.close()
    f_cand.close()

    print("\n" + "=" * 80)
    print("MASTER SUBMISSION FILES WRITTEN SUCCESSFULLY")
    print("=" * 80)
    print(f"Total Rows Written     : {total:,} (Expected: 1,732,544)")
    print(f"Entities with Matches  : {matched:,} ({matched/total*100:.2f}%)")
    print(f"Natural Singletons     : {singletons:,} ({singletons/total*100:.2f}%)")
    print(f"Total Matches          : {total_matches:,} (avg {total_matches/total:.2f}/entity)")
    print(f"Total Candidates       : {total_cands:,} (avg {total_cands/total:.2f}/entity)")
    print(f"Containment Violations : {containment_violations} (MUST BE 0)")
    print(f"Candidate Cap Violations: {cand_cap_violations} (MUST BE 0)")
    print(f"Match Cap Violations   : {match_cap_violations} (MUST BE 0)")

    print("\n--- Match Distribution ---")
    for k in sorted(match_dist.keys()):
        print(f"  {k:>2} matches: {match_dist[k]:>8,d} entities ({match_dist[k]/total*100:>5.2f}%)")

    # 2. Synchronize to BitMinds/output/
    print("\n" + "=" * 80)
    print("SYNCHRONIZING WITH BitMinds/output/ TO PREVENT MISMATCHES")
    print("=" * 80)
    bm_out_dir = "BitMinds/output"
    os.makedirs(bm_out_dir, exist_ok=True)
    bm_match_file = os.path.join(bm_out_dir, "matching_results.tsv")
    bm_cand_file = os.path.join(bm_out_dir, "candidate_pairs.tsv")

    shutil.copy2(match_file, bm_match_file)
    shutil.copy2(cand_file, bm_cand_file)

    h_m1 = compute_sha256(match_file)
    h_m2 = compute_sha256(bm_match_file)
    h_c1 = compute_sha256(cand_file)
    h_c2 = compute_sha256(bm_cand_file)

    print(f"output/matching_results.tsv           SHA-256: {h_m1}")
    print(f"BitMinds/output/matching_results.tsv  SHA-256: {h_m2}")
    assert h_m1 == h_m2, "MATCHING HASH MISMATCH!"
    print(f"  -> MATCHING RESULTS HASHES ARE 100% IDENTICAL!")

    print(f"output/candidate_pairs.tsv            SHA-256: {h_c1}")
    print(f"BitMinds/output/candidate_pairs.tsv   SHA-256: {h_c2}")
    assert h_c1 == h_c2, "CANDIDATE HASH MISMATCH!"
    print(f"  -> CANDIDATE PAIRS HASHES ARE 100% IDENTICAL!")

    # 3. Official Validation
    print("\n" + "=" * 80)
    print("RUNNING OFFICIAL VALIDATION SCRIPT")
    print("=" * 80)
    val_cmd = [
        sys.executable,
        "student_resource/utils/validate_submission.py",
        "--matching", match_file,
        "--candidate", cand_file,
        "--test-dir", test_dir
    ]
    res = subprocess.run(val_cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr)
    print(f"Validation exit code: {res.returncode}")
    assert res.returncode == 0, f"OFFICIAL VALIDATION FAILED with code {res.returncode}"

    # 4. Create BitMinds_submission.zip
    print("\n" + "=" * 80)
    print("PACKAGING BitMinds_submission.zip")
    print("=" * 80)
    zip_path = "BitMinds_submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(match_file, "output/matching_results.tsv")
        zipf.write(cand_file, "output/candidate_pairs.tsv")
        # Include clean code
        for root, _, files in os.walk("code"):
            for f in files:
                full = os.path.join(root, f)
                rel = os.path.relpath(full, ".")
                zipf.write(full, rel)
    print(f"Created {zip_path} ({os.path.getsize(zip_path):,} bytes).")
    print(f"\nSTEP 5 COMPLETE in {time.time() - t0:.1f}s.")

if __name__ == "__main__":
    main()
