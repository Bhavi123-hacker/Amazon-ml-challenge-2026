"""Apply 2-Set Disjoint Bipartite Resolution to final submission and validate."""

import os
import sys
import hashlib
import zipfile
import subprocess

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 80)
    print("APPLYING 2-SET DISJOINT BIPARTITE RESOLUTION TO MASTER SUBMISSION")
    print("=" * 80)

    # 1. Read existing matching results and resolve collisions
    cand_best_claim = {} # cid -> (sid, rank_idx)
    s1_rows = []

    print("Pass 1: Identifying unique best S1 entity for every candidate...", flush=True)
    with open("output/matching_results.tsv", "r", encoding="utf-8") as f:
        header = next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            sid = parts[0]
            matches = [c.strip() for c in parts[1].split(",") if c.strip()] if len(parts) > 1 and parts[1] else []
            s1_rows.append((sid, matches))
            for r_idx, cid in enumerate(matches):
                if cid not in cand_best_claim:
                    cand_best_claim[cid] = (sid, r_idx)
                else:
                    prev_sid, prev_rank = cand_best_claim[cid]
                    if r_idx < prev_rank:
                        cand_best_claim[cid] = (sid, r_idx)

    # 2. Write resolved output
    print("Pass 2: Writing 2-Set Disjoint matching_results.tsv...", flush=True)
    out_path = "output/matching_results.tsv"
    bitminds_path = "BitMinds/output/matching_results.tsv"

    total_entities = len(s1_rows)
    total_matches = 0
    singletons = 0
    match_hist = {}

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid, matches in s1_rows:
            kept = []
            for cid in matches:
                best_sid, _ = cand_best_claim[cid]
                if best_sid == sid:
                    kept.append(cid)
            m_str = ",".join(kept)
            f.write(f"{sid}\t{m_str}\n")
            k = len(kept)
            total_matches += k
            if k == 0:
                singletons += 1
            match_hist[k] = match_hist.get(k, 0) + 1

    print(f"\n2-Set Disjoint Results Written:")
    print(f"  Total Entities      : {total_entities:,}")
    print(f"  Total Matches       : {total_matches:,} (avg {total_matches/total_entities:.2f}/entity)")
    print(f"  Natural Singletons  : {singletons:,} ({singletons/total_entities*100:.2f}%)")

    # 3. Synchronize to BitMinds/output/
    print("\nSynchronizing to BitMinds/output/...", flush=True)
    with open(out_path, "rb") as src, open(bitminds_path, "wb") as dst:
        dst.write(src.read())

    hash_out = sha256_file(out_path)
    hash_bm = sha256_file(bitminds_path)
    print(f"  output/matching_results.tsv          SHA-256: {hash_out}")
    print(f"  BitMinds/output/matching_results.tsv SHA-256: {hash_bm}")
    assert hash_out == hash_bm, "FATAL: Hash mismatch between output/ and BitMinds/output/!"
    print("  -> 100% byte-for-byte identical verified!")

    # 4. Run Validator
    print("\nRunning official submission validator...", flush=True)
    val_cmd = [
        sys.executable,
        "student_resource/utils/validate_submission.py",
        "--test-dir", "student_resource/dataset/test",
        "--matching", out_path,
        "--candidate", "output/candidate_pairs.tsv"
    ]
    res = subprocess.run(val_cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)
    assert res.returncode == 0, f"Validator failed with code {res.returncode}!"
    print("Validation return code: 0 (PASS)")

    # 5. Repackage BitMinds_submission.zip
    print("\nRepackaging BitMinds_submission.zip...", flush=True)
    zip_path = "BitMinds_submission.zip"
    if os.path.exists(zip_path):
        os.remove(zip_path)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(bitminds_path, "BitMinds/output/matching_results.tsv")
        z.write("BitMinds/output/candidate_pairs.tsv", "BitMinds/output/candidate_pairs.tsv")

    zip_size = os.path.getsize(zip_path)
    zip_hash = sha256_file(zip_path)
    print(f"BitMinds_submission.zip created successfully ({zip_size:,} bytes, {zip_size/(1024*1024):.2f} MB)")
    print(f"ZIP SHA-256: {zip_hash}")

    print("\n" + "=" * 80)
    print("ALL SUBMISSION ARTIFACTS FINALIZED AND VERIFIED!")
    print("=" * 80)

if __name__ == "__main__":
    main()
