import os
import sys
import shutil
import hashlib
import subprocess
import zipfile

def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 80)
    print("FINALIZING COMBINED-GUARD SUBMISSION & OFFICIAL VALIDATION")
    print("=" * 80)

    src_guarded = "output/matching_results_final_guarded.tsv"
    dest_matching = "output/matching_results.tsv"
    dest_bitminds = "BitMinds/output/matching_results.tsv"
    zip_path = "BitMinds_submission.zip"

    assert os.path.exists(src_guarded), f"Missing {src_guarded}"

    print("Copying final guarded file to output/matching_results.tsv...", flush=True)
    shutil.copyfile(src_guarded, dest_matching)

    print("Copying final guarded file to BitMinds/output/matching_results.tsv...", flush=True)
    os.makedirs(os.path.dirname(dest_bitminds), exist_ok=True)
    shutil.copyfile(dest_matching, dest_bitminds)

    hash_out = compute_sha256(dest_matching)
    hash_bm = compute_sha256(dest_bitminds)
    print(f"  output/matching_results.tsv          SHA-256: {hash_out}")
    print(f"  BitMinds/output/matching_results.tsv SHA-256: {hash_bm}")
    assert hash_out == hash_bm, "FATAL: Hash mismatch between output/ and BitMinds/output/!"
    print("  -> 100% byte-for-byte identical verified!")

    # Run official validator
    print("\nRunning official submission validator...", flush=True)
    val_cmd = [
        sys.executable,
        "student_resource/utils/validate_submission.py",
        "--test-dir", "student_resource/dataset/test",
        "--matching", dest_matching,
        "--candidate", "output/candidate_pairs.tsv"
    ]
    res = subprocess.run(val_cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)
    assert res.returncode == 0, f"Validator failed with code {res.returncode}!"
    print("Validation return code: 0 (PASS)")

    # Repackage BitMinds_submission.zip
    print("\nRepackaging BitMinds_submission.zip...", flush=True)
    if os.path.exists(zip_path):
        os.remove(zip_path)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk("BitMinds"):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, ".")
                zf.write(full_path, rel_path)

    zip_hash = compute_sha256(zip_path)
    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"BitMinds_submission.zip created successfully!")
    print(f"  Size   : {zip_size_mb:.2f} MB")
    print(f"  SHA-256: {zip_hash}")

    # Verify matching_results.tsv inside zip matches exactly
    with zipfile.ZipFile(zip_path, "r") as zf:
        zip_inner_hash = hashlib.sha256(zf.read("BitMinds/output/matching_results.tsv")).hexdigest()
        print(f"  Inner matching_results.tsv SHA-256: {zip_inner_hash}")
        assert zip_inner_hash == hash_out, "FATAL: Hash mismatch inside zip!"
        print("  -> Inner zip matching_results.tsv is 100% byte-for-byte identical!")

    print("\n" + "=" * 80)
    print("SUBMISSION SUCCESSFULLY FINALIZED AND VALIDATED")
    print("=" * 80)

if __name__ == "__main__":
    main()
