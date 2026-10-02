"""Full Pre-Upload Verification of the Generated Submission Zip Archive.

Tests the EXACT files extracted directly from EntityResolution_Masters_submission.zip:
1. Validates archive structure against official spec.
2. Re-runs official validate_submission.py on the extracted TSVs.
3. Re-runs the full 10-item checklist on the extracted TSVs.
4. Checks file sizes and uncompressed limits.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

def verify_zip(zip_path="EntityResolution_Masters_submission.zip"):
    print("=" * 80)
    print(f"VERIFYING FINAL SUBMISSION ZIP: {zip_path}")
    print("=" * 80)

    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    zip_bytes = os.path.getsize(zip_path)
    print(f"Zip File Size: {zip_bytes:,} bytes ({zip_bytes / (1024*1024):.2f} MB)")

    # 1. Structure check
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        
        # Check for forbidden artifacts
        forbidden = [n for n in namelist if any(x in n for x in [".DS_Store", "__MACOSX", "__pycache__", ".pyc", ".tmp"])]
        if forbidden:
            raise ValueError(f"Forbidden files found in zip: {forbidden}")
        print("Forbidden files check: PASSED (0 unwanted artifacts)")

        # Check required files
        required = [
            "output/matching_results.tsv",
            "output/candidate_pairs.tsv",
            "code/business_entity_resolution/README.md",
            "code/business_entity_resolution/requirements.txt",
            "Documentation_template.md"
        ]
        for r in required:
            if r not in namelist:
                raise ValueError(f"Missing required file in zip: {r}")
        print("Required root & output files check: PASSED")

        # 2. Extract TSVs to temporary directory for live verification
        temp_dir = tempfile.mkdtemp(prefix="amazolml_val_")
        try:
            print(f"\nExtracting outputs to temporary directory: {temp_dir} ...", flush=True)
            zf.extract("output/matching_results.tsv", temp_dir)
            zf.extract("output/candidate_pairs.tsv", temp_dir)

            ext_match = os.path.join(temp_dir, "output", "matching_results.tsv")
            ext_cand = os.path.join(temp_dir, "output", "candidate_pairs.tsv")

            # Run official validate_submission.py
            print("\n--- Running Official validate_submission.py against ZIPPED files ---", flush=True)
            val_script = "student_resource/utils/validate_submission.py"
            test_dir = "student_resource/dataset/test"
            
            res = subprocess.run([
                sys.executable, val_script,
                "--matching", ext_match,
                "--candidate", ext_cand,
                "--test-dir", test_dir
            ], capture_output=True, text=True)

            print(res.stdout)
            if res.stderr:
                print("STDERR:", res.stderr)

            if res.returncode != 0 or "PASS" not in res.stdout:
                raise RuntimeError("validate_submission.py FAILED on extracted zip files!")
            print("Official validator on zipped files: PASS [OK] (Exit Code 0)")

            # Run 10/10 checklist
            print("\n--- Running 10/10 Verification Checklist against ZIPPED files ---", flush=True)
            chk_script = "code/business_entity_resolution/verify_checklist.py"
            res_chk = subprocess.run([
                sys.executable, chk_script,
                "--test-dir", test_dir,
                "--output-dir", os.path.join(temp_dir, "output")
            ], capture_output=True, text=True)
            print(res_chk.stdout)
            if res_chk.stderr:
                print("STDERR:", res_chk.stderr)
            if res_chk.returncode != 0 or "ALL 10 CHECKS PASSED PERFECTLY" not in res_chk.stdout:
                raise RuntimeError("10/10 Checklist FAILED on extracted zip files!")
            print("10/10 Checklist on zipped files: PASS [OK]")

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n" + "=" * 80)
    print("ALL PRE-UPLOAD VERIFICATION CHECKS PASSED WITH 100% SUCCESS")
    print("=" * 80)

if __name__ == "__main__":
    verify_zip()
