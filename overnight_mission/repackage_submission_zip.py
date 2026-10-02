"""Repackage BitMinds_submission.zip with exact required structure and verify SHA-256."""

import hashlib
import os
import zipfile

def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 80)
    print("PACKAGING & VERIFYING BitMinds_submission.zip")
    print("=" * 80)

    zip_path = "BitMinds_submission.zip"
    if os.path.exists(zip_path):
        os.remove(zip_path)

    match_file = "output/matching_results.tsv"
    cand_file = "output/candidate_pairs.tsv"
    doc_file = "Documentation_template.md"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # 1. Output files
        print("Adding output files...")
        zipf.write(match_file, "output/matching_results.tsv")
        zipf.write(cand_file, "output/candidate_pairs.tsv")

        # 2. Documentation
        print("Adding documentation...")
        zipf.write(doc_file, "Documentation_template.md")

        # 3. Code files
        print("Adding code repository...")
        for root, dirs, files in os.walk("code/business_entity_resolution"):
            for f in files:
                if f.endswith((".py", ".md", ".txt", ".sh", ".json")):
                    full_p = os.path.join(root, f)
                    rel_p = os.path.relpath(full_p, ".")
                    zipf.write(full_p, rel_p)

    zip_size = os.path.getsize(zip_path)
    zip_sha = compute_sha256(zip_path)

    print("\n" + "=" * 80)
    print("SUBMISSION ZIP PACKAGE READY")
    print("=" * 80)
    print(f"Path   : {os.path.abspath(zip_path)}")
    print(f"Size   : {zip_size:,} bytes ({zip_size / (1024*1024):.2f} MB)")
    print(f"SHA-256: {zip_sha}")

    # Inspect zip contents
    print("\nZip Contents:")
    with zipfile.ZipFile(zip_path, "r") as zipf:
        for info in zipf.infolist()[:15]:
            print(f"  - {info.filename:<50} ({info.file_size:>10,d} bytes)")
        if len(zipf.infolist()) > 15:
            print(f"  ... and {len(zipf.infolist()) - 15} more code files.")

if __name__ == "__main__":
    main()
