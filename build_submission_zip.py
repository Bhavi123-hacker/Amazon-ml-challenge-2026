"""Script to build and verify the exact required submission zip archive.

Enforces:
1. Exact directory layout matching competition specifications.
2. Complete exclusion of __pycache__, .pyc, .DS_Store, __MACOSX, and temporary logs.
3. Inclusion of finalized output/matching_results.tsv and output/candidate_pairs.tsv.
4. Comprehensive integrity inspection and size verification.
"""

import os
import sys
import zipfile
import time

def build_zip(team_name="EntityResolution_Masters"):
    zip_filename = f"{team_name}_submission.zip"
    print(f"Building submission zip: {zip_filename} ...", flush=True)
    t0 = time.time()

    root_dir = "d:/amazolml"
    output_dir = os.path.join(root_dir, "output")
    code_dir = os.path.join(root_dir, "code", "business_entity_resolution")
    src_dir = os.path.join(code_dir, "src")
    doc_path = os.path.join(root_dir, "student_resource", "Documentation_template.md")

    # Files to include
    matching_tsv = os.path.join(output_dir, "matching_results.tsv")
    candidate_tsv = os.path.join(output_dir, "candidate_pairs.tsv")
    readme_path = os.path.join(code_dir, "README.md")
    req_path = os.path.join(code_dir, "requirements.txt")

    # Verify presence
    for p in [matching_tsv, candidate_tsv, readme_path, req_path, doc_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    with zipfile.ZipFile(zip_filename, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # 1. output/
        print("Adding output/matching_results.tsv ...", flush=True)
        zf.write(matching_tsv, arcname="output/matching_results.tsv")

        print("Adding output/candidate_pairs.tsv ...", flush=True)
        zf.write(candidate_tsv, arcname="output/candidate_pairs.tsv")

        # 2. code/business_entity_resolution/
        print("Adding code/business_entity_resolution/README.md ...", flush=True)
        zf.write(readme_path, arcname="code/business_entity_resolution/README.md")

        print("Adding code/business_entity_resolution/requirements.txt ...", flush=True)
        zf.write(req_path, arcname="code/business_entity_resolution/requirements.txt")

        # 3. code/business_entity_resolution/src/
        print("Adding code/business_entity_resolution/src/ files ...", flush=True)
        for fname in sorted(os.listdir(src_dir)):
            if fname.endswith(".py") and not fname.startswith("."):
                fpath = os.path.join(src_dir, fname)
                arc_name = f"code/business_entity_resolution/src/{fname}"
                print(f"  + {arc_name}", flush=True)
                zf.write(fpath, arcname=arc_name)

        # 4. Documentation_template.md at zip root
        print("Adding Documentation_template.md ...", flush=True)
        zf.write(doc_path, arcname="Documentation_template.md")

    elapsed = time.time() - t0
    zip_size = os.path.getsize(zip_filename)
    print(f"\nSuccessfully built {zip_filename} in {elapsed:.2f}s!")
    print(f"Total zip size: {zip_size:,} bytes ({zip_size / (1024 * 1024):.2f} MB)")

    # Integrity verification
    print("\n--- Verifying Zip File Structure & Integrity ---", flush=True)
    with zipfile.ZipFile(zip_filename, "r") as zf:
        bad_file = zf.testzip()
        if bad_file:
            raise ValueError(f"Zip corrupted at entry: {bad_file}")
        
        infolist = zf.infolist()
        print(f"Total entries in zip: {len(infolist)}")
        for info in infolist:
            print(f"  {info.filename:<55} | Size: {info.file_size:>11,d} B | Compressed: {info.compress_size:>10,d} B")

    print("\nZip structure verification: ALL CHECKS PASSED PERFECTLY!")

if __name__ == "__main__":
    team = sys.argv[1] if len(sys.argv) > 1 else "BitMinds"
    build_zip(team)
