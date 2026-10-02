import os
import sys
import time
import re
import pandas as pd

clean_re1 = re.compile(r'[#№°]')
clean_re2 = re.compile(r'\b(no|n°|nº|num|number)\b\.?', re.IGNORECASE)
clean_re3 = re.compile(r'(\d+)[-/](\d+)')
ocr_re = re.compile(r'(?<=\d)[oO]\b|\b[oO](?=\d)')
tok_re = re.compile(r'\b\d+[a-zA-Z]?\b')
non_digit_re = re.compile(r'\D')

def extract_street_numbers(addr: str) -> tuple:
    if not addr or not isinstance(addr, str):
        return ()
    c = ocr_re.sub('0', addr)
    c = clean_re1.sub(' ', c)
    c = clean_re2.sub(' ', c)
    c = clean_re3.sub(r'\1 \2', c)
    tokens = tok_re.findall(c)
    nums = []
    for tok in tokens:
        bd = non_digit_re.sub('', tok).lstrip('0')
        if bd and len(bd) <= 6:
            nums.append(int(bd))
    return tuple(set(nums))

def is_street_mismatch(nums1: tuple, nums2: tuple) -> bool:
    if not nums1 or not nums2:
        return False
    for n in nums1:
        if n in nums2:
            return False
    return True

def main():
    t0 = time.time()
    print("="*80)
    print("APPLYING STREET NUMBER CONSISTENCY GUARD TO SUBMISSION")
    print("="*80)

    # 1. Load addresses and extract street numbers
    print("Pass 1: Loading test_source1 addresses...", flush=True)
    df1 = pd.read_csv('student_resource/dataset/test/test_source1.tsv', sep='\t', usecols=['entity_id', 'business_address'])
    s1_nums = {row[0]: extract_street_numbers(row[1]) for row in df1.itertuples(index=False)}
    del df1
    print(f"  Source 1 loaded: {len(s1_nums):,} entities in {time.time()-t0:.1f}s", flush=True)

    t1 = time.time()
    print("Pass 2: Loading test_source2 addresses...", flush=True)
    df2 = pd.read_csv('student_resource/dataset/test/test_source2.tsv', sep='\t', usecols=['entity_id', 'business_address'])
    cand_nums = {row[0]: extract_street_numbers(row[1]) for row in df2.itertuples(index=False)}
    del df2
    print(f"  Source 2 loaded: {len(cand_nums):,} entities in {time.time()-t1:.1f}s", flush=True)

    t2 = time.time()
    print("Pass 3: Loading test_source3 addresses...", flush=True)
    df3 = pd.read_csv('student_resource/dataset/test/test_source3.tsv', sep='\t', usecols=['entity_id', 'business_address'])
    cand_nums.update({row[0]: extract_street_numbers(row[1]) for row in df3.itertuples(index=False)})
    del df3
    print(f"  Source 2+3 total candidates: {len(cand_nums):,} in {time.time()-t2:.1f}s", flush=True)

    # 2. Filter existing matching_results.tsv
    print("\nPass 4: Filtering matching_results.tsv with street number consistency guard...", flush=True)
    in_path = "output/matching_results.tsv"
    out_path = "output/matching_results_street_guarded.tsv"

    total_rows = 0
    total_original_matches = 0
    total_kept_matches = 0
    total_rejected_matches = 0
    original_singletons = 0
    new_singletons = 0
    match_dist = {}

    with open(in_path, "r", encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
        header = next(f_in)
        f_out.write(header)
        for line in f_in:
            total_rows += 1
            parts = line.rstrip("\n").split("\t")
            sid = parts[0]
            matches = [c.strip() for c in parts[1].split(",") if c.strip()] if len(parts) > 1 and parts[1] else []
            total_original_matches += len(matches)
            if not matches:
                original_singletons += 1

            s1_n = s1_nums.get(sid, ())
            kept = []
            for cid in matches:
                c_n = cand_nums.get(cid, ())
                if is_street_mismatch(s1_n, c_n):
                    total_rejected_matches += 1
                else:
                    kept.append(cid)

            k = len(kept)
            total_kept_matches += k
            if k == 0:
                new_singletons += 1
            match_dist[k] = match_dist.get(k, 0) + 1

            f_out.write(f"{sid}\t{','.join(kept)}\n")

    print("\n" + "="*80)
    print("STREET NUMBER CONSISTENCY GUARD FILTERING RESULTS")
    print("="*80)
    print(f"Total S1 Entities           : {total_rows:,}")
    print(f"Original Total Matches      : {total_original_matches:,} (avg {total_original_matches/total_rows:.2f}/entity)")
    print(f"Original Singletons         : {original_singletons:,} ({original_singletons/total_rows*100:.2f}%)")
    print(f"Rejected Matches (Mismatches): {total_rejected_matches:,} ({total_rejected_matches/total_original_matches*100:.2f}%)")
    print(f"Kept Genuine Matches        : {total_kept_matches:,} (avg {total_kept_matches/total_rows:.2f}/entity)")
    print(f"New Natural Singletons      : {new_singletons:,} ({new_singletons/total_rows*100:.2f}%)")

    print("\n--- New Match Distribution ---")
    for k in sorted(match_dist.keys()):
        print(f"  {k:>2} matches: {match_dist[k]:>8,d} entities ({match_dist[k]/total_rows*100:>5.2f}%)")
    print("="*80)
    print(f"Total execution time: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
