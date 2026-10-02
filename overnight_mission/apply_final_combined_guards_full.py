import os
import sys
import time
import re
import pandas as pd
from rapidfuzz import fuzz

clean_re1 = re.compile(r'[#№°]')
clean_re2 = re.compile(r'\b(no|n°|nº|num|number)\b\.?', re.IGNORECASE)
ocr_re = re.compile(r'(?<=\d)[oO]\b|\b[oO](?=\d)')
tok_re = re.compile(r'\b\d+[a-zA-Z]?\b')
non_digit_re = re.compile(r'\D')

PH_NAMES = {'', 'null', 'n/a', 'none', '-', '.', 'unknown', 'na', '<null>', 'no name', 'nan'}

def is_placeholder(name: str) -> bool:
    if not name or not isinstance(name, str):
        return True
    return name.strip().lower() in PH_NAMES

def is_latin_text(text: str) -> bool:
    alpha = [ch for ch in text if ch.isalpha()]
    if not alpha: return True
    return sum(1 for ch in alpha if ord(ch) < 128) / len(alpha) > 0.8

def extract_standard_nums(addr: str) -> tuple:
    if not addr or not isinstance(addr, str):
        return ()
    c = ocr_re.sub('0', addr)
    c = clean_re1.sub(' ', c)
    c = clean_re2.sub(' ', c)
    c = re.sub(r'(\d+)[-/](\d+)', r'\1 \2', c)
    tokens = tok_re.findall(c)
    nums = []
    for tok in tokens:
        bd = non_digit_re.sub('', tok).lstrip('0')
        if bd and len(bd) <= 6:
            nums.append(int(bd))
    return tuple(set(nums))

def extract_compound_door_nums(addr: str) -> tuple:
    if not addr or not isinstance(addr, str):
        return ()
    c = ocr_re.sub('0', addr)
    c = clean_re1.sub(' ', c)
    c = clean_re2.sub(' ', c)
    compounds = re.findall(r'\b\d+[-/]\d+(?:[-/]\d+)?\b', c)
    normalized = set()
    for comp in compounds:
        parts = [p.lstrip('0') or '0' for p in re.split(r'[-/]', comp)]
        normalized.add('/'.join(parts))
    return tuple(normalized)

def extract_indian_pincode(addr: str) -> str:
    if not addr or not isinstance(addr, str):
        return ""
    m = re.findall(r'\b[1-9]\d{5}\b', addr)
    return m[-1] if m else ""

def is_street_mismatch(nums1: tuple, nums2: tuple) -> bool:
    if not nums1 or not nums2:
        return False
    for n in nums1:
        if n in nums2:
            return False
    return True

def is_compound_mismatch(c1: tuple, c2: tuple) -> bool:
    if not c1 or not c2:
        return False
    for n in c1:
        if n in c2:
            return False
    return True

def main():
    t0 = time.time()
    print("=" * 80)
    print("APPLYING FINAL COMPREHENSIVE COMBINED GUARDS TO MASTER SUBMISSION")
    print("=" * 80)

    # 1. Load S1 data
    print("Pass 1: Loading test_source1...", flush=True)
    df1 = pd.read_csv('student_resource/dataset/test/test_source1.tsv', sep='\t')
    s1_dict = {}
    for row in df1.itertuples(index=False):
        sid = row.entity_id
        name = str(row.business_name) if pd.notna(row.business_name) else ""
        addr = str(row.business_address) if pd.notna(row.business_address) else ""
        country = str(row.country)
        s1_dict[sid] = (
            name,
            addr,
            country,
            extract_standard_nums(addr),
            extract_compound_door_nums(addr) if country == "India" else (),
            extract_indian_pincode(addr) if country == "India" else ""
        )
    del df1
    print(f"  Source 1 loaded: {len(s1_dict):,} entities in {time.time()-t0:.1f}s", flush=True)

    # 2. Identify candidate IDs currently matched
    t1 = time.time()
    print("Pass 2: Scanning matched candidate IDs from matching_results.tsv...", flush=True)
    in_matching = "output/matching_results.tsv"
    needed_cids = set()
    s1_matches = []
    with open(in_matching, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            sid = parts[0]
            matches = [c.strip() for c in parts[1].split(",") if c.strip()] if len(parts) > 1 and parts[1] else []
            s1_matches.append((sid, matches))
            for cid in matches:
                needed_cids.add(cid)
    print(f"  Unique candidates to load: {len(needed_cids):,} in {time.time()-t1:.1f}s", flush=True)

    # 3. Load Cand text and precompute features
    t2 = time.time()
    print("Pass 3: Loading needed candidate records from test_source2 & 3...", flush=True)
    cand_dict = {}
    for src in ["student_resource/dataset/test/test_source2.tsv", "student_resource/dataset/test/test_source3.tsv"]:
        print(f"  Scanning {src}...", flush=True)
        with open(src, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                cid = parts[0]
                if cid in needed_cids:
                    cname = parts[1] if len(parts) > 1 else ""
                    caddr = parts[2] if len(parts) > 2 else ""
                    cand_dict[cid] = (
                        cname,
                        caddr,
                        extract_standard_nums(caddr),
                        extract_compound_door_nums(caddr),
                        extract_indian_pincode(caddr)
                    )
    print(f"  Candidates loaded: {len(cand_dict):,} in {time.time()-t2:.1f}s", flush=True)

    # 4. Stream and filter matching results
    t3 = time.time()
    print("\nPass 4: Applying combined guard rules across all entities...", flush=True)
    out_matching = "output/matching_results_final_guarded.tsv"
    
    total_entities = len(s1_matches)
    total_orig_matches = 0
    total_kept_matches = 0
    rej_counts = {
        "PLACEHOLDER": 0,
        "STREET_NUM": 0,
        "COMPOUND_DOOR": 0,
        "PIN": 0,
        "MULTI_TENANT": 0
    }
    singletons = 0
    match_dist = {}

    with open(out_matching, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        for sid, matches in s1_matches:
            s1_info = s1_dict.get(sid)
            if not s1_info:
                f_out.write(f"{sid}\t{','.join(matches)}\n")
                continue

            s1_name, s1_addr, country, s1_nums, s1_compounds, s1_pin = s1_info
            total_orig_matches += len(matches)
            kept = []

            for cid in matches:
                c_info = cand_dict.get(cid)
                if not c_info:
                    kept.append(cid)
                    continue

                c_name, c_addr, c_nums, c_compounds, c_pin = c_info
                rejected = False

                # Guard 1: Placeholder name check
                if is_placeholder(c_name):
                    asim = fuzz.token_sort_ratio(s1_addr.lower(), c_addr.lower())
                    if asim < 85.0:
                        rej_counts["PLACEHOLDER"] += 1
                        rejected = True

                # Guard 2: Street number mismatch
                if not rejected and is_street_mismatch(s1_nums, c_nums):
                    rej_counts["STREET_NUM"] += 1
                    rejected = True

                # Country-specific guards
                if not rejected and country == "India":
                    # Guard 3: Compound door mismatch
                    if is_compound_mismatch(s1_compounds, c_compounds):
                        rej_counts["COMPOUND_DOOR"] += 1
                        rejected = True

                    # Guard 4: PIN code mismatch
                    elif s1_pin and c_pin and s1_pin != c_pin:
                        rej_counts["PIN"] += 1
                        rejected = True

                    # Guard 5: Multi-tenant collision for Latin names
                    elif is_latin_text(s1_name) and is_latin_text(c_name):
                        ns = fuzz.token_sort_ratio(s1_name.lower(), c_name.lower())
                        asim = fuzz.token_sort_ratio(s1_addr.lower(), c_addr.lower())
                        if ns < 40.0 and asim > 75.0:
                            rej_counts["MULTI_TENANT"] += 1
                            rejected = True

                elif not rejected and country == "France":
                    ns = fuzz.token_sort_ratio(s1_name.lower(), c_name.lower())
                    asim = fuzz.token_sort_ratio(s1_addr.lower(), c_addr.lower())
                    if ns < 45.0 and asim > 70.0:
                        rej_counts["MULTI_TENANT"] += 1
                        rejected = True

                elif not rejected and country == "US":
                    if len(s1_name) > 4 and len(c_name) > 4:
                        ns = fuzz.token_sort_ratio(s1_name.lower(), c_name.lower())
                        asim = fuzz.token_sort_ratio(s1_addr.lower(), c_addr.lower())
                        if ns < 40.0 and asim > 75.0:
                            rej_counts["MULTI_TENANT"] += 1
                            rejected = True

                if not rejected:
                    kept.append(cid)

            k = len(kept)
            total_kept_matches += k
            if k == 0:
                singletons += 1
            match_dist[k] = match_dist.get(k, 0) + 1

            f_out.write(f"{sid}\t{','.join(kept)}\n")

    total_rejected = sum(rej_counts.values())
    print("\n" + "=" * 80)
    print("FINAL COMBINED GUARDS EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total S1 Entities           : {total_entities:,}")
    print(f"Input Total Matches         : {total_orig_matches:,} (avg {total_orig_matches/total_entities:.2f}/entity)")
    print(f"Total Rejected Matches      : {total_rejected:,} ({total_rejected/total_orig_matches*100:.2f}%)")
    print(f"  - Street Number Mismatches : {rej_counts['STREET_NUM']:,}")
    print(f"  - Compound Door Mismatches : {rej_counts['COMPOUND_DOOR']:,}")
    print(f"  - PIN Code Mismatches      : {rej_counts['PIN']:,}")
    print(f"  - Multi-Tenant Collisions  : {rej_counts['MULTI_TENANT']:,}")
    print(f"  - Placeholder False Traps  : {rej_counts['PLACEHOLDER']:,}")
    print(f"Kept Master Matches         : {total_kept_matches:,} (avg {total_kept_matches/total_entities:.2f}/entity)")
    print(f"Natural Singletons          : {singletons:,} ({singletons/total_entities*100:.2f}%)")

    print("\n--- Final Match Distribution ---")
    for k in sorted(match_dist.keys()):
        print(f"  {k:>2} matches: {match_dist[k]:>8,d} entities ({match_dist[k]/total_entities*100:>5.2f}%)")
    print("=" * 80)
    print(f"Total filtering runtime: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
