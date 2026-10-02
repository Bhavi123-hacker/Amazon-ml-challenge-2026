import os
import sys
import random
import re
import pandas as pd
from rapidfuzz import fuzz

clean_re1 = re.compile(r'[#№°]')
clean_re2 = re.compile(r'\b(no|n°|nº|num|number)\b\.?', re.IGNORECASE)
ocr_re = re.compile(r'(?<=\d)[oO]\b|\b[oO](?=\d)')
tok_re = re.compile(r'\b\d+[a-zA-Z]?\b')
non_digit_re = re.compile(r'\D')

def extract_standard_nums(addr: str) -> set:
    if not addr or not isinstance(addr, str):
        return set()
    c = ocr_re.sub('0', addr)
    c = clean_re1.sub(' ', c)
    c = clean_re2.sub(' ', c)
    c = re.sub(r'(\d+)[-/](\d+)', r'\1 \2', c)
    tokens = tok_re.findall(c)
    nums = set()
    for tok in tokens:
        bd = non_digit_re.sub('', tok).lstrip('0')
        if bd and len(bd) <= 6:
            nums.add(int(bd))
    return nums

def extract_compound_door_nums(addr: str) -> set:
    """Extract compound door/plot numbers like 1/5448, 6/103, 8-14-14, B-65/1."""
    if not addr or not isinstance(addr, str):
        return set()
    c = ocr_re.sub('0', addr)
    c = clean_re1.sub(' ', c)
    c = clean_re2.sub(' ', c)
    # Find patterns like \b\d+[-/]\d+([-/]\d+)?\b
    compounds = re.findall(r'\b\d+[-/]\d+(?:[-/]\d+)?\b', c)
    normalized = set()
    for comp in compounds:
        parts = [p.lstrip('0') or '0' for p in re.split(r'[-/]', comp)]
        normalized.add('/'.join(parts))
    return normalized

def extract_indian_pincode(addr: str) -> str:
    if not addr or not isinstance(addr, str):
        return ""
    # Indian PIN code is 6 digits starting with 1-9
    m = re.findall(r'\b[1-9]\d{5}\b', addr)
    return m[-1] if m else ""

def is_v1_mismatch(addr1: str, addr2: str) -> bool:
    """Variation 1: Current street-number guard as-is."""
    n1 = extract_standard_nums(addr1)
    n2 = extract_standard_nums(addr2)
    return bool(n1 and n2 and n1.isdisjoint(n2))

def is_v2_mismatch(addr1: str, addr2: str) -> bool:
    """Variation 2: If street number absent on either side, check PIN code if available."""
    n1 = extract_standard_nums(addr1)
    n2 = extract_standard_nums(addr2)
    if n1 and n2:
        return n1.isdisjoint(n2)
    # If numbers absent, check PIN
    pin1 = extract_indian_pincode(addr1)
    pin2 = extract_indian_pincode(addr2)
    if pin1 and pin2 and pin1 != pin2:
        return True # PIN mismatch!
    return False

def is_v3_mismatch(addr1: str, addr2: str) -> bool:
    """Variation 3: Standard street guard + Compound door number check + PIN code check."""
    # Check compound door numbers first (e.g. 1/5448 vs 1/5459)
    cd1 = extract_compound_door_nums(addr1)
    cd2 = extract_compound_door_nums(addr2)
    if cd1 and cd2 and cd1.isdisjoint(cd2):
        return True # Compound door mismatch!

    # Check PIN code
    pin1 = extract_indian_pincode(addr1)
    pin2 = extract_indian_pincode(addr2)
    if pin1 and pin2 and pin1 != pin2:
        return True # PIN mismatch!

    # Standard street number
    n1 = extract_standard_nums(addr1)
    n2 = extract_standard_nums(addr2)
    if n1 and n2 and n1.isdisjoint(n2):
        return True

    return False

def main():
    print("=" * 80)
    print("PART A: TESTING INDIA GUARD VARIATIONS ON FRESH 50-PAIR SAMPLE")
    print("=" * 80)

    # Load S1 India entities
    print("Loading test_source1...", flush=True)
    df1 = pd.read_csv("student_resource/dataset/test/test_source1.tsv", sep="\t")
    in_s1_dict = {
        row["entity_id"]: (str(row["business_name"]), str(row["business_address"]))
        for _, row in df1[df1["country"] == "India"].iterrows()
    }
    del df1

    # Load India candidate matches from dynamic_results_India.tsv
    print("Loading dynamic_results_India.tsv...", flush=True)
    in_matches = {}
    with open("output/temp_work/dynamic_results_India.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            sid = parts[0]
            m = [c.strip() for c in parts[2].split(",") if c.strip()] if len(parts) > 2 and parts[2] else []
            if m:
                in_matches[sid] = m

    # Sample ~15 entities with ~50 pairs (new seed 999)
    rng = random.Random(999)
    sampled_sids = rng.sample(list(in_matches.keys()), 25)

    pairs = []
    needed_cids = set()
    for sid in sampled_sids:
        for cid in in_matches[sid]:
            pairs.append((sid, cid))
            needed_cids.add(cid)
            if len(pairs) >= 50:
                break
        if len(pairs) >= 50:
            break

    print(f"Sampled {len(pairs)} pairs across {len(set(p[0] for p in pairs))} entities.")

    # Look up raw text of candidates from S2 and S3
    cand_dict = {}
    print("Looking up candidates in test_source2 and 3...", flush=True)
    for src in ["student_resource/dataset/test/test_source2.tsv", "student_resource/dataset/test/test_source3.tsv"]:
        with open(src, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                cid = parts[0]
                if cid in needed_cids:
                    cand_dict[cid] = (parts[1] if len(parts) > 1 else "", parts[2] if len(parts) > 2 else "")

    print(f"Found text for {len(cand_dict)} / {len(needed_cids)} candidates.\n")

    # Evaluate each pair
    evaluated_pairs = []
    for sid, cid in pairs:
        s1_name, s1_addr = in_s1_dict[sid]
        c_name, c_addr = cand_dict.get(cid, ("", ""))

        rej_v1 = is_v1_mismatch(s1_addr, c_addr)
        rej_v2 = is_v2_mismatch(s1_addr, c_addr)
        rej_v3 = is_v3_mismatch(s1_addr, c_addr)

        evaluated_pairs.append({
            "sid": sid, "cid": cid,
            "s1_name": s1_name, "s1_addr": s1_addr,
            "c_name": c_name, "c_addr": c_addr,
            "rej_v1": rej_v1, "rej_v2": rej_v2, "rej_v3": rej_v3
        })

    # Save to file for manual audit of the 50 pairs
    with open("output/part_a_india_50_pairs_audit.txt", "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("PART A: 50 REAL INDIA PAIRS EVALUATION ACROSS VARIATIONS\n")
        f.write("="*80 + "\n\n")
        for idx, ep in enumerate(evaluated_pairs, 1):
            ns = fuzz.token_sort_ratio(ep['s1_name'].lower(), ep['c_name'].lower())
            asim = fuzz.token_sort_ratio(ep['s1_addr'].lower(), ep['c_addr'].lower())
            f.write(f"[{idx:02d}] {ep['sid']} vs {ep['cid']}\n")
            f.write(f"  S1: {ep['s1_name']} | {ep['s1_addr']}\n")
            f.write(f"  C : {ep['c_name']} | {ep['c_addr']}\n")
            f.write(f"  Sim: NameSim={ns:.1f}, AddrSim={asim:.1f}\n")
            f.write(f"  Rejections: V1={ep['rej_v1']} | V2={ep['rej_v2']} | V3={ep['rej_v3']}\n\n")

    print(f"Total pairs: {len(evaluated_pairs)}")
    print(f"Rejections by V1 (Current Street Guard) : {sum(1 for ep in evaluated_pairs if ep['rej_v1'])}")
    print(f"Rejections by V2 (PIN Fallback)         : {sum(1 for ep in evaluated_pairs if ep['rej_v2'])}")
    print(f"Rejections by V3 (Compound+PIN Guard)   : {sum(1 for ep in evaluated_pairs if ep['rej_v3'])}")
    print("Saved 50 pairs to output/part_a_india_50_pairs_audit.txt")

if __name__ == "__main__":
    main()
