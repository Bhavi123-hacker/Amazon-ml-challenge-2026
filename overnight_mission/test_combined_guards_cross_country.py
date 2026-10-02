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

PH_NAMES = {'', 'null', 'n/a', 'none', '-', '.', 'unknown', 'na', '<null>', 'no name', 'nan'}

def is_placeholder(name: str) -> bool:
    if not name or not isinstance(name, str):
        return True
    return name.strip().lower() in PH_NAMES

def is_latin_text(text: str) -> bool:
    alpha = [ch for ch in text if ch.isalpha()]
    if not alpha: return True
    return sum(1 for ch in alpha if ord(ch) < 128) / len(alpha) > 0.8

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
    if not addr or not isinstance(addr, str):
        return set()
    c = ocr_re.sub('0', addr)
    c = clean_re1.sub(' ', c)
    c = clean_re2.sub(' ', c)
    compounds = re.findall(r'\b\d+[-/]\d+(?:[-/]\d+)?\b', c)
    normalized = set()
    for comp in compounds:
        parts = [p.lstrip('0') or '0' for p in re.split(r'[-/]', comp)]
        normalized.add('/'.join(parts))
    return normalized

def extract_indian_pincode(addr: str) -> str:
    if not addr or not isinstance(addr, str):
        return ""
    m = re.findall(r'\b[1-9]\d{5}\b', addr)
    return m[-1] if m else ""

def combined_guard_reject(country: str, name1: str, addr1: str, name2: str, addr2: str) -> tuple:
    """Evaluate full suite of validated guards."""
    # Guard 1: Placeholder-name guard
    if is_placeholder(name2):
        asim = fuzz.token_sort_ratio(addr1.lower(), addr2.lower())
        if asim < 85.0:
            return True, "PLACEHOLDER_LOW_ADDR_SIM"

    # Guard 2: Street number mismatch
    n1 = extract_standard_nums(addr1)
    n2 = extract_standard_nums(addr2)
    if n1 and n2 and n1.isdisjoint(n2):
        return True, "STREET_NUMBER_MISMATCH"

    # Country-specific guards
    if country == "India":
        # Guard 3: Compound door number mismatch
        cd1 = extract_compound_door_nums(addr1)
        cd2 = extract_compound_door_nums(addr2)
        if cd1 and cd2 and cd1.isdisjoint(cd2):
            return True, "COMPOUND_DOOR_MISMATCH"

        # Guard 4: PIN code mismatch
        pin1 = extract_indian_pincode(addr1)
        pin2 = extract_indian_pincode(addr2)
        if pin1 and pin2 and pin1 != pin2:
            return True, "PIN_MISMATCH"

        # Guard 5: Multi-tenant collision for Latin names in India
        if is_latin_text(name1) and is_latin_text(name2):
            ns = fuzz.token_sort_ratio(name1.lower(), name2.lower())
            asim = fuzz.token_sort_ratio(addr1.lower(), addr2.lower())
            if ns < 40.0 and asim > 75.0:
                return True, "INDIA_MULTI_TENANT_COLLISION"

    elif country == "France":
        # Guard 5: France Guard D (Multi-tenant collision)
        ns = fuzz.token_sort_ratio(name1.lower(), name2.lower())
        asim = fuzz.token_sort_ratio(addr1.lower(), addr2.lower())
        if ns < 45.0 and asim > 70.0:
            return True, "FRANCE_GUARD_D"

    elif country == "US":
        # Guard 5: US Multi-tenant collision (protect short acronyms len <= 4)
        if len(name1) > 4 and len(name2) > 4:
            ns = fuzz.token_sort_ratio(name1.lower(), name2.lower())
            asim = fuzz.token_sort_ratio(addr1.lower(), addr2.lower())
            if ns < 40.0 and asim > 75.0:
                return True, "US_MULTI_TENANT_COLLISION"

    return False, "ACCEPT"

def main():
    print("=" * 80)
    print("PART C: CROSS-COUNTRY VALIDATION OF COMBINED GUARDS (75 FRESH PAIRS)")
    print("=" * 80)

    # 1. Load S1
    print("Loading test_source1...", flush=True)
    df1 = pd.read_csv("student_resource/dataset/test/test_source1.tsv", sep="\t")
    s1_dict = {
        row["entity_id"]: {
            "name": str(row["business_name"]),
            "addr": str(row["business_address"]),
            "country": str(row["country"])
        }
        for _, row in df1.iterrows()
    }
    del df1

    # 2. Load existing predictions per country from output/temp_work/dynamic_results_*.tsv
    print("Loading dynamic candidate results...", flush=True)
    matches = {"France": {}, "US": {}, "India": {}}
    for c in ["France", "US", "India"]:
        p = f"output/temp_work/dynamic_results_{c}.tsv"
        with open(p, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                sid = parts[0]
                m = [x.strip() for x in parts[2].split(",") if x.strip()] if len(parts) > 2 and parts[2] else []
                if m:
                    matches[c][sid] = m

    # 3. Sample 25 pairs per country with reproducible seed 5555
    rng = random.Random(5555)
    sample_pairs = []
    needed_cids = set()

    for c in ["France", "US", "India"]:
        country_sids = rng.sample(list(matches[c].keys()), 30)
        c_pairs = []
        for sid in country_sids:
            for cid in matches[c][sid]:
                c_pairs.append((c, sid, cid))
                needed_cids.add(cid)
                if len(c_pairs) >= 25:
                    break
            if len(c_pairs) >= 25:
                break
        sample_pairs.extend(c_pairs)
        print(f"  {c}: sampled {len(c_pairs)} pairs across {len(set(p[1] for p in c_pairs))} entities.")

    # 4. Look up candidates from S2 and S3
    cand_dict = {}
    print(f"Looking up {len(needed_cids)} candidates across S2 and S3...", flush=True)
    for src in ["student_resource/dataset/test/test_source2.tsv", "student_resource/dataset/test/test_source3.tsv"]:
        with open(src, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                cid = parts[0]
                if cid in needed_cids:
                    cand_dict[cid] = (parts[1] if len(parts) > 1 else "", parts[2] if len(parts) > 2 else "")

    # 5. Evaluate all 75 pairs
    eval_results = []
    for c, sid, cid in sample_pairs:
        s1 = s1_dict[sid]
        c_name, c_addr = cand_dict.get(cid, ("", ""))
        rej, reason = combined_guard_reject(c, s1["name"], s1["addr"], c_name, c_addr)
        eval_results.append({
            "country": c, "sid": sid, "cid": cid,
            "s1_name": s1["name"], "s1_addr": s1["addr"],
            "c_name": c_name, "c_addr": c_addr,
            "rej": rej, "reason": reason
        })

    # Save to file for manual audit
    with open("output/part_c_combined_guards_audit.txt", "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("PART C: 75 REAL CROSS-COUNTRY PAIRS EVALUATION WITH ALL COMBINED GUARDS\n")
        f.write("="*80 + "\n\n")
        for idx, ep in enumerate(eval_results, 1):
            ns = fuzz.token_sort_ratio(ep['s1_name'].lower(), ep['c_name'].lower())
            asim = fuzz.token_sort_ratio(ep['s1_addr'].lower(), ep['c_addr'].lower())
            f.write(f"[{idx:02d}] [{ep['country']}] {ep['sid']} vs {ep['cid']} -> {'REJECTED (' + ep['reason'] + ')' if ep['rej'] else 'ACCEPTED'}\n")
            f.write(f"  S1: {ep['s1_name']} | {ep['s1_addr']}\n")
            f.write(f"  C : {ep['c_name']} | {ep['c_addr']}\n")
            f.write(f"  Metrics: NameSim={ns:.1f}, AddrSim={asim:.1f}\n\n")

    print(f"\nAudit saved to output/part_c_combined_guards_audit.txt")
    print(f"Total pairs: {len(eval_results)}")
    print(f"Accepted: {sum(1 for ep in eval_results if not ep['rej'])} ({sum(1 for ep in eval_results if not ep['rej'])/len(eval_results)*100:.1f}%)")
    print(f"Rejected: {sum(1 for ep in eval_results if ep['rej'])} ({sum(1 for ep in eval_results if ep['rej'])/len(eval_results)*100:.1f}%)")

if __name__ == "__main__":
    main()
