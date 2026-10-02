import os
import sys
import re
import pandas as pd
from rapidfuzz import fuzz

clean_re1 = re.compile(r'[#№°]')
clean_re2 = re.compile(r'\b(no|n°|nº|num|number)\b\.?', re.IGNORECASE)
ocr_re = re.compile(r'(?<=\d)[oO]\b|\b[oO](?=\d)')
tok_re = re.compile(r'\b\d+[a-zA-Z]?\b')
non_digit_re = re.compile(r'\D')

DEV_MAP = {
    '\u0905': 'a', '\u0906': 'aa', '\u0907': 'i', '\u0908': 'ee', '\u0909': 'u', '\u090a': 'oo', '\u090b': 'ri',
    '\u090f': 'e', '\u0910': 'ai', '\u0913': 'o', '\u0914': 'au',
    '\u0915': 'k', '\u0916': 'kh', '\u0917': 'g', '\u0918': 'gh', '\u0919': 'ng',
    '\u091a': 'ch', '\u091b': 'chh', '\u091c': 'j', '\u091d': 'jh', '\u091e': 'ny',
    '\u091f': 't', '\u0920': 'th', '\u0921': 'd', '\u0922': 'dh', '\u0923': 'n',
    '\u0924': 't', '\u0925': 'th', '\u0926': 'd', '\u0927': 'dh', '\u0928': 'n',
    '\u092a': 'p', '\u092b': 'f', '\u092b': 'ph', '\u092c': 'b', '\u092d': 'bh', '\u092e': 'm',
    '\u092f': 'y', '\u0930': 'r', '\u0932': 'l', '\u0935': 'v', '\u0936': 'sh', '\u0937': 'sh', '\u0938': 's', '\u0939': 'h',
    '\u093e': 'a', '\u093f': 'i', '\u0940': 'i', '\u0941': 'u', '\u0942': 'u', '\u0943': 'ri',
    '\u0947': 'e', '\u0948': 'ai', '\u094b': 'o', '\u094c': 'au', '\u094d': '', '\u0902': 'n', '\u0901': 'n', '\u0903': 'h'
}

def transliterate_indic(text: str) -> str:
    if not text:
        return ""
    return "".join(DEV_MAP.get(ch, ch) for ch in text)

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

# Variations:
def eval_v1(name1, addr1, name2, addr2) -> bool:
    """Variation 1: Baseline street number guard as-is."""
    n1 = extract_standard_nums(addr1)
    n2 = extract_standard_nums(addr2)
    return bool(n1 and n2 and n1.isdisjoint(n2))

def eval_v2(name1, addr1, name2, addr2) -> bool:
    """Variation 2: Street guard + PIN mismatch fallback."""
    n1 = extract_standard_nums(addr1)
    n2 = extract_standard_nums(addr2)
    if n1 and n2:
        return n1.isdisjoint(n2)
    p1 = extract_indian_pincode(addr1)
    p2 = extract_indian_pincode(addr2)
    if p1 and p2 and p1 != p2:
        return True
    return False

def eval_v3(name1, addr1, name2, addr2) -> bool:
    """Variation 3: Street guard + Compound Door guard + PIN fallback + Indic Multi-tenant Guard."""
    # 1. Compound door number check
    cd1 = extract_compound_door_nums(addr1)
    cd2 = extract_compound_door_nums(addr2)
    if cd1 and cd2 and cd1.isdisjoint(cd2):
        return True

    # 2. Standard street number check
    n1 = extract_standard_nums(addr1)
    n2 = extract_standard_nums(addr2)
    if n1 and n2 and n1.isdisjoint(n2):
        return True

    # 3. PIN mismatch check
    p1 = extract_indian_pincode(addr1)
    p2 = extract_indian_pincode(addr2)
    if p1 and p2 and p1 != p2:
        return True

    # 4. Multi-tenant collision for India (name sim < 40 and addr sim > 75)
    t1 = transliterate_indic(name1.lower())
    t2 = transliterate_indic(name2.lower())
    ns = fuzz.token_sort_ratio(t1, t2)
    asim = fuzz.token_sort_ratio(addr1.lower(), addr2.lower())
    if ns < 40.0 and asim > 75.0:
        return True

    return False

# Load the 50 pairs from output/part_a_india_50_pairs_audit.txt and label ground truth
with open("output/part_a_india_50_pairs_audit.txt", "r", encoding="utf-8") as f:
    text = f.read()

sections = text.split("--------------------------------------------------------------------------------" if "---" in text else "\n\n")

pairs = []
for block in text.split("\n\n"):
    lines = [l.strip() for l in block.split("\n") if l.strip()]
    if not lines or not lines[0].startswith("["):
        continue
    sid_cid = lines[0].split()[1:]
    sid = sid_cid[0]
    cid = sid_cid[2] if len(sid_cid) > 2 else ""
    s1_line = lines[1].replace("S1:", "").strip()
    s1_name, s1_addr = s1_line.split("|", 1)
    c_line = lines[2].replace("C :", "").strip()
    c_name, c_addr = c_line.split("|", 1) if "|" in c_line else (c_line, "")
    
    pairs.append({
        "sid": sid, "cid": cid,
        "s1_name": s1_name.strip(), "s1_addr": s1_addr.strip(),
        "c_name": c_name.strip(), "c_addr": c_addr.strip()
    })

print(f"Loaded {len(pairs)} pairs for evaluation.")

# Manual Ground Truth Annotations for the 50 pairs based on careful text inspection:
# Pairs:
# [01] Unique Care vs Unique Care (S2-661381476): Genuine
# [02] Unique Care vs Unique Care (S3-816237356): Genuine
# [03] Unique Care vs Unique Care (S2-364228797): Genuine
# [04] Unique Care vs Shri Unique Cee (S2-525034708): Genuine
# [05] Swastik vs Swastik AW 117 (S2-259595429): Genuine
# [06] Swastik vs Swastik AW 117 (S2-398690737): Genuine
# [07] Swastik AW 117 vs Swastik AW 118 (S2-736443302): FALSE (plot AW 118 vs AW 117)
# [08] City Solutions vs City Solutions Tamil (S2-521368560): Genuine
# [09] City Solutions vs Best International Tamil (S2-233495587): FALSE (different company in 60/3)
# [10] City Solutions vs City Solutions Tamil (S3-640661822): Genuine
# [11] City Solutions vs Guljag Foundation (S2-58076072): FALSE (Guljag vs City Solutions)
# [12] City Solutions vs Guljag Foundation (S2-82929136): FALSE (Guljag vs City Solutions)
# [13] City Solutions vs Guljag Foundation (S2-335067405): FALSE (Guljag vs City Solutions)
# [14] Wow Advisory vs Wow Advisory 6374 (S2-252037762): Genuine
# [15] Wow Advisory vs Wow Advisory 6374 (S2-214501807): Genuine
# [16] Wow Advisory 6374 vs Wow Advisory 6385 (S3-144711727): FALSE (Flat 6385 vs 6374)
# [17] Gold Finance vs Gold Finance C-76 (S2-934665768): Genuine
# [18] Gold Finance vs Gold Finance C-76 (S3-922970895): Genuine
# [19] Gold Finance C-76 vs Gold Finance C-59 (S3-253485166): FALSE (C-59 vs C-76)
# [20] Tech Builders vs Tech Builders 170 (S2-235695164): Genuine
# [21] Best Agro vs Best Agro (S3-684428223): Genuine
# [22] Best Agro vs Best (S3-559543610): Ambiguous / False (truncated name "Best" at same address)
# [23] Faridabad Ur vs Faridabad Ur Plot 3 (S3-118894596): Genuine
# [24] Faridabad Ur vs Faridabad Ur Plot 3 (S3-439213778): Genuine
# [25] Faridabad Ur vs Faridabad Ur Plot 12 (S3-987904858): FALSE (Plot 12 vs Plot 3)
# [26] Faridabad Ur vs Faridabad Ur Plot 3 (S2-362652934): Genuine
# [27] Faridabad Ur vs Faridabad Ur Plot 12 (S3-144365733): FALSE (Plot 12 vs Plot 3)
# [28] Sarasesh Services vs Sarasesh (S2-782053774): Genuine
# [29] Sarasesh Services vs saraseshservices.com (S2-268130084): Genuine
# [30] Sarasesh Services vs 5arasesh Services (S2-74998437): Genuine
# [31] Sarasesh Services vs Sarasesh No 122 29/1 (S3-247700875): Genuine
# [32] Sarasesh Services vs Sarasesh (S3-738455087): Genuine
# [33] Sarasesh Services vs One Shakti Services (S2-431660426): FALSE (One Shakti in different area)
# [34] Shiva Global 16-44 vs Shiva Global 16-48 (S3-870866106): FALSE (16-48 vs 16-44)
# [35] Shiva Global 16-44 vs Shiva Global 16-44 (S3-732876307): Genuine
# [36] Sharma Iron vs Sharma Iron (S3-426943490): Genuine
# [37] Sharma Iron vs Sharma Iron (S3-54868008): Genuine
# [38] Sharma Iron vs Sharma Iron (S2-989599872): Genuine
# [39] Raviendra Builders vs Raviendra (S2-978913141): Genuine
# [40] Raviendra Builders vs Raviendra H.no 31 (S3-172417470): Genuine
# [41] Ashwin Land 3231 vs Ashwin Land 3231 (S2-850808273): Genuine
# [42] Ashwin Land 3231 vs Ashwin Land 3231 (S3-893073501): Genuine
# [43] Ashwin Land 3231 vs Ashwin Manufactures 3232 (S3-276645515): FALSE (Plot 3232 vs 3231)
# [44] Ashwin Land 3231 vs Ashwin Marketing 03232 (S2-604616593): FALSE (Plot 3232 vs 3231)
# [45] Finance Memon 32/C vs Finance Memon 32/C (S3-837816768): Genuine
# [46] Finance Memon 32/C vs Finance Memon 32/C (S2-752052521): Genuine
# [47] Finance Memon 32/C vs Finance Memon 32/C (S2-741278583): Genuine
# [48] Finance Memon 32/C vs Finance Memon 41/ (S2-518693425): FALSE (Plot 41/ vs 32/C)
# [49] Finance Memon 32/C vs Finance Memon 41/ (S2-457431757): FALSE (Plot 41/ vs 32/C)
# [50] Finance Memon 32/C vs Finance Memon No 9-20 (S3-624067488): FALSE (different address 9-20 vs 32/C)

ground_truth = [
    "GENUINE", "GENUINE", "GENUINE", "GENUINE", "GENUINE", "GENUINE", "FALSE",
    "GENUINE", "FALSE", "GENUINE", "FALSE", "FALSE", "FALSE",
    "GENUINE", "GENUINE", "FALSE",
    "GENUINE", "GENUINE", "FALSE",
    "GENUINE",
    "GENUINE", "AMBIGUOUS",
    "GENUINE", "GENUINE", "FALSE", "GENUINE", "FALSE",
    "GENUINE", "GENUINE", "GENUINE", "GENUINE", "GENUINE", "FALSE",
    "FALSE", "GENUINE",
    "GENUINE", "GENUINE", "GENUINE",
    "GENUINE", "GENUINE",
    "GENUINE", "GENUINE", "FALSE", "FALSE",
    "GENUINE", "GENUINE", "GENUINE", "FALSE", "FALSE", "FALSE"
]

print(f"Ground Truth counts: {ground_truth.count('GENUINE')} Genuine, {ground_truth.count('FALSE')} False, {ground_truth.count('AMBIGUOUS')} Ambiguous.")

def evaluate_variation(name, eval_fn):
    kept_genuine = 0
    kept_false = 0
    kept_ambig = 0
    rej_genuine = 0
    rej_false = 0
    rej_ambig = 0

    for p, gt in zip(pairs, ground_truth):
        rej = eval_fn(p["s1_name"], p["s1_addr"], p["c_name"], p["c_addr"])
        if rej:
            if gt == "GENUINE": rej_genuine += 1
            elif gt == "FALSE": rej_false += 1
            else: rej_ambig += 1
        else:
            if gt == "GENUINE": kept_genuine += 1
            elif gt == "FALSE": kept_false += 1
            else: kept_ambig += 1

    total_kept = kept_genuine + kept_false + kept_ambig
    strict_p = kept_genuine / total_kept if total_kept else 0
    eff_p = (kept_genuine + 0.5 * kept_ambig) / total_kept if total_kept else 0

    print(f"\n--- {name} ---")
    print(f"  Total Pairs Kept       : {total_kept} / {len(pairs)}")
    print(f"  False Pairs Caught     : {rej_false} / {ground_truth.count('FALSE')} ({rej_false/ground_truth.count('FALSE')*100:.1f}%)")
    print(f"  False Rejections (Lost): {rej_genuine} / {ground_truth.count('GENUINE')}")
    print(f"  Strict Precision       : {strict_p*100:.2f}% ({kept_genuine}/{total_kept})")
    print(f"  Effective Precision    : {eff_p*100:.2f}%")

print("="*80)
print("PERFORMANCE COMPARISON ON 50 REAL INDIA PAIRS")
print("="*80)
evaluate_variation("Raw Model (No Guard)", lambda n1, a1, n2, a2: False)
evaluate_variation("Variation 1: Current Street Number Guard", eval_v1)
evaluate_variation("Variation 2: Street Guard + PIN Fallback", eval_v2)
evaluate_variation("Variation 3: Street + Compound Door + PIN + Multi-tenant Guard", eval_v3)
