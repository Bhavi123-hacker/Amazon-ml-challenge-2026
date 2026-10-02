"""Reconciliation and Calibration of Expanded Audit Heuristic.

Draws a FRESH, purely random (unstratified) sample of 60 pairs per country (180 total),
looks up their raw unnormalized data, applies both Old and Expanded Heuristics,
and reconciles the implied population false rates against plain random sample ground truth.
"""

import json
import os
import random
import re
import sys
from typing import Any, Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

RANDOM_SEED = 2026
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

LEGAL_STOPWORDS = {
    "sas", "sarl", "sasu", "sa", "sci", "eurl", "ei", "gmbh", "llc", "inc",
    "corp", "ltd", "pvt", "limited", "co", "company", "plc", "association",
    "club", "france", "comite", "ecole", "ets", "etablissements", "societe",
    "groupement", "center", "centre", "services", "solutions", "holdings",
    "group", "enterprises", "industries", "trading", "corporation", "incorporated",
    "cie", "freres", "associates", "associés", "bureau", "agence"
}

def extract_core_name(name: str) -> str:
    """Normalize and remove common legal entity forms and business stopwords."""
    if not name:
        return ""
    # remove punctuation
    clean = re.sub(r"[^\w\s]", " ", name.lower())
    tokens = clean.split()
    core_tokens = [t for t in tokens if t not in LEGAL_STOPWORDS and len(t) > 1]
    return " ".join(core_tokens) if core_tokens else clean.strip()

def compute_token_jaccard(s1: str, s2: str) -> float:
    t1 = set(s1.split())
    t2 = set(s2.split())
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)

def extract_street_number(addr: str) -> str:
    m = re.search(r"\b(\d+[a-zA-Z]?)\b", addr)
    return m.group(1).lower() if m else ""

def evaluate_heuristics(s1_name: str, s1_addr: str, c_name: str, c_addr: str) -> Dict[str, Any]:
    """Compute features and evaluate Old vs Expanded Heuristics."""
    s1_name_clean = s1_name.lower().strip()
    c_name_clean = c_name.lower().strip()
    s1_addr_clean = s1_addr.lower().strip()
    c_addr_clean = c_addr.lower().strip()

    name_lev = fuzz.ratio(s1_name_clean, c_name_clean) / 100.0
    name_tok_sort = fuzz.token_sort_ratio(s1_name_clean, c_name_clean) / 100.0
    name_jaccard = compute_token_jaccard(s1_name_clean, c_name_clean)

    addr_lev = fuzz.ratio(s1_addr_clean, c_addr_clean) / 100.0
    addr_tok_sort = fuzz.token_sort_ratio(s1_addr_clean, c_addr_clean) / 100.0
    addr_jaccard = compute_token_jaccard(s1_addr_clean, c_addr_clean)

    # Core names
    s1_core = extract_core_name(s1_name)
    c_core = extract_core_name(c_name)
    core_name_lev = fuzz.ratio(s1_core, c_core) / 100.0 if s1_core and c_core else name_lev
    core_name_jaccard = compute_token_jaccard(s1_core, c_core) if s1_core and c_core else name_jaccard

    # Street numbers
    num1 = extract_street_number(s1_addr)
    num2 = extract_street_number(c_addr)
    num_match = (num1 == num2) if (num1 and num2) else False

    # Old Heuristic
    old_suspicious = (addr_lev >= 0.90 or addr_tok_sort >= 0.90) and (name_lev < 0.60 and name_tok_sort < 0.60)
    old_reverse = (name_lev >= 0.85 or name_tok_sort >= 0.85) and (addr_lev < 0.40 and addr_tok_sort < 0.40)
    old_flagged = old_suspicious or old_reverse

    # Expanded Heuristic
    # Failure Mode 1: Core Name Mismatch with High Address (e.g. co-located, shared office / mall)
    flag_colocation = (addr_tok_sort >= 0.85 or (num_match and addr_jaccard >= 0.50)) and (core_name_lev < 0.60 and core_name_jaccard < 0.45)

    # Failure Mode 2: Suffix Inflation (raw name looks okay ~0.65-0.75, but core name is totally distinct)
    flag_suffix_inflation = (core_name_jaccard == 0.0 or core_name_lev < 0.40) and (core_name_lev < name_lev - 0.20)

    # Failure Mode 3: Different location / Franchise
    flag_franchise = (name_tok_sort >= 0.80 or core_name_lev >= 0.85) and (addr_tok_sort < 0.50 and addr_jaccard < 0.35)

    # Failure Mode 4: Same building number, but street name completely diverges
    flag_street_mismatch = num_match and (addr_jaccard < 0.30 and addr_lev < 0.55) and (name_tok_sort < 0.80)

    # Failure Mode 5: Low Mutual Agreement (both name and addr mediocre)
    flag_low_mutual = (name_tok_sort < 0.70 and core_name_lev < 0.65) and (addr_tok_sort < 0.70)

    expanded_flagged = bool(
        flag_colocation or
        flag_suffix_inflation or
        flag_franchise or
        flag_street_mismatch or
        flag_low_mutual or
        old_flagged
    )

    return {
        "name_lev": round(name_lev, 4),
        "name_tok_sort": round(name_tok_sort, 4),
        "addr_lev": round(addr_lev, 4),
        "addr_tok_sort": round(addr_tok_sort, 4),
        "core_name_lev": round(core_name_lev, 4),
        "core_name_jaccard": round(core_name_jaccard, 4),
        "num_match": num_match,
        "old_flagged": old_flagged,
        "expanded_flagged": expanded_flagged,
        "flag_colocation": bool(flag_colocation),
        "flag_suffix_inflation": bool(flag_suffix_inflation),
        "flag_franchise": bool(flag_franchise),
        "flag_street_mismatch": bool(flag_street_mismatch),
        "flag_low_mutual": bool(flag_low_mutual),
    }

def main():
    print("=== Step 1: Loading results and extracting fresh random pairs ===", flush=True)
    countries = ["France", "US", "India"]
    fresh_sample_by_country = {}

    for c in countries:
        res_path = f"output/temp_work/results_{c}.tsv"
        df = pd.read_csv(res_path, sep="\t")
        pairs = []
        for _, row in df.iterrows():
            s1_id = row["source1_entity_id"]
            m_str = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
            if m_str.strip():
                for cid in m_str.split(","):
                    cid = cid.strip()
                    if cid:
                        pairs.append((s1_id, cid))
        
        print(f"{c}: Total matched pairs = {len(pairs):,}", flush=True)
        # Random sample of 60 pairs
        chosen = random.sample(pairs, 60)
        fresh_sample_by_country[c] = chosen

    # Collect needed IDs
    needed_s1 = set()
    needed_s2 = set()
    needed_s3 = set()
    for c, pairs in fresh_sample_by_country.items():
        for s1, cid in pairs:
            needed_s1.add(s1)
            if cid.startswith("S2-"):
                needed_s2.add(cid)
            elif cid.startswith("S3-"):
                needed_s3.add(cid)

    print(f"Needed entity IDs: S1={len(needed_s1)}, S2={len(needed_s2)}, S3={len(needed_s3)}", flush=True)

    # Load raw records
    s1_map = {}
    for chunk in pd.read_csv("student_resource/dataset/test/test_source1.tsv", sep="\t", chunksize=100000, keep_default_na=False):
        m = chunk[chunk["entity_id"].isin(needed_s1)]
        for _, r in m.iterrows():
            s1_map[r["entity_id"]] = {"name": r["business_name"], "addr": r["business_address"], "country": r.get("country", "")}
        if len(s1_map) == len(needed_s1):
            break

    cand_map = {}
    for chunk in pd.read_csv("student_resource/dataset/test/test_source2.tsv", sep="\t", chunksize=100000, keep_default_na=False):
        m = chunk[chunk["entity_id"].isin(needed_s2)]
        for _, r in m.iterrows():
            cand_map[r["entity_id"]] = {"name": r["business_name"], "addr": r["business_address"], "country": r.get("country", "")}
        if len([k for k in cand_map if k.startswith("S2-")]) == len(needed_s2):
            break

    for chunk in pd.read_csv("student_resource/dataset/test/test_source3.tsv", sep="\t", chunksize=100000, keep_default_na=False):
        m = chunk[chunk["entity_id"].isin(needed_s3)]
        for _, r in m.iterrows():
            cand_map[r["entity_id"]] = {"name": r["business_name"], "addr": r["business_address"], "country": r.get("country", "")}
        if len([k for k in cand_map if k.startswith("S3-")]) == len(needed_s3):
            break

    print(f"Loaded raw records: S1={len(s1_map)}, Cands={len(cand_map)}", flush=True)

    # Process pairs and label
    all_evaluated = []
    for c in countries:
        pairs = fresh_sample_by_country[c]
        for s1, cid in pairs:
            s1_rec = s1_map.get(s1, {"name": "", "addr": "", "country": c})
            c_rec = cand_map.get(cid, {"name": "", "addr": "", "country": c})
            h_res = evaluate_heuristics(s1_rec["name"], s1_rec["addr"], c_rec["name"], c_rec["addr"])
            
            entry = {
                "country": c,
                "s1_id": s1,
                "cand_id": cid,
                "s1_name": s1_rec["name"],
                "s1_addr": s1_rec["addr"],
                "cand_name": c_rec["name"],
                "cand_addr": c_rec["addr"],
                **h_res
            }
            all_evaluated.append(entry)

    # Save to JSON for exact labeling
    with open("fresh_random_sample_raw.json", "w", encoding="utf-8") as f:
        json.dump(all_evaluated, f, indent=2, ensure_ascii=False)

    print("Saved 180 fresh pairs to fresh_random_sample_raw.json", flush=True)

if __name__ == "__main__":
    main()
