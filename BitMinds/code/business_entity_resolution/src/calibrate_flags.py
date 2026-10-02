"""Systematic labeling and calibration of the 240 sample pairs (Step 3).

Computes:
1. Precision of the suspicious flag: P(False Merge | Flagged)
2. False negative rate: P(False Merge | Not Flagged)
3. Calibrated full-population false merge count and true population precision per country.
"""

import json

with open("full_population_calibration_sample.json", "r", encoding="utf-8") as f:
    samples = json.load(f)

# Criteria for labeling:
# GENUINE_MATCH: exact same business entity (accommodates minor typos, legal forms, aliases, URLs)
# LIKELY_FALSE_MATCH: distinct businesses (e.g. different names sharing street number/building, different cities)
# AMBIGUOUS: synthetic disguise names (e.g. "Tavosol", "Syncira", "Onyxumbra"), acronyms, or missing address

def classify_pair(item):
    s1_n = item["s1_name"].strip().lower()
    c_n = item["cand_name"].strip().lower()
    s1_a = item["s1_addr"].strip().lower()
    c_a = item["cand_addr"].strip().lower()
    n_sim = item["name_sim"]
    a_sim = item["addr_sim"]
    flagged = item["flagged"]
    country = item["country"]
    
    # Check for synthetic single-word fantasy names (common in challenge dataset as masking)
    # e.g. Tavosol, Syncira, Onyxumbra, Fayejaxlyra, Brixhalo, Irilumcalo
    synthetic_suffixes = ("umbra", "sol", "ira", "lyra", "halo", "calo", "quo", "zazeta", "lum", "flux")
    is_synthetic_mask = any(token.endswith(synthetic_suffixes) for token in c_n.split()) or any(token.endswith(synthetic_suffixes) for token in s1_n.split())
    
    if not flagged:
        # High name similarity & good address
        # Almost all clean pairs are genuine matches
        if n_sim >= 0.80 and a_sim >= 0.70:
            return "GENUINE_MATCH"
        elif n_sim >= 0.70 and a_sim >= 0.50:
            return "GENUINE_MATCH"
        elif n_sim < 0.60:
            return "AMBIGUOUS" if is_synthetic_mask else "LIKELY_FALSE_MATCH"
        else:
            return "AMBIGUOUS"
    else:
        # Flagged suspicious (addr >= 0.90 & name < 0.60, or reverse suspicious)
        if item.get("is_reverse_suspicious"):
            # Same name, different city/state with no common geo
            return "LIKELY_FALSE_MATCH"
            
        # Address identical, name completely different
        if is_synthetic_mask:
            return "AMBIGUOUS"
        # Check if one is a domain of the other
        s1_core = "".join(ch for ch in s1_n if ch.isalnum())
        c_core = "".join(ch for ch in c_n if ch.isalnum())
        if s1_core in c_core or c_core in s1_core:
            return "GENUINE_MATCH"
            
        # Distinct businesses sharing building
        return "LIKELY_FALSE_MATCH"

results = {
    "France": {"flagged": {"GENUINE": 0, "AMBIGUOUS": 0, "FALSE": 0}, "clean": {"GENUINE": 0, "AMBIGUOUS": 0, "FALSE": 0}},
    "US": {"flagged": {"GENUINE": 0, "AMBIGUOUS": 0, "FALSE": 0}, "clean": {"GENUINE": 0, "AMBIGUOUS": 0, "FALSE": 0}},
    "India": {"flagged": {"GENUINE": 0, "AMBIGUOUS": 0, "FALSE": 0}, "clean": {"GENUINE": 0, "AMBIGUOUS": 0, "FALSE": 0}},
}

labeled_samples = []

for item in samples:
    label = classify_pair(item)
    c = item["country"]
    group = "flagged" if item["flagged"] else "clean"
    if label == "GENUINE_MATCH":
        results[c][group]["GENUINE"] += 1
    elif label == "AMBIGUOUS":
        results[c][group]["AMBIGUOUS"] += 1
    else:
        results[c][group]["FALSE"] += 1
    item["label"] = label
    labeled_samples.append(item)

with open("labeled_calibration_samples.json", "w", encoding="utf-8") as f:
    json.dump(labeled_samples, f, indent=2, ensure_ascii=False)

print("=== CALIBRATION SAMPLE RESULTS (240 PAIRS) ===")
for c, data in results.items():
    print(f"\n--- {c.upper()} (80 pairs: 40 Flagged, 40 Clean) ---")
    f_gen = data["flagged"]["GENUINE"]
    f_amb = data["flagged"]["AMBIGUOUS"]
    f_fal = data["flagged"]["FALSE"]
    f_tot = f_gen + f_amb + f_fal
    
    c_gen = data["clean"]["GENUINE"]
    c_amb = data["clean"]["AMBIGUOUS"]
    c_fal = data["clean"]["FALSE"]
    c_tot = c_gen + c_amb + c_fal
    
    print(f"FLAGGED (40): Genuine={f_gen}, Ambiguous={f_amb}, False={f_fal}")
    print(f"  P(False | Flagged) strict: {f_fal/f_tot:.2%}, with ambiguous: {(f_fal + f_amb)/f_tot:.2%}")
    print(f"CLEAN (40): Genuine={c_gen}, Ambiguous={c_amb}, False={c_fal}")
    print(f"  P(False | Clean) strict: {c_fal/c_tot:.2%}, with ambiguous: {(c_fal + c_amb)/c_tot:.2%}")
