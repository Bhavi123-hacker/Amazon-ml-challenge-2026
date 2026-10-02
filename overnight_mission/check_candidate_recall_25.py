import os
import sys
import random
import re
import time
import pandas as pd
from rapidfuzz import fuzz

def main():
    print("=" * 80)
    print("TASK 2: RECALL & BLOCKING COVERAGE AUDIT ON 25 ENTITIES")
    print("=" * 80)

    # 1. Load S1 entities
    print("Loading test_source1...", flush=True)
    df1 = pd.read_csv("student_resource/dataset/test/test_source1.tsv", sep="\t")
    
    # Stratified pick of 25 entities (8 France, 8 US, 9 India) with fixed seed 12345
    rng = random.Random(12345)
    fr_sids = rng.sample(df1[df1["country"] == "France"]["entity_id"].tolist(), 8)
    us_sids = rng.sample(df1[df1["country"] == "US"]["entity_id"].tolist(), 8)
    in_sids = rng.sample(df1[df1["country"] == "India"]["entity_id"].tolist(), 9)

    selected_sids = fr_sids + us_sids + in_sids
    df_selected = df1[df1["entity_id"].isin(selected_sids)].copy()
    
    s1_dict = {}
    by_country = {}
    for _, row in df_selected.iterrows():
        sid = row["entity_id"]
        c = row["country"]
        name = str(row["business_name"])
        addr = str(row["business_address"])
        # Name tokens of length >= 3
        name_tokens = set(re.findall(r"\w{3,}", name.lower()))
        # Address tokens of length >= 3
        addr_tokens = set(re.findall(r"\w{3,}", addr.lower()))
        
        info = {
            "sid": sid,
            "name": name,
            "addr": addr,
            "country": c,
            "name_tokens": name_tokens,
            "addr_tokens": addr_tokens
        }
        s1_dict[sid] = info
        if c not in by_country:
            by_country[c] = []
        by_country[c].append(info)

    del df1

    # 2. Load candidate_pairs.tsv for these 25 entities
    print("Loading candidate_pairs.tsv for the 25 entities...", flush=True)
    cand_pairs_dict = {}
    with open("output/candidate_pairs.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            sid = parts[0]
            if sid in s1_dict:
                c_list = [c.strip() for c in parts[1].split(",") if c.strip()] if len(parts) > 1 and parts[1] else []
                cand_pairs_dict[sid] = set(c_list)

    print(f"Loaded candidate lists for {len(cand_pairs_dict)} / 25 entities.")

    # 2b. Also load matching_results.tsv (the 0.774 submission) to see what was actually chosen!
    print("Loading matching_results.tsv (0.774 submission) for these 25 entities...", flush=True)
    submitted_dict = {}
    with open("output/matching_results.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            sid = parts[0]
            if sid in s1_dict:
                c_list = [c.strip() for c in parts[1].split(",") if c.strip()] if len(parts) > 1 and parts[1] else []
                submitted_dict[sid] = set(c_list)

    # 3. Exhaustive search in test_source2 and test_source3 for genuine matches
    print("\nScanning test_source2 and test_source3 (3.4M records) with country & token indexing...", flush=True)
    plausible_matches = {sid: [] for sid in selected_sids}

    t0 = time.time()
    for src_path in ["student_resource/dataset/test/test_source2.tsv", "student_resource/dataset/test/test_source3.tsv"]:
        print(f"  Scanning {src_path}...", flush=True)
        with open(src_path, "r", encoding="utf-8") as f:
            next(f)
            for line_idx, line in enumerate(f):
                parts = line.rstrip("\n").split("\t")
                cid = parts[0]
                c_name = parts[1] if len(parts) > 1 else ""
                c_addr = parts[2] if len(parts) > 2 else ""
                c_country = parts[3] if len(parts) > 3 else ""

                if c_country not in by_country:
                    continue

                c_name_lower = c_name.lower()
                c_addr_lower = c_addr.lower()

                for s1 in by_country[c_country]:
                    # Check if there is token overlap in name OR address
                    has_name_token = any(t in c_name_lower for t in s1["name_tokens"])
                    has_addr_token = any(t in c_addr_lower for t in s1["addr_tokens"])
                    
                    if not (has_name_token or has_addr_token):
                        continue

                    ns = fuzz.token_sort_ratio(s1["name"].lower(), c_name_lower)
                    asim = fuzz.token_sort_ratio(s1["addr"].lower(), c_addr_lower)

                    # Genuine match criteria:
                    # High name similarity (> 80) and reasonable address similarity (> 55)
                    # OR identical/near-identical name (> 90) and matching city/street (> 40)
                    is_match = False
                    if ns >= 80 and asim >= 55:
                        is_match = True
                    elif ns >= 90 and asim >= 40:
                        is_match = True

                    if is_match:
                        in_candidates = cid in cand_pairs_dict.get(s1["sid"], set())
                        in_submitted = cid in submitted_dict.get(s1["sid"], set())
                        plausible_matches[s1["sid"]].append({
                            "cid": cid,
                            "name": c_name,
                            "addr": c_addr,
                            "ns": ns,
                            "asim": asim,
                            "in_candidates": in_candidates,
                            "in_submitted": in_submitted
                        })

    t1 = time.time()
    print(f"Dataset scan completed in {t1 - t0:.1f} seconds.")

    # 4. Report Recall & Blocking Coverage
    print("\n" + "=" * 80)
    print("RECALL & BLOCKING COVERAGE AUDIT RESULTS")
    print("=" * 80)

    total_true_matches = 0
    total_in_candidates = 0
    total_in_submitted = 0

    log_lines = []
    log_lines.append("=" * 80)
    log_lines.append("DETAILED RECALL & BLOCKING AUDIT (25 ENTITIES)")
    log_lines.append("=" * 80)

    for idx, sid in enumerate(selected_sids, 1):
        s1 = s1_dict[sid]
        matches = plausible_matches[sid]
        n_true = len(matches)
        n_cand = sum(1 for m in matches if m["in_candidates"])
        n_sub = sum(1 for m in matches if m["in_submitted"])
        
        total_true_matches += n_true
        total_in_candidates += n_cand
        total_in_submitted += n_sub

        status = f"Cand: {n_cand}/{n_true} | Sub: {n_sub}/{n_true}"
        log_lines.append(f"[{idx:02d}] [{s1['country']}] {sid} | True: {n_true} | In Cand: {n_cand} | In Sub: {n_sub}")
        log_lines.append(f"  S1: {s1['name']} | {s1['addr']}")
        if not matches:
            log_lines.append("    (No true matches found in S2/S3 - True Singleton)")
        for m in matches:
            c_tag = "[IN CAND]" if m["in_candidates"] else "[MISSED IN CAND!]"
            s_tag = "[IN SUB]" if m["in_submitted"] else "[DROPPED IN SUB]"
            log_lines.append(f"    * {c_tag} {s_tag} (ns={m['ns']:.1f}, asim={m['asim']:.1f}) {m['cid']}: {m['name']} | {m['addr']}")
        log_lines.append("")

    cand_recall = total_in_candidates / total_true_matches if total_true_matches else 0
    sub_recall = total_in_submitted / total_true_matches if total_true_matches else 0

    print(f"Total Genuine True Matches Found in S2/S3 : {total_true_matches}")
    print(f"Present in candidate_pairs.tsv (BLOCKING)  : {total_in_candidates} / {total_true_matches} ({cand_recall*100:.2f}%)")
    print(f"Selected in matching_results.tsv (FINAL)   : {total_in_submitted} / {total_true_matches} ({sub_recall*100:.2f}%)")
    print(f"Matches Lost at Blocking Stage             : {total_true_matches - total_in_candidates}")
    print(f"Matches Lost at Ranking / Filtering Stage  : {total_in_candidates - total_in_submitted}")

    log_lines.append("=" * 80)
    log_lines.append(f"TOTAL GENUINE TRUE MATCHES: {total_true_matches}")
    log_lines.append(f"BLOCKING COVERAGE (candidate_pairs.tsv): {cand_recall*100:.2f}% ({total_in_candidates}/{total_true_matches})")
    log_lines.append(f"FINAL SUBMISSION RECALL (matching_results.tsv): {sub_recall*100:.2f}% ({total_in_submitted}/{total_true_matches})")
    log_lines.append(f"DROPPED AT BLOCKING: {total_true_matches - total_in_candidates}")
    log_lines.append(f"DROPPED AT RANKING/FILTERING: {total_in_candidates - total_in_submitted}")
    log_lines.append("=" * 80)

    with open("output/recall_blocking_audit_25.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print("\nSaved detailed report to output/recall_blocking_audit_25.txt")

if __name__ == "__main__":
    main()
