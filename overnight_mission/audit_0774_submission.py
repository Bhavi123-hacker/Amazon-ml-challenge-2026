import os
import sys
import random
import pandas as pd
from rapidfuzz import fuzz

def main():
    print("=" * 80)
    print("TASK 1: FRESH RANDOM AUDIT OF 0.774 SUBMISSION (60-80 PAIRS)")
    print("=" * 80)

    # 1. Load S1 data
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

    # 2. Read literal matching_results.tsv (the 0.774 file)
    print("Loading output/matching_results.tsv...", flush=True)
    matches_dict = {}
    with open("output/matching_results.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            sid = parts[0]
            m = [c.strip() for c in parts[1].split(",") if c.strip()] if len(parts) > 1 and parts[1] else []
            if m:
                matches_dict[sid] = m

    # 3. Stratified random sample of 25 pairs per country (75 total pairs) with seed 44444
    rng = random.Random(44444)
    sampled_pairs = []
    needed_cids = set()

    for c in ["France", "US", "India"]:
        country_sids = [sid for sid, data in s1_dict.items() if data["country"] == c and sid in matches_dict]
        rng.shuffle(country_sids)
        c_pairs = []
        for sid in country_sids:
            for cid in matches_dict[sid]:
                c_pairs.append((c, sid, cid))
                needed_cids.add(cid)
                if len(c_pairs) >= 25:
                    break
            if len(c_pairs) >= 25:
                break
        sampled_pairs.extend(c_pairs)
        print(f"  {c}: sampled {len(c_pairs)} pairs across {len(set(p[1] for p in c_pairs))} entities.")

    print(f"\nTotal sampled pairs: {len(sampled_pairs)} (Target: 60-80 pairs)")

    # 4. Look up candidates from S2 and S3
    cand_dict = {}
    print(f"Scanning test_source2 and 3 for {len(needed_cids)} candidates...", flush=True)
    for src in ["student_resource/dataset/test/test_source2.tsv", "student_resource/dataset/test/test_source3.tsv"]:
        with open(src, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                cid = parts[0]
                if cid in needed_cids:
                    cand_dict[cid] = {
                        "name": parts[1] if len(parts) > 1 else "",
                        "addr": parts[2] if len(parts) > 2 else "",
                        "country": parts[3] if len(parts) > 3 else ""
                    }

    # 5. Format raw text output for audit
    out_lines = []
    out_lines.append("=" * 80)
    out_lines.append("AUDIT OF 75 RANDOM PAIRS FROM 0.774 LEADERBOARD SUBMISSION")
    out_lines.append("=" * 80)

    for idx, (country, sid, cid) in enumerate(sampled_pairs, 1):
        s1 = s1_dict[sid]
        c = cand_dict.get(cid, {"name": "MISSING", "addr": "MISSING", "country": "MISSING"})
        ns = fuzz.token_sort_ratio(s1["name"].lower(), c["name"].lower())
        asim = fuzz.token_sort_ratio(s1["addr"].lower(), c["addr"].lower())
        out_lines.append(f"[{idx:02d}] [{country}] S1: {sid} vs C: {cid}")
        out_lines.append(f"  Source 1 : {s1['name']} | {s1['addr']}")
        out_lines.append(f"  Candidate: {c['name']} | {c['addr']}")
        out_lines.append(f"  Metrics  : NameSim={ns:.1f}, AddrSim={asim:.1f}")
        out_lines.append("")

    out_text = "\n".join(out_lines)
    audit_file = "output/audit_0774_sample_raw_text.txt"
    with open(audit_file, "w", encoding="utf-8") as f:
        f.write(out_text)

    print(f"Saved raw text of 75 pairs to {audit_file}")

if __name__ == "__main__":
    main()
