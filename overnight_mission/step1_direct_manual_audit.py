"""Step 1: Direct Manual Audit of Actual Submitted File (Fast Pure Streaming)."""

import random
import os

def main():
    print("=" * 80)
    print("STEP 1: DIRECT MANUAL AUDIT OF ACTUAL SUBMITTED PREDICTIONS")
    print("=" * 80)

    # 1. Read matching_results.tsv and sample 10 entities per country
    print("Reading BitMinds/output/matching_results.tsv...", flush=True)
    matches_map = {}
    with open("BitMinds/output/matching_results.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) > 1 and p[1].strip():
                matches_map[p[0]] = [c.strip() for c in p[1].split(",") if c.strip()]

    # 2. Read test_source1.tsv and categorize by country
    print("Scanning test_source1.tsv...", flush=True)
    s1_by_country = {"France": [], "US": [], "India": []}
    s1_meta = {}
    with open("student_resource/dataset/test/test_source1.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            sid = p[0]
            if sid in matches_map:
                c = p[3] if len(p) > 3 else ""
                if c in s1_by_country:
                    s1_by_country[c].append(sid)
                    s1_meta[sid] = (p[1], p[2], c)

    # 3. Sample 10 from each country
    random.seed(42)
    sampled_s1 = {}
    needed_cands = set()
    for c in ["France", "US", "India"]:
        sampled_s1[c] = random.sample(s1_by_country[c], 10)
        for sid in sampled_s1[c]:
            needed_cands.update(matches_map[sid])

    print(f"Sampled 30 S1 entities with {len(needed_cands)} matched candidates across sources.", flush=True)

    # 4. Lookup candidate metadata
    cand_meta = {}
    for p in ["student_resource/dataset/test/test_source2.tsv", "student_resource/dataset/test/test_source3.tsv"]:
        print(f"Scanning {p}...", flush=True)
        with open(p, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                cid = parts[0]
                if cid in needed_cands:
                    cand_meta[cid] = (parts[1], parts[2], parts[3] if len(parts) > 3 else "")

    # 5. Write audit output
    out_path = "output/step1_audit_results.txt"
    with open(out_path, "w", encoding="utf-8") as out:
        out.write("=" * 80 + "\n")
        out.write("AUDIT OF 30 SUBMITTED ENTITIES (10 FRANCE, 10 US, 10 INDIA)\n")
        out.write("=" * 80 + "\n")
        for country in ["France", "US", "India"]:
            out.write(f"\n==================== COUNTRY: {country.upper()} ====================\n")
            for i, sid in enumerate(sampled_s1[country], 1):
                s1_name, s1_addr, _ = s1_meta[sid]
                m_list = matches_map[sid]
                out.write(f"\n[{country} #{i:02d}] {sid}\n")
                out.write(f"  Source 1 : {s1_name} | {s1_addr}\n")
                out.write(f"  Matches ({len(m_list)}):\n")
                for cid in m_list:
                    c_name, c_addr, c_country = cand_meta.get(cid, ("NOT_FOUND", "NOT_FOUND", "NOT_FOUND"))
                    out.write(f"    * {cid} [{c_country}]: {c_name} | {c_addr}\n")

    print(f"Audit results written to {out_path}", flush=True)

if __name__ == "__main__":
    main()
