import os
import sys
import random
import pandas as pd
from rapidfuzz import fuzz

def main():
    print("=" * 80)
    print("FINAL DIRECT MANUAL AUDIT: 30 FRESH ENTITIES FROM THE FINAL MASTER SUBMISSION")
    print("=" * 80)

    # 1. Load test_source1 to get country and text
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

    # 2. Group by country
    fr_s1 = [sid for sid, data in s1_dict.items() if data["country"] == "France"]
    us_s1 = [sid for sid, data in s1_dict.items() if data["country"] == "US"]
    in_s1 = [sid for sid, data in s1_dict.items() if data["country"] == "India"]

    # 3. Read matching_results.tsv (the final regenerated file!)
    print("Loading final output/matching_results.tsv...", flush=True)
    matches_dict = {}
    with open("output/matching_results.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            sid = parts[0]
            m = [c.strip() for c in parts[1].split(",") if c.strip()] if len(parts) > 1 and parts[1] else []
            matches_dict[sid] = m

    # 4. Sample 10 France, 10 US, 10 India with matches using brand new seed 77777
    rng = random.Random(77777)
    fr_sample = rng.sample([sid for sid in fr_s1 if len(matches_dict.get(sid, [])) > 0], 10)
    us_sample = rng.sample([sid for sid in us_s1 if len(matches_dict.get(sid, [])) > 0], 10)
    in_sample = rng.sample([sid for sid in in_s1 if len(matches_dict.get(sid, [])) > 0], 10)

    # 5. Collect all candidate IDs needed for raw text lookup
    needed_cids = set()
    for sid in fr_sample + us_sample + in_sample:
        for cid in matches_dict.get(sid, []):
            needed_cids.add(cid)

    print(f"Total candidate IDs to look up across 30 sampled entities: {len(needed_cids)}")

    # 6. Look up raw text from test_source2 and test_source3
    cand_dict = {}
    print("Scanning test_source2...", flush=True)
    with open("student_resource/dataset/test/test_source2.tsv", "r", encoding="utf-8") as f:
        header = next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            cid = parts[0]
            if cid in needed_cids:
                cand_dict[cid] = {
                    "name": parts[1] if len(parts) > 1 else "",
                    "addr": parts[2] if len(parts) > 2 else "",
                    "country": parts[3] if len(parts) > 3 else ""
                }

    print("Scanning test_source3...", flush=True)
    with open("student_resource/dataset/test/test_source3.tsv", "r", encoding="utf-8") as f:
        header = next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            cid = parts[0]
            if cid in needed_cids:
                cand_dict[cid] = {
                    "name": parts[1] if len(parts) > 1 else "",
                    "addr": parts[2] if len(parts) > 2 else "",
                    "country": parts[3] if len(parts) > 3 else ""
                }

    print(f"Found raw text for {len(cand_dict)} / {len(needed_cids)} candidates.")

    # 7. Format raw text output for detailed audit
    out_lines = []
    out_lines.append("=" * 80)
    out_lines.append("FINAL MASTER AUDIT: 30 RANDOM ENTITIES FROM FINAL SUBMISSION (SEED 77777)")
    out_lines.append("=" * 80)

    def audit_country(name, sample_list):
        out_lines.append(f"\n{'='*20} COUNTRY: {name} {'='*20}\n")
        country_pairs = 0
        for idx, sid in enumerate(sample_list, 1):
            s1_info = s1_dict[sid]
            c_list = matches_dict.get(sid, [])
            out_lines.append(f"[{name} #{idx:02d}] {sid}")
            out_lines.append(f"  Source 1 : {s1_info['name']} | {s1_info['addr']}")
            out_lines.append(f"  Matches ({len(c_list)}):")
            for cid in c_list:
                country_pairs += 1
                c_info = cand_dict.get(cid, {"name": "MISSING", "addr": "MISSING", "country": "MISSING"})
                ns = fuzz.token_sort_ratio(s1_info['name'].lower(), c_info['name'].lower())
                asim = fuzz.token_sort_ratio(s1_info['addr'].lower(), c_info['addr'].lower())
                out_lines.append(f"    * {cid} [{c_info['country']}]: {c_info['name']} | {c_info['addr']}")
                out_lines.append(f"      [Metrics: NameSim={ns:.1f}, AddrSim={asim:.1f}]")
            out_lines.append("")
        return country_pairs

    fr_pairs = audit_country("France", fr_sample)
    us_pairs = audit_country("US", us_sample)
    in_pairs = audit_country("India", in_sample)

    total_pairs = fr_pairs + us_pairs + in_pairs
    out_lines.append("=" * 80)
    out_lines.append(f"SUMMARY OF SAMPLE: {total_pairs} total pairs across 30 entities")
    out_lines.append(f"  France : {fr_pairs} pairs across 10 entities (avg {fr_pairs/10:.1f}/entity)")
    out_lines.append(f"  US     : {us_pairs} pairs across 10 entities (avg {us_pairs/10:.1f}/entity)")
    out_lines.append(f"  India  : {in_pairs} pairs across 10 entities (avg {in_pairs/10:.1f}/entity)")
    out_lines.append("=" * 80)

    audit_text = "\n".join(out_lines)
    audit_file = "output/final_master_audit_raw_text.txt"
    with open(audit_file, "w", encoding="utf-8") as f:
        f.write(audit_text)

    print(f"Final master audit raw text written to {audit_file}")
    print(f"Total pairs to audit: {total_pairs} (France: {fr_pairs}, US: {us_pairs}, India: {in_pairs})")

if __name__ == "__main__":
    main()
