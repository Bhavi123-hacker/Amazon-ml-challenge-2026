import json

with open("audit_consolidated_sample.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("audit_clean_dump.txt", "w", encoding="utf-8") as out:
    for c in ["France", "US", "India"]:
        sub = [x for x in data if x["country"] == c]
        out.write(f"\n=======================================================\n")
        out.write(f"COUNTRY: {c} ({len(sub)} pairs)\n")
        out.write(f"=======================================================\n")
        for item in sub:
            p = item["prob"]
            n = item["pair_num"]
            s1_n = item["s1_name"]
            c_n = item["cand_name"]
            s1_a = item["s1_address"]
            c_a = item["cand_address"]
            out.write(f"[{n:02d}] P={p:.5f}\n")
            out.write(f"    S1:   {s1_n}  @  {s1_a}\n")
            out.write(f"    CAND: {c_n}  @  {c_a}\n")

print("Dumped successfully with UTF-8 encoding.")
