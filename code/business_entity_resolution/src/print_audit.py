import json

with open("audit_consolidated_sample.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for country in ["France", "US", "India"]:
    sub = [x for x in data if x["country"] == country]
    print(f"\n=======================================================")
    print(f"COUNTRY: {country} ({len(sub)} pairs)")
    print(f"=======================================================")
    for item in sub:
        p = item["prob"]
        num = item["pair_num"]
        s1_n = item["s1_name"]
        c_n = item["cand_name"]
        s1_a = item["s1_address"]
        c_a = item["cand_address"]
        print(f"#{num:02d} [P={p:.5f}]")
        print(f"   S1:   {s1_n} | {s1_a}")
        print(f"   CAND: {c_n} | {c_a}")
