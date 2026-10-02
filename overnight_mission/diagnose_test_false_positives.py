import re
import pandas as pd

def extract_street_num(addr):
    if not addr: return ""
    m = re.search(r'\b\d+[a-zA-Z]?\b', addr)
    return m.group(0).lower() if m else ""

# Load 100 S1 records
s1_sample = {}
for chunk in pd.read_csv("student_resource/dataset/test/test_source1.tsv", sep="\t", chunksize=10000, keep_default_na=False):
    for _, r in chunk.head(100).iterrows():
        s1_sample[r["entity_id"]] = r.to_dict()
    break

# Load candidate records
needed_c = set()
s1_matches = {}
with open("BitMinds/output/matching_results.tsv", "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.rstrip().split("\t")
        sid = p[0]
        if sid in s1_sample:
            m = [c.strip() for c in p[1].split(",") if c.strip()] if len(p) > 1 and p[1] else []
            s1_matches[sid] = m
            needed_c.update(m)

c_records = {}
for p in ["student_resource/dataset/test/test_source2.tsv", "student_resource/dataset/test/test_source3.tsv"]:
    for chunk in pd.read_csv(p, sep="\t", chunksize=100000, keep_default_na=False):
        sub = chunk[chunk["entity_id"].isin(needed_c)]
        if len(sub) > 0:
            for _, r in sub.iterrows():
                c_records[r["entity_id"]] = r.to_dict()

# Analyze matches
total_m = 0
street_num_mismatches = 0
name_token_mismatches = 0

print("=" * 80)
print("AUDIT OF FIRST 20 ENTITIES IN SUBMISSION")
print("=" * 80)

for i, (sid, s1_r) in enumerate(list(s1_sample.items())[:20]):
    m_list = s1_matches.get(sid, [])
    s1_num = extract_street_num(s1_r["business_address"])
    print(f"\n[{i+1}] {sid} ({s1_r['country']}): {s1_r['business_name']} | {s1_r['business_address']}")
    print(f"    Total predicted matches: {len(m_list)}")
    for cid in m_list:
        cr = c_records.get(cid, {})
        c_num = extract_street_num(cr.get("business_address", ""))
        num_mismatch = (s1_num and c_num and s1_num != c_num)
        flag = " [STREET NUM MISMATCH!]" if num_mismatch else ""
        print(f"      -> {cid}: {cr.get('business_name')} | {cr.get('business_address')}{flag}")

