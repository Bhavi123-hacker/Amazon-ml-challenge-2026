import os, sys
import pandas as pd

# Let's inspect 10 S1 entities and ALL their true matches from train
df_gt = pd.read_csv('student_resource/dataset/train/train_ground_truth.tsv', sep='\t', nrows=50, keep_default_na=False)

s1_needed = set()
s2_needed = set()
s3_needed = set()

for _, r in df_gt.iterrows():
    s1 = r['source1_entity_id'].strip()
    m_str = r['matched_entity_ids'].strip()
    if not m_str: continue
    s1_needed.add(s1)
    for cid in m_str.split(','):
        cid = cid.strip()
        if cid.startswith('S2-'): s2_needed.add(cid)
        elif cid.startswith('S3-'): s3_needed.add(cid)

def load_recs(tsv, ids):
    res = {}
    for chunk in pd.read_csv(tsv, sep='\t', chunksize=50000, keep_default_na=False):
        sub = chunk[chunk['entity_id'].isin(ids)]
        for _, row in sub.iterrows():
            res[row['entity_id']] = (row['business_name'], row['business_address'], row['country'])
        if len(res) == len(ids): break
    return res

s1_recs = load_recs('student_resource/dataset/train/train_source1.tsv', s1_needed)
s2_recs = load_recs('student_resource/dataset/train/train_source2.tsv', s2_needed)
s3_recs = load_recs('student_resource/dataset/train/train_source3.tsv', s3_needed)

with open('inspect_details.txt', 'w', encoding='utf-8') as out:
    for _, r in df_gt.iterrows():
        s1 = r['source1_entity_id'].strip()
        m_str = r['matched_entity_ids'].strip()
        if not m_str or s1 not in s1_recs: continue
        s1_n, s1_a, s1_c = s1_recs[s1]
        out.write(f"=== S1: {s1} ({s1_c}) ===\n")
        out.write(f"  NAME: {s1_n}\n")
        out.write(f"  ADDR: {s1_a}\n")
        for cid in m_str.split(','):
            cid = cid.strip()
            rec = s2_recs.get(cid) or s3_recs.get(cid)
            if rec:
                out.write(f"  -> {cid}: {rec[0]} | {rec[1]}\n")
            else:
                out.write(f"  -> {cid}: NOT FOUND\n")
        out.write("\n")

print("Wrote inspect_details.txt successfully.")
