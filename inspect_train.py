import pandas as pd

df_gt = pd.read_csv('student_resource/dataset/train/train_ground_truth.tsv', sep='\t', nrows=20, keep_default_na=False)
s1_ids = set()
s2_ids = set()
s3_ids = set()

for _, row in df_gt.iterrows():
    s1_ids.add(row['source1_entity_id'])
    for cid in row['matched_entity_ids'].split(','):
        cid = cid.strip()
        if cid.startswith('S2-'): s2_ids.add(cid)
        elif cid.startswith('S3-'): s3_ids.add(cid)

print(f"Sample S1 IDs: {len(s1_ids)}, S2 IDs: {len(s2_ids)}, S3 IDs: {len(s3_ids)}")

# Find their records by scanning
def find_records(tsv_path, target_ids):
    records = {}
    for chunk in pd.read_csv(tsv_path, sep='\t', chunksize=50000, keep_default_na=False):
        sub = chunk[chunk['entity_id'].isin(target_ids)]
        for _, r in sub.iterrows():
            records[r['entity_id']] = r.to_dict()
        if len(records) == len(target_ids):
            break
    return records

s1_recs = find_records('student_resource/dataset/train/train_source1.tsv', s1_ids)
s2_recs = find_records('student_resource/dataset/train/train_source2.tsv', s2_ids)
s3_recs = find_records('student_resource/dataset/train/train_source3.tsv', s3_ids)

for _, row in df_gt.iterrows():
    s1 = row['source1_entity_id']
    m = row['matched_entity_ids']
    if not m:
        print(f"=== S1: {s1} (SINGLETON) ===")
        if s1 in s1_recs:
            print(f"  Name: {s1_recs[s1]['business_name']} | Addr: {s1_recs[s1]['business_address']}")
        continue
    print(f"=== S1: {s1} ({s1_recs.get(s1, {}).get('country', '')}) ===")
    print(f"  S1: {s1_recs.get(s1, {}).get('business_name')} | {s1_recs.get(s1, {}).get('business_address')}")
    for cid in m.split(','):
        cid = cid.strip()
        rec = s2_recs.get(cid) or s3_recs.get(cid)
        if rec:
            print(f"  -> {cid}: {rec['business_name']} | {rec['business_address']}")
        else:
            print(f"  -> {cid}: [Not found]")
    print()
