import os
import sys
import pandas as pd

sys.path.insert(0, "code/business_entity_resolution")
from src.features import extract_pair_features
from src.normalize import apply_normalization_df

cap_pairs = [
    ('S1-172721407', 'S2-411064726', 'Pair 05 (False)'),
    ('S1-959676854', 'S3-112272421', 'Pair 10 (False)'),
    ('S1-406999539', 'S2-202434190', 'Pair 11 (Ambiguous)'),
    ('S1-506765502', 'S2-412662838', 'Pair 13 (False)'),
    ('S1-328512347', 'S2-211457940', 'Pair 19 (Ambiguous)'),
    ('S1-615522263', 'S2-870652615', 'Pair 20 (Ambiguous)'),
    ('S1-136460985', 'S2-95124110',  'Pair 22 (Ambiguous)'),
    ('S1-380388368', 'S2-154809880', 'Pair 23 (False)'),
    ('S1-819135569', 'S3-158405649', 'Pair 24 (Genuine)'),
    ('S1-634657033', 'S2-941023406', 'Pair 38 (Genuine)')
]

needed_s1 = {p[0] for p in cap_pairs}
needed_cands = {p[1] for p in cap_pairs}

s1_df = pd.read_csv('student_resource/dataset/test/test_source1.tsv', sep='\t', dtype=str)
s1_sub = s1_df[s1_df['entity_id'].isin(needed_s1)].copy()
s1_norm = apply_normalization_df(s1_sub)
s1_map = {r['entity_id']: r for r in s1_norm.to_dict('records')}

c_df = pd.read_csv('output/temp_work/cands_France.tsv', sep='\t', dtype=str)
c_sub = c_df[c_df['entity_id'].isin(needed_cands)].copy()
c_norm = apply_normalization_df(c_sub)
c_map = {r['entity_id']: r for r in c_norm.to_dict('records')}

for s1_id, cid, label in cap_pairs:
    r1 = s1_map[s1_id]
    rc = c_map[cid]
    feats = extract_pair_features(r1, rc)
    st_sim = feats['street_name_sim']
    st_jac = feats['st_name_jaccard']
    st_match = feats['street_number_match']
    b1 = r1['business_name']
    b2 = rc['business_name']
    print(f"{label} | st_name_sim: {st_sim:.3f} | st_jac: {st_jac:.3f} | st_match: {st_match:.0f} | {b1} vs {b2}")
