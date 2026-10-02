import os, sys, time
import pandas as pd
from rapidfuzz import fuzz

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))
from src.normalize import apply_normalization_df

print("Testing fast re-scoring on France sample...")
t0 = time.time()

# Load S1 France sample (5,000)
df_s1 = pd.read_csv('student_resource/dataset/test/test_source1.tsv', sep='\t', dtype=str, keep_default_na=False)
df_s1_fr = df_s1[df_s1['country'] == 'France'].head(5000).copy()
df_s1_fr = apply_normalization_df(df_s1_fr)
s1_map = {r['entity_id']: r for r in df_s1_fr.to_dict('records')}

# Load candidate records
df_cands = pd.read_csv('output/temp_work/cands_France.tsv', sep='\t', dtype=str, keep_default_na=False)
df_cands = apply_normalization_df(df_cands)
cand_map = {r['entity_id']: r for r in df_cands.to_dict('records')}

print(f"Loaded {len(s1_map)} S1 and {len(cand_map)} candidates in {time.time() - t0:.2f}s")

# Load existing candidate lists from results_France.tsv
lines = []
with open('output/temp_work/results_France.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for i, line in enumerate(f):
        parts = line.strip().split('\t')
        if parts[0] in s1_map:
            lines.append((parts[0], parts[1].split(',') if len(parts) > 1 and parts[1] else []))
        if len(lines) >= 5000: break

print(f"Scoring {len(lines)} entities...")
t_score = time.time()

total_matches = 0
entities_with_match = 0

for s1_id, cands in lines:
    r1 = s1_map[s1_id]
    n1 = r1['name_norm']
    a1 = r1['addr_norm']
    c1 = r1.get('city_extracted', '')
    p1 = r1.get('pin_extracted', '')
    st1 = r1.get('street_number', '')
    
    matched = []
    for cid in cands:
        rc = cand_map.get(cid)
        if not rc: continue
        nc = rc['name_norm']
        ac = rc['addr_norm']
        
        sn_set = fuzz.token_set_ratio(n1, nc)
        sn_sort = fuzz.token_sort_ratio(n1, nc)
        sa_set = fuzz.token_set_ratio(a1, ac)
        
        c2 = rc.get('city_extracted', '')
        city_conflict = bool(c1 and c2 and c1 != c2 and fuzz.ratio(c1, c2) < 70)
        p2 = rc.get('pin_extracted', '')
        pin_conflict = bool(p1 and p2 and p1 != p2 and p1[:3] != p2[:3])
        conflict = city_conflict or pin_conflict
        
        st2 = rc.get('street_number', '')
        st_match = bool(st1 and st2 and st1 == st2)
        
        is_match = False
        # 1. Solid dual match
        if sn_set >= 70 and sa_set >= 58 and not conflict:
            is_match = True
        # 2. Address match with street number confirmed
        elif st_match and sa_set >= 65 and not conflict:
            is_match = True
        # 3. High address match (alias name or transliterated Indian name)
        elif sa_set >= 72 and not conflict:
            is_match = True
        # 4. High token_sort name match
        elif sn_sort >= 74 and (not ac or not a1 or (sa_set >= 45 and not conflict)):
            is_match = True
        # 5. Composite score with token_sort check
        elif sn_sort >= 60 and sa_set >= 48 and (max(sn_set, sa_set) + 0.3 * min(sn_set, sa_set)) >= 104 and not conflict:
            is_match = True
            
        if is_match:
            matched.append(cid)
            
    if matched:
        entities_with_match += 1
        total_matches += len(matched)

elapsed = time.time() - t_score
rate = len(lines) / max(0.01, elapsed)
print(f"Scored {len(lines)} entities in {elapsed:.2f}s ({rate:,.0f} entities/s)")
print(f"Entities with match: {entities_with_match}/{len(lines)} ({entities_with_match/len(lines)*100:.2f}%)")
print(f"Total matched pairs: {total_matches} (avg {total_matches/len(lines):.2f}/entity)")
