import collections

cand_counts = collections.Counter()
total_m = 0
singletons = 0
with open('output/matching_results.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip().split('\t')
        if len(p) > 1 and p[1]:
            cands = [c.strip() for c in p[1].split(',') if c.strip()]
            total_m += len(cands)
            for c in cands:
                cand_counts[c] += 1
        else:
            singletons += 1

multi = sum(1 for c, cnt in cand_counts.items() if cnt > 1)
extra_fp = sum(cnt - 1 for c, cnt in cand_counts.items() if cnt > 1)
print(f"Total entities: 1,732,544")
print(f"Total matches: {total_m:,}")
print(f"Unique candidates matched: {len(cand_counts):,}")
print(f"Multi-claimed candidates: {multi:,} ({multi/len(cand_counts)*100:.2f}%)")
print(f"Total extra false positive merges from collisions: {extra_fp:,}")
print(f"Average matches per entity: {total_m / 1732544:.2f}")
print(f"Singletons: {singletons:,} ({singletons / 1732544 * 100:.2f}%)")
