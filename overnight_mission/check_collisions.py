import collections

for country, path in [("France", "output/temp_work/dynamic_results_France.tsv"), ("US", "output/temp_work/dynamic_results_US.tsv")]:
    cand_counts = collections.Counter()
    total_matches = 0
    with open(path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip().split("\t")
            if len(p) > 2 and p[2]:
                for cid in p[2].split(","):
                    cid = cid.strip()
                    if cid:
                        cand_counts[cid] += 1
                        total_matches += 1

    multi_claimed = sum(1 for cid, c in cand_counts.items() if c > 1)
    extra_false_positives = sum(c - 1 for cid, c in cand_counts.items() if c > 1)
    print(f"--- {country} ---")
    print(f"Total Matches: {total_matches:,}")
    print(f"Unique candidates matched: {len(cand_counts):,}")
    print(f"Candidates claimed by multiple S1: {multi_claimed:,} ({multi_claimed/len(cand_counts)*100:.2f}%)")
    print(f"Guaranteed false positive merges from collisions: {extra_false_positives:,}")
