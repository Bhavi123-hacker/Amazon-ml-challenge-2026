import os
import sys

def main():
    print("=" * 80)
    print("SIMULATING 2-SET DISJOINT BIPARTITE RESOLUTION ON TEST SUBMISSION")
    print("=" * 80)

    # Pass 1: For each candidate, find the S1 entity where it has the earliest rank
    # In matched_entity_ids, candidate at index 0 is rank 0 (highest prob), index 1 is rank 1, etc.
    cand_best_claim = {} # cid -> (sid, rank_idx, line_idx)
    s1_rows = []

    with open("output/matching_results.tsv", "r", encoding="utf-8") as f:
        header = next(f)
        for line_idx, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            sid = parts[0]
            matches = [c.strip() for c in parts[1].split(",") if c.strip()] if len(parts) > 1 and parts[1] else []
            s1_rows.append((sid, matches))
            for r_idx, cid in enumerate(matches):
                if cid not in cand_best_claim:
                    cand_best_claim[cid] = (sid, r_idx, line_idx)
                else:
                    prev_sid, prev_rank, prev_line = cand_best_claim[cid]
                    # Lower rank index means higher probability match
                    if r_idx < prev_rank:
                        cand_best_claim[cid] = (sid, r_idx, line_idx)

    # Pass 2: Reconstruct matches where each candidate is claimed ONLY by its best S1 entity
    total_entities = len(s1_rows)
    total_matches_orig = sum(len(m) for _, m in s1_rows)
    
    resolved_matches = []
    total_matches_disjoint = 0
    singletons = 0
    match_hist = {}

    for line_idx, (sid, matches) in enumerate(s1_rows):
        kept_matches = []
        for r_idx, cid in enumerate(matches):
            best_sid, best_rank, best_line = cand_best_claim[cid]
            if best_sid == sid:
                kept_matches.append(cid)
        resolved_matches.append((sid, kept_matches))
        k = len(kept_matches)
        total_matches_disjoint += k
        if k == 0:
            singletons += 1
        match_hist[k] = match_hist.get(k, 0) + 1

    print(f"Total Entities          : {total_entities:,}")
    print(f"Original Matches        : {total_matches_orig:,} (avg {total_matches_orig/total_entities:.2f}/entity)")
    print(f"Disjoint Matches (2-Set): {total_matches_disjoint:,} (avg {total_matches_disjoint/total_entities:.2f}/entity)")
    print(f"False Positives Purged  : {total_matches_orig - total_matches_disjoint:,}")
    print(f"Singletons              : {singletons:,} ({singletons/total_entities*100:.2f}%)")
    print("\n--- 2-Set Disjoint Match Distribution ---")
    for k in sorted(match_hist.keys()):
        cnt = match_hist[k]
        print(f"  {k:2d} matches: {cnt:9,d} entities ({cnt/total_entities*100:5.2f}%)")

if __name__ == "__main__":
    main()
