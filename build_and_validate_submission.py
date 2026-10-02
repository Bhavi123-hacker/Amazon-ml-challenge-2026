"""Stitch verified country outputs into final submission files and validate.

Rules:
- Exact line order matching test_source1.tsv (1,732,544 rows + 1 header).
- France: tau = 0.98 + street_name_sim >= 0.70 filter.
- India: tau = 0.98.
- US: tau = 0.75.
- 100% containment: matches subset of candidates.
- Caps: matches <= 10, candidates <= 12.
"""

import os
import sys
import time
import pandas as pd
from rapidfuzz import fuzz

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))
from src.features import extract_street_tokens
from src.normalize import apply_normalization_df


def main():
    t0 = time.time()
    print("=" * 80, flush=True)
    print("CHECKPOINT 5: FULL SUBMISSION ASSEMBLY & VALIDATION", flush=True)
    print("=" * 80, flush=True)

    test_dir = "student_resource/dataset/test"
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    temp_dir = "output/temp_work"

    # 1. Prepare France filtered results
    print("Preparing France results (tau=0.98 + street_name_sim >= 0.70)...", flush=True)
    fr_res_path = os.path.join(temp_dir, "lgbm_clean_results_France_tau_98.tsv")

    # Load S1 France records for street tokens
    print("  Loading France S1 records...", flush=True)
    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
    df_s1_fr = df_s1[df_s1["country"] == "France"].copy()
    df_s1_fr = apply_normalization_df(df_s1_fr)
    fr_s1_st = {}
    for r in df_s1_fr.to_dict("records"):
        toks = extract_street_tokens(r.get("addr_norm", ""), r.get("city_extracted", ""))
        fr_s1_st[r["entity_id"]] = " ".join(sorted(toks))
    del df_s1, df_s1_fr

    # Load France candidates for street tokens
    print("  Loading France candidate records...", flush=True)
    df_cands_fr = pd.read_csv(os.path.join(temp_dir, "cands_France.tsv"), sep="\t", dtype=str, keep_default_na=False)
    df_cands_fr = apply_normalization_df(df_cands_fr)
    fr_c_st = {}
    for r in df_cands_fr.to_dict("records"):
        toks = extract_street_tokens(r.get("addr_norm", ""), r.get("city_extracted", ""))
        fr_c_st[r["entity_id"]] = " ".join(sorted(toks))
    del df_cands_fr

    france_data = {}  # sid -> (cands_str, matches_str)
    with open(fr_res_path, "r", encoding="utf-8") as f:
        header = next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            sid = parts[0]
            cands = [c.strip() for c in parts[1].split(",") if c.strip()] if len(parts) > 1 and parts[1] else []
            matches = [m.strip() for m in parts[2].split(",") if m.strip()] if len(parts) > 2 and parts[2] else []

            s1_st = fr_s1_st.get(sid, "")
            # Filter matches by street_name_sim >= 0.70
            accepted_matches = []
            for m in matches:
                c_st = fr_c_st.get(m, "")
                sim = fuzz.token_sort_ratio(s1_st, c_st) / 100.0 if s1_st and c_st else 0.0
                if sim >= 0.70:
                    accepted_matches.append(m)

            # Re-assemble candidates: all accepted matches first, then remaining top candidates up to 12
            matched_set = set(accepted_matches[:10])
            final_matches = accepted_matches[:10]

            final_cands = list(final_matches)
            for c in cands:
                if c not in matched_set:
                    final_cands.append(c)
                if len(final_cands) >= 12:
                    break

            france_data[sid] = (",".join(final_cands), ",".join(final_matches))

    del fr_s1_st, fr_c_st
    print(f"France prepared: {len(france_data):,} entities.", flush=True)

    # 2. Load India results (tau=0.98)
    print("Loading India results (tau=0.98)...", flush=True)
    in_res_path = os.path.join(temp_dir, "lgbm_clean_results_India_tau_98.tsv")
    india_data = {}
    with open(in_res_path, "r", encoding="utf-8") as f:
        header = next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            sid = parts[0]
            cands_str = parts[1] if len(parts) > 1 else ""
            matches_str = parts[2] if len(parts) > 2 else ""
            india_data[sid] = (cands_str, matches_str)
    print(f"India loaded: {len(india_data):,} entities.", flush=True)

    # 3. Load US results (tau=0.75)
    print("Loading US results (tau=0.75)...", flush=True)
    us_res_path = os.path.join(temp_dir, "lgbm_clean_results_US.tsv")
    us_data = {}
    with open(us_res_path, "r", encoding="utf-8") as f:
        header = next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            sid = parts[0]
            cands_str = parts[1] if len(parts) > 1 else ""
            matches_str = parts[2] if len(parts) > 2 else ""
            us_data[sid] = (cands_str, matches_str)
    print(f"US loaded: {len(us_data):,} entities.", flush=True)

    # 4. Stream test_source1.tsv and write final files
    print("\nStitching final output files in exact test_source1.tsv line order...", flush=True)
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)
    match_file = os.path.join(out_dir, "matching_results.tsv")
    cand_file = os.path.join(out_dir, "candidate_pairs.tsv")

    f_match = open(match_file, "w", encoding="utf-8")
    f_cand = open(cand_file, "w", encoding="utf-8")

    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

    total_entities = 0
    matched_entities = 0
    singletons = 0
    total_matches = 0
    dist = {k: 0 for k in range(11)}

    with open(s1_path, "r", encoding="utf-8") as f_s1:
        header = next(f_s1)
        for line in f_s1:
            total_entities += 1
            sid = line.split("\t")[0]

            if sid in india_data:
                cands_str, matches_str = india_data[sid]
            elif sid in us_data:
                cands_str, matches_str = us_data[sid]
            elif sid in france_data:
                cands_str, matches_str = france_data[sid]
            else:
                cands_str, matches_str = "", ""

            # Verify containment and caps
            cands = [c for c in cands_str.split(",") if c]
            matches = [m for m in matches_str.split(",") if m]

            assert len(matches) <= 10, f"Match cap exceeded for {sid}: {len(matches)}"
            assert len(cands) <= 12, f"Candidate cap exceeded for {sid}: {len(cands)}"
            assert set(matches).issubset(set(cands)), f"Containment violated for {sid}!"

            f_match.write(f"{sid}\t{matches_str}\n")
            f_cand.write(f"{sid}\t{cands_str}\n")

            n_m = len(matches)
            dist[n_m] += 1
            if n_m > 0:
                matched_entities += 1
                total_matches += n_m
            else:
                singletons += 1

            if total_entities % 250000 == 0:
                print(f"  Written {total_entities:,}/1,732,544 rows...", flush=True)

    f_match.close()
    f_cand.close()

    print(f"\nAssembly complete in {time.time() - t0:.1f}s.", flush=True)

    # 5. Output summary metrics
    match_rate = matched_entities / total_entities * 100.0
    singleton_rate = singletons / total_entities * 100.0
    avg_per_matched = total_matches / matched_entities if matched_entities > 0 else 0.0

    print("\n" + "=" * 80, flush=True)
    print("FINAL SUBMISSION SUMMARY (ALL COUNTRIES COMBINED)", flush=True)
    print("=" * 80, flush=True)
    print(f"Total entities written   : {total_entities:,} (Target: 1,732,544)")
    print(f"Matched entities         : {matched_entities:,} ({match_rate:.2f}%)")
    print(f"Singleton entities       : {singletons:,} ({singleton_rate:.2f}%)")
    print(f"Total matches predicted  : {total_matches:,}")
    print(f"Avg matches per matched  : {avg_per_matched:.2f}")

    print("\n" + "=" * 80, flush=True)
    print("COMBINED MATCH COUNT DISTRIBUTION (BINS 0 THROUGH 10)", flush=True)
    print("=" * 80, flush=True)
    dist_rows = []
    for k in range(11):
        cnt = dist[k]
        pct = cnt / total_entities * 100.0
        dist_rows.append({
            "matches_k": k,
            "count": f"{cnt:>9,d}",
            "pct": f"{pct:>6.2f}%"
        })
    df_dist = pd.DataFrame(dist_rows)
    print(df_dist.to_string(index=False), flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
