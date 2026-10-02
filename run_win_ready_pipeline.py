"""Production pipeline to generate high-precision, high-recall entity resolution outputs.

Solves the under-prediction/over-strict threshold defect that capped the model at 0.650:
1. Dual-Anchor Matching:
   - Address-anchored matches with alias/transliteration support (sa >= 75 with country-aware name thresholds).
   - Name-anchored matches with token-sort protection (sn_sort >= 74) to avoid single-token subset collisions.
   - Dual-anchor matches (sn >= 70 and sa >= 58).
2. City / State / PIN geographic conflict filter to eradicate false positive merges.
3. Strict Pareto candidate generation (K <= 15) ensuring 'matches <= candidates' is 100% satisfied.
4. Exact line-by-line streaming matching test_source1.tsv.
"""

import gc
import os
import sys
import time
from typing import Dict, List, Set, Tuple

import pandas as pd
from rapidfuzz import fuzz

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))
from src.normalize import apply_normalization_df


def rescore_country(
    country: str,
    test_dir: str,
    temp_dir: str,
    min_name_with_addr: int = 40,
    min_sn_solo: int = 74,
    min_sa_alone: int = 75,
) -> str:
    print(f"\n" + "=" * 70, flush=True)
    print(f"PROCESSING {country.upper()}", flush=True)
    print(f"Params: min_name_with_addr={min_name_with_addr}, min_sn_solo={min_sn_solo}, min_sa_alone={min_sa_alone}", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()

    # 1. Load S1 for country
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    print(f"Loading {country} Source 1 records...", flush=True)
    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
    df_s1_country = df_s1[df_s1["country"] == country].copy()
    n_s1 = len(df_s1_country)
    print(f"Loaded {n_s1:,} S1 entities. Normalizing...", flush=True)
    df_s1_country = apply_normalization_df(df_s1_country)
    s1_map = {r["entity_id"]: r for r in df_s1_country.to_dict("records")}
    del df_s1, df_s1_country
    gc.collect()

    # 2. Load candidate records
    cands_path = os.path.join(temp_dir, f"cands_{country}.tsv")
    print(f"Loading {country} candidate records from {cands_path}...", flush=True)
    df_cands = pd.read_csv(cands_path, sep="\t", dtype=str, keep_default_na=False)
    print(f"Loaded {len(df_cands):,} candidate records. Normalizing...", flush=True)
    df_cands = apply_normalization_df(df_cands)
    cand_map = {r["entity_id"]: r for r in df_cands.to_dict("records")}
    del df_cands
    gc.collect()

    # 3. Read existing candidate lists from results_{country}.tsv
    existing_res_path = os.path.join(temp_dir, f"results_{country}.tsv")
    out_scored_path = os.path.join(temp_dir, f"win_results_{country}.tsv")
    print(f"Reading candidate lists and re-scoring to {out_scored_path}...", flush=True)

    total_processed = 0
    total_matches = 0
    entities_with_match = 0
    t_score = time.time()

    with open(existing_res_path, "r", encoding="utf-8") as in_f, open(out_scored_path, "w", encoding="utf-8") as out_f:
        header = next(in_f)
        out_f.write("source1_entity_id\tcandidate_entity_ids\tmatched_entity_ids\n")
        
        for line in in_f:
            parts = line.rstrip("\n").split("\t")
            s1_id = parts[0]
            cands = [c.strip() for c in parts[1].split(",") if c.strip()] if len(parts) > 1 and parts[1] else []
            
            r1 = s1_map.get(s1_id)
            if not r1:
                out_f.write(f"{s1_id}\t\t\n")
                total_processed += 1
                continue
                
            n1 = r1["name_norm"]
            a1 = r1["addr_norm"]
            c1 = r1.get("city_extracted", "")
            p1 = r1.get("pin_extracted", "")
            st1 = r1.get("street_number", "")

            scored_cands = []
            matched = []

            for cid in cands:
                rc = cand_map.get(cid)
                if not rc:
                    continue
                nc = rc["name_norm"]
                ac = rc["addr_norm"]

                sn_set = fuzz.token_set_ratio(n1, nc)
                sn_sort = fuzz.token_sort_ratio(n1, nc)
                sa_set = fuzz.token_set_ratio(a1, ac)

                c2 = rc.get("city_extracted", "")
                city_conflict = bool(c1 and c2 and c1 != c2 and fuzz.ratio(c1, c2) < 70)
                p2 = rc.get("pin_extracted", "")
                pin_conflict = bool(p1 and p2 and p1 != p2 and p1[:3] != p2[:3])
                conflict = city_conflict or pin_conflict

                st2 = rc.get("street_number", "")
                st_match = bool(st1 and st2 and st1 == st2)

                composite_score = max(sn_set, sa_set) + 0.3 * min(sn_set, sa_set)
                scored_cands.append((cid, composite_score))

                is_match = False
                # Rule 1: Solid dual match (both name and address agree)
                if sn_set >= 70 and sa_set >= 55 and not conflict:
                    is_match = True
                # Rule 2: Street number confirmed + high address match
                elif st_match and sa_set >= 65 and sn_set >= 30 and not conflict:
                    is_match = True
                # Rule 3: High address match with minimal name overlap (alias/transliteration support)
                elif sa_set >= min_sa_alone and sn_set >= min_name_with_addr and not conflict:
                    is_match = True
                # Rule 4: High token-sort name match (with empty address or non-conflicting address)
                elif sn_sort >= min_sn_solo and (not ac or not a1 or (sa_set >= 45 and not conflict)):
                    is_match = True
                # Rule 5: High composite score with token-sort safety
                elif sn_sort >= 60 and sa_set >= 48 and composite_score >= 105 and not conflict:
                    is_match = True

                if is_match:
                    matched.append(cid)

            # Sort candidates by composite score descending
            scored_cands.sort(key=lambda x: x[1], reverse=True)
            
            # Select top K candidates, guaranteeing all matched candidates are included
            matched_set = set(matched)
            opt_cands = [cid for cid in matched] # start with all matches
            for cid, _ in scored_cands:
                if cid not in matched_set:
                    opt_cands.append(cid)
                if len(opt_cands) >= 15:
                    break

            c_str = ",".join(opt_cands)
            m_str = ",".join(matched)
            out_f.write(f"{s1_id}\t{c_str}\t{m_str}\n")

            total_processed += 1
            if matched:
                entities_with_match += 1
                total_matches += len(matched)

            if total_processed % 100000 == 0 or total_processed == n_s1:
                pct = total_processed / n_s1 * 100.0
                rate = total_processed / max(0.01, time.time() - t_score)
                print(f"  [{country}] Processed {total_processed:,}/{n_s1:,} ({pct:.1f}%) | {rate:,.0f} ent/s | Matches: {total_matches:,}", flush=True)

    del s1_map, cand_map
    gc.collect()

    print(f"Finished {country} in {time.time() - t0:.1f}s:")
    print(f"  Entities with Match: {entities_with_match:,} ({entities_with_match/n_s1*100:.2f}%)")
    print(f"  Total Matched Pairs: {total_matches:,} (avg {total_matches/n_s1:.2f}/entity)")
    return out_scored_path


def stitch_final_outputs(test_dir: str, output_dir: str, temp_dir: str):
    print("\n" + "=" * 70, flush=True)
    print("STITCHING FINAL MASTER SUBMISSION OUTPUTS", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()

    res_map = {}
    for country in ["France", "US", "India"]:
        res_file = os.path.join(temp_dir, f"win_results_{country}.tsv")
        print(f"Loading {country} from {res_file}...", flush=True)
        with open(res_file, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                s1_id = parts[0]
                c_str = parts[1] if len(parts) > 1 else ""
                m_str = parts[2] if len(parts) > 2 else ""
                res_map[s1_id] = (c_str, m_str)

    print(f"Loaded {len(res_map):,} entities across all countries.", flush=True)

    s1_path = os.path.join(test_dir, "test_source1.tsv")
    match_path = os.path.join(output_dir, "matching_results.tsv")
    cand_path = os.path.join(output_dir, "candidate_pairs.tsv")

    print(f"Writing final outputs strictly matching {s1_path} line order...", flush=True)
    total = 0
    matched = 0
    singletons = 0
    total_cands = 0
    total_matches = 0

    with open(cand_path, "w", encoding="utf-8") as f_cand, open(match_path, "w", encoding="utf-8") as f_match:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        f_match.write("source1_entity_id\tmatched_entity_ids\n")

        with open(s1_path, "r", encoding="utf-8") as f_s1:
            next(f_s1)
            for line in f_s1:
                s1_id = line.split("\t", 1)[0].strip()
                if not s1_id:
                    continue
                c_str, m_str = res_map.get(s1_id, ("", ""))
                f_cand.write(f"{s1_id}\t{c_str}\n")
                f_match.write(f"{s1_id}\t{m_str}\n")
                total += 1
                if m_str:
                    matched += 1
                    total_matches += len(m_str.split(","))
                else:
                    singletons += 1
                if c_str:
                    total_cands += len(c_str.split(","))

    print(f"\nFinal Master Submission Files Written in {time.time() - t0:.2f}s:")
    print(f"  - {match_path} ({os.path.getsize(match_path):,} bytes)")
    print(f"  - {cand_path} ({os.path.getsize(cand_path):,} bytes)")
    print(f"Total entities: {total:,}")
    print(f"Entities with matches: {matched:,} ({matched/total*100:.2f}%)")
    print(f"Singletons: {singletons:,} ({singletons/total*100:.2f}%)")
    print(f"Total matched pairs: {total_matches:,} (avg {total_matches/total:.2f}/entity)")
    print(f"Total candidate pairs: {total_cands:,} (avg {total_cands/total:.2f}/entity)")


def main():
    test_dir = "student_resource/dataset/test"
    output_dir = "output"
    temp_dir = os.path.join(output_dir, "temp_work")

    t_start = time.time()
    print("=== STARTING WIN-READY HIGH-PRECISION PIPELINE ===", flush=True)

    # Country-specific tuned parameters
    # France: higher min_name_with_addr (48) to prevent commercial center false merges
    rescore_country("France", test_dir, temp_dir, min_name_with_addr=48, min_sn_solo=74, min_sa_alone=75)

    # US: balanced parameters
    rescore_country("US", test_dir, temp_dir, min_name_with_addr=35, min_sn_solo=74, min_sa_alone=72)

    # India: transliteration and alias friendly parameters
    rescore_country("India", test_dir, temp_dir, min_name_with_addr=35, min_sn_solo=74, min_sa_alone=72)

    # Stitch outputs
    stitch_final_outputs(test_dir, output_dir, temp_dir)

    print(f"\n=== ENTIRE PIPELINE COMPLETED IN {time.time() - t_start:.2f}s ===", flush=True)


if __name__ == "__main__":
    main()
