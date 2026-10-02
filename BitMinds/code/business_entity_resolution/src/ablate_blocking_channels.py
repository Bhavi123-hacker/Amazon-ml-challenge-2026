"""Task 2: Blocking Recall and Efficiency Ablation by Channel.

Measures the marginal value of EACH blocking channel independently:
1. Channel 1: Sorted First-2-Tokens (idx_tok2)
2. Channel 2: Phonetic Key Double Metaphone (idx_phone)
3. Channel 3: Postal PIN + Token Prefix (idx_postal)
4. Channel 4: Street Number + Street / PIN / City (idx_addr)
5. Channel 5: Significant Tokens & Prefix 4-grams (idx_sigtoken)
6. Channel 6: Acronyms (idx_acronym)

Evaluates on 3,000 ground truth entities from train dataset (with ~10,000+ true matches):
- Total true matches retrieved by channel
- Unique-catch count (true matches found ONLY by that specific channel and no other)
- Total candidate volume contributed by channel
- % of candidate volume
- Value density: (Unique Catches / Candidates Contributed) * 1,000
- Ablation Impact: Ground-truth recall drop if this channel were removed from the union

Outputs results table to models/blocking_ablation_results.json.
"""

import json
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))

import numpy as np
import pandas as pd
from metaphone import dm

from src.blocking import (
    ADDRESS_STOPS,
    GENERIC_STOPS,
    get_acronym,
    get_first_two_tokens,
    get_phonetic_key,
)
from src.normalize import apply_normalization_df


class ChannelDecomposedIndex:
    """Multi-channel index tracking channel provenance for every retrieved candidate."""

    def __init__(self):
        self.idx_tok2 = defaultdict(lambda: defaultdict(set))
        self.idx_phone = defaultdict(lambda: defaultdict(set))
        self.idx_postal = defaultdict(lambda: defaultdict(set))
        self.idx_addr = defaultdict(lambda: defaultdict(set))
        self.idx_sigtoken = defaultdict(lambda: defaultdict(set))
        self.idx_acronym = defaultdict(lambda: defaultdict(set))
        self.cand_records = {}

    def build(self, df_cands: pd.DataFrame):
        for rec in df_cands.to_dict("records"):
            cid = rec["entity_id"]
            country = rec["country"]
            n_norm = rec["name_norm"]
            pin = rec["pin_extracted"]
            st_num = rec["street_number"]
            addr_norm = rec["addr_norm"]
            city = rec["city_extracted"]
            self.cand_records[cid] = rec

            # 1. Sorted first-2-token
            k_tok2 = get_first_two_tokens(n_norm)
            if k_tok2:
                self.idx_tok2[country][k_tok2].add(cid)

            # 2. Phonetic
            k_phone = get_phonetic_key(n_norm)
            if k_phone:
                self.idx_phone[country][k_phone].add(cid)

            # 3. Postal + token prefix
            if pin:
                first_tok = n_norm.split()[0][:3] if n_norm else ""
                self.idx_postal[country][f"{pin}_{first_tok}"].add(cid)

            # 4. Address Street Number + Street tokens / PIN / City
            if st_num:
                addr_tokens = [
                    t for t in addr_norm.split()
                    if t != st_num and len(t) > 2 and t not in ADDRESS_STOPS
                ]
                for t in addr_tokens[:3]:
                    self.idx_addr[country][f"{st_num}_{t}"].add(cid)
                if pin:
                    self.idx_addr[country][f"{pin}_{st_num}"].add(cid)
                if city:
                    self.idx_addr[country][f"{st_num}_{city}"].add(cid)

            # 5. Significant tokens and prefix 4-grams
            toks = [t for t in n_norm.split() if t not in GENERIC_STOPS]
            for tok in toks:
                if len(tok) >= 4:
                    self.idx_sigtoken[country][tok].add(cid)
                    self.idx_sigtoken[country][tok[:4]].add(cid)

            # 6. Acronyms
            if len(toks) >= 2:
                acr = get_acronym(toks)
                if acr:
                    self.idx_acronym[country][acr].add(cid)
            elif len(toks) == 1 and 2 <= len(toks[0]) <= 4:
                self.idx_acronym[country][toks[0]].add(cid)

    def retrieve_by_channel(self, s1_rec: dict) -> Dict[str, Set[str]]:
        """Retrieve candidate sets partitioned by channel."""
        country = s1_rec["country"]
        n_norm = s1_rec["name_norm"]
        pin = s1_rec["pin_extracted"]
        st_num = s1_rec["street_number"]
        addr_norm = s1_rec["addr_norm"]
        city = s1_rec["city_extracted"]

        ch_cands = {
            "Sorted First-2-Tokens": set(),
            "Phonetic Key (Double Metaphone)": set(),
            "Postal PIN + Prefix": set(),
            "Address Street/PIN/City Keys": set(),
            "Significant Tokens & 4-grams": set(),
            "Acronyms": set(),
        }

        # 1. Sorted first-2-token
        k_tok2 = get_first_two_tokens(n_norm)
        if k_tok2:
            ch_cands["Sorted First-2-Tokens"].update(self.idx_tok2[country].get(k_tok2, set()))

        # 2. Phonetic
        k_phone = get_phonetic_key(n_norm)
        if k_phone:
            ch_cands["Phonetic Key (Double Metaphone)"].update(self.idx_phone[country].get(k_phone, set()))

        # 3. Postal + token prefix
        if pin:
            first_tok = n_norm.split()[0][:3] if n_norm else ""
            ch_cands["Postal PIN + Prefix"].update(self.idx_postal[country].get(f"{pin}_{first_tok}", set()))

        # 4. Address Street Number + Street tokens / PIN / City
        if st_num:
            addr_tokens = [
                t for t in addr_norm.split()
                if t != st_num and len(t) > 2 and t not in ADDRESS_STOPS
            ]
            for t in addr_tokens[:3]:
                ch_cands["Address Street/PIN/City Keys"].update(self.idx_addr[country].get(f"{st_num}_{t}", set()))
            if pin:
                ch_cands["Address Street/PIN/City Keys"].update(self.idx_addr[country].get(f"{pin}_{st_num}", set()))
            if city:
                ch_cands["Address Street/PIN/City Keys"].update(self.idx_addr[country].get(f"{st_num}_{city}", set()))

        # 5. Significant tokens and prefix 4-grams
        toks = [t for t in n_norm.split() if t not in GENERIC_STOPS]
        for tok in toks:
            if len(tok) >= 4:
                ch_cands["Significant Tokens & 4-grams"].update(self.idx_sigtoken[country].get(tok, set()))
                p_list = self.idx_sigtoken[country].get(tok[:4], set())
                if len(p_list) <= 500:
                    ch_cands["Significant Tokens & 4-grams"].update(p_list)

        # 6. Acronyms
        if len(toks) >= 2:
            acr = get_acronym(toks)
            if acr:
                ch_cands["Acronyms"].update(self.idx_acronym[country].get(acr, set()))
        elif len(toks) == 1 and 2 <= len(toks[0]) <= 4:
            ch_cands["Acronyms"].update(self.idx_acronym[country].get(toks[0], set()))

        return ch_cands


def main():
    print("=== Task 2: Blocking Recall & Efficiency Ablation by Channel ===", flush=True)
    t0 = time.time()

    data_dir = "student_resource/dataset/train"
    gt_path = os.path.join(data_dir, "train_ground_truth.tsv")

    # 1. Load Ground Truth (3,000 S1 entities)
    print("Loading training ground truth...", flush=True)
    df_gt = pd.read_csv(gt_path, sep="\t", nrows=3000, keep_default_na=False)
    gt_map = {}
    needed_s2 = set()
    needed_s3 = set()
    for _, row in df_gt.iterrows():
        sid = row["source1_entity_id"].strip()
        m_str = row["matched_entity_ids"].strip()
        m_set = {x.strip() for x in m_str.split(",") if x.strip()} if m_str else set()
        gt_map[sid] = m_set
        for m in m_set:
            if m.startswith("S2-"):
                needed_s2.add(m)
            elif m.startswith("S3-"):
                needed_s3.add(m)

    target_s1_ids = set(gt_map.keys())
    total_true_matches = sum(len(v) for v in gt_map.values())
    print(f"Sampled {len(gt_map):,} S1 entities containing {total_true_matches:,} true matches.", flush=True)

    # 2. Load S1 Records
    s1_path = os.path.join(data_dir, "train_source1.tsv")
    df_s1 = pd.read_csv(s1_path, sep="\t", keep_default_na=False)
    df_s1 = df_s1[df_s1["entity_id"].isin(target_s1_ids)].copy()
    df_s1 = apply_normalization_df(df_s1)
    s1_records = df_s1.to_dict("records")
    print(f"Loaded {len(s1_records):,} Source 1 records.", flush=True)

    # 3. Load Candidates (Source 2 and Source 3)
    print("Loading candidate records for true matches + distractors...", flush=True)
    cands_rows = []
    s2_path = os.path.join(data_dir, "train_source2.tsv")
    for chunk in pd.read_csv(s2_path, sep="\t", chunksize=100000, keep_default_na=False):
        m = chunk[chunk["entity_id"].isin(needed_s2)]
        cands_rows.append(m)
        if len(cands_rows) == 1:
            cands_rows.append(chunk.head(40000))
        if sum(len(x[x["entity_id"].isin(needed_s2)]) for x in cands_rows) == len(needed_s2):
            break

    s3_path = os.path.join(data_dir, "train_source3.tsv")
    for chunk in pd.read_csv(s3_path, sep="\t", chunksize=100000, keep_default_na=False):
        m = chunk[chunk["entity_id"].isin(needed_s3)]
        cands_rows.append(m)
        if len(cands_rows) == 1:
            cands_rows.append(chunk.head(40000))
        if sum(len(x[x["entity_id"].isin(needed_s3)]) for x in cands_rows) == len(needed_s3):
            break

    df_all_cands = pd.concat(cands_rows, ignore_index=True).drop_duplicates("entity_id")
    print(f"Loaded {len(df_all_cands):,} candidate pool records.", flush=True)
    df_all_cands = apply_normalization_df(df_all_cands)

    # 4. Build Decomposed Blocking Index
    print("Building Decomposed Blocking Index...", flush=True)
    t_idx = time.time()
    indexer = ChannelDecomposedIndex()
    indexer.build(df_all_cands)
    print(f"Index built in {time.time() - t_idx:.2f}s.", flush=True)

    # 5. Measure Channels on all S1 entities
    channel_names = [
        "Sorted First-2-Tokens",
        "Phonetic Key (Double Metaphone)",
        "Postal PIN + Prefix",
        "Address Street/PIN/City Keys",
        "Significant Tokens & 4-grams",
        "Acronyms",
    ]

    channel_total_cands = Counter()
    channel_true_catches = Counter()
    channel_unique_catches = Counter()

    overall_union_cands_count = 0
    overall_union_hits = 0

    print("Evaluating channel-by-channel candidate retrieval and unique recall...", flush=True)
    for s1_rec in s1_records:
        sid = s1_rec["entity_id"]
        true_set = gt_map.get(sid, set())

        ch_cands = indexer.retrieve_by_channel(s1_rec)
        union_cands = set()

        for ch_name, c_set in ch_cands.items():
            channel_total_cands[ch_name] += len(c_set)
            union_cands.update(c_set)

        overall_union_cands_count += len(union_cands)

        # Evaluate matches
        for m in true_set:
            if m in union_cands:
                overall_union_hits += 1

            retrieving_channels = [ch_name for ch_name, c_set in ch_cands.items() if m in c_set]
            for ch in retrieving_channels:
                channel_true_catches[ch] += 1

            if len(retrieving_channels) == 1:
                channel_unique_catches[retrieving_channels[0]] += 1

    overall_recall = overall_union_hits / total_true_matches * 100.0
    sum_channel_cands = sum(channel_total_cands.values())

    print("\n" + "=" * 110)
    print("BLOCKING CHANNELS RECALL & EFFICIENCY ABLATION TABLE")
    print("=" * 110)
    print(f"{'Channel Name':<32} | {'Total Hits':<11} | {'Unique Catches':<15} | {'Cand Volume':<12} | {'Vol %':<7} | {'Val Density':<11} | {'Ablation Recall Drop'}")
    print("-" * 110)

    ablation_summary = []
    for ch_name in channel_names:
        tot_hits = channel_true_catches[ch_name]
        uniq_hits = channel_unique_catches[ch_name]
        cand_vol = channel_total_cands[ch_name]
        vol_pct = (cand_vol / sum_channel_cands * 100.0) if sum_channel_cands > 0 else 0.0
        val_density = (uniq_hits / cand_vol * 1000.0) if cand_vol > 0 else 0.0
        recall_drop_pct = (uniq_hits / total_true_matches * 100.0)
        new_recall_if_ablated = (overall_union_hits - uniq_hits) / total_true_matches * 100.0

        print(f"{ch_name:<32} | {tot_hits:<11,d} | {uniq_hits:<15,d} | {cand_vol:<12,d} | {vol_pct:<6.1f}% | {val_density:<11.3f} | -{recall_drop_pct:.2f}% (drops to {new_recall_if_ablated:.2f}%)")

        ablation_summary.append({
            "channel_name": ch_name,
            "total_hits": tot_hits,
            "unique_catches": uniq_hits,
            "candidate_volume": cand_vol,
            "volume_percentage": round(vol_pct, 2),
            "value_density_per_1k": round(val_density, 3),
            "ablation_recall_drop_pct": round(recall_drop_pct, 3),
            "recall_without_channel": round(new_recall_if_ablated, 2),
        })

    print("-" * 110)
    print(f"Overall Multi-Channel Union Recall: {overall_recall:.2f}% ({overall_union_hits:,}/{total_true_matches:,} true matches captured)")
    print(f"Average Candidates per Entity (Pre-Truncation Union): {overall_union_cands_count / len(s1_records):.1f}")
    print("=" * 110)

    report = {
        "evaluation_entities": len(s1_records),
        "total_ground_truth_matches": total_true_matches,
        "overall_union_hits": overall_union_hits,
        "overall_union_recall_pct": round(overall_recall, 2),
        "avg_union_candidates_per_entity": round(overall_union_cands_count / len(s1_records), 1),
        "channel_ablation_results": ablation_summary,
        "execution_time_s": round(time.time() - t0, 1),
    }

    out_file = "models/blocking_ablation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved blocking ablation report to {out_file}")


if __name__ == "__main__":
    main()
