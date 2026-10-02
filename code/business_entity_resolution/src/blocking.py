"""Candidate Generation / Blocking Module (Stage 1).

Implements multi-channel blocking partitioned strictly by country:
1. Exact Country Partitioning (hard filter: country_s1 == country_candidate)
2. Sorted First-2-Tokens Inverted Index (drops generic stopwords, transliteration-aware)
3. Phonetic Inverted Index (Double Metaphone on primary name token)
4. Address & Street Number Inverted Index (street_num + street_token, pin + street_num)
   - Excludes dangerous street_num + city collisions (Part B Fix #3).
5. Significant Token & Prefix 4-gram Inverted Index
6. Acronym Inverted Index (e.g. 'ZB' <-> 'Zander Blue')
7. Fast Composite Similarity Pre-filter (top-K nearest candidates per S1)
"""

import os
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from metaphone import dm
from rapidfuzz import fuzz

from src.features import transliterate_indic

GENERIC_STOPS = {
    "the", "and", "of", "for", "in", "to", "a", "an", "de", "la", "du",
    "company", "services", "corporation", "limited", "private", "enterprises",
    "solutions", "inc", "llc", "ltd", "corp", "pvt",
}

ADDRESS_STOPS = {
    "road", "street", "avenue", "drive", "lane", "boulevard", "floor",
    "unit", "apartment", "suite", "near", "opp", "building", "block",
    "sector", "plot", "door", "no", "pobox", "st", "ave", "rd", "blvd",
}


def get_first_two_tokens(name_norm: str) -> str:
    """Extract and sort first two significant tokens (transliteration aware)."""
    norm = transliterate_indic(name_norm)
    tokens = [t for t in norm.split() if len(t) > 1 and t not in GENERIC_STOPS]
    if not tokens:
        return norm[:4] if norm else ""
    if len(tokens) == 1:
        return tokens[0]
    return "_".join(sorted(tokens[:2]))


def get_phonetic_key(name_norm: str) -> str:
    """Extract Double Metaphone phonetic key of primary token."""
    norm = transliterate_indic(name_norm)
    tokens = [t for t in norm.split() if len(t) > 1 and t not in GENERIC_STOPS]
    if not tokens:
        return ""
    first = tokens[0]
    try:
        res = dm(first)
        return res[0] if res and res[0] else first[:4]
    except Exception:
        return first[:4]


def get_acronym(tokens: List[str]) -> str:
    """Extract acronym from list of tokens."""
    if len(tokens) < 2:
        return ""
    acronym = "".join(t[0] for t in tokens if t and t[0].isalnum())
    return acronym if 2 <= len(acronym) <= 4 else ""


class BlockingIndex:
    """Multi-channel inverted index partitioned strictly by country."""

    def __init__(self):
        self.idx_tok2 = defaultdict(lambda: defaultdict(list))
        self.idx_phone = defaultdict(lambda: defaultdict(list))
        self.idx_postal = defaultdict(lambda: defaultdict(list))
        self.idx_addr = defaultdict(lambda: defaultdict(list))
        self.idx_sigtoken = defaultdict(lambda: defaultdict(list))
        self.idx_prefix = defaultdict(lambda: defaultdict(list))
        self.cand_records = []

    def build(self, df_candidates: pd.DataFrame):
        """Index candidate records from Source 2 and Source 3."""
        self.cand_records = df_candidates[[
            "entity_id", "country", "name_norm", "addr_norm",
            "pin_extracted", "city_extracted", "street_number"
        ]].to_dict("records")

        for i, rec in enumerate(self.cand_records):
            country = rec["country"]
            n_norm = rec["name_norm"]
            pin = rec["pin_extracted"]
            st_num = rec["street_number"]
            addr_norm = rec["addr_norm"]
            city = rec["city_extracted"]

            # 1. Sorted first-2-token (with Indic transliteration)
            k_tok2 = get_first_two_tokens(n_norm)
            if k_tok2:
                self.idx_tok2[country][k_tok2].append(i)

            # 2. Phonetic
            k_phone = get_phonetic_key(n_norm)
            if k_phone:
                self.idx_phone[country][k_phone].append(i)

            # 3. Postal + token prefix
            if pin:
                first_tok = transliterate_indic(n_norm).split()[0][:3] if n_norm else ""
                self.idx_postal[country][f"{pin}_{first_tok}"].append(i)

            # 4. Address Street Number + Street tokens / PIN
            # (Excludes st_num + city alone to prevent massive building-number city collisions)
            if st_num:
                addr_tokens = [
                    t for t in addr_norm.split()
                    if t != st_num and len(t) > 2 and t not in ADDRESS_STOPS
                ]
                for t in addr_tokens[:3]:
                    self.idx_addr[country][f"{st_num}_{t}"].append(i)
                if pin:
                    self.idx_addr[country][f"{pin}_{st_num}"].append(i)

            # 5. Significant tokens and prefix 4-grams
            norm_trans = transliterate_indic(n_norm)
            toks = [t for t in norm_trans.split() if t not in GENERIC_STOPS]
            for tok in toks:
                if len(tok) >= 4:
                    self.idx_sigtoken[country][tok].append(i)
                    self.idx_prefix[country][tok[:4]].append(i)

            # 6. Acronyms
            if len(toks) >= 2:
                acr = get_acronym(toks)
                if acr:
                    self.idx_sigtoken[country][acr].append(i)
            elif len(toks) == 1 and 2 <= len(toks[0]) <= 4:
                self.idx_sigtoken[country][toks[0]].append(i)

    def retrieve_candidates_for_entity(
        self,
        s1_rec: dict,
        top_k: int = 50,
    ) -> List[Tuple[str, float]]:
        """Retrieve and rank candidates for a single Source 1 entity record.

        Returns:
            List of (candidate_entity_id, fast_score) sorted descending by score.
        """
        country = s1_rec["country"]
        n_norm = s1_rec["name_norm"]
        pin = s1_rec["pin_extracted"]
        st_num = s1_rec["street_number"]
        addr_norm = s1_rec["addr_norm"]
        city = s1_rec["city_extracted"]

        cand_counts = Counter()

        # 1. Sorted first-2-token
        k_tok2 = get_first_two_tokens(n_norm)
        if k_tok2:
            for idx in self.idx_tok2[country].get(k_tok2, []):
                cand_counts[idx] += 12

        # 2. Phonetic
        k_phone = get_phonetic_key(n_norm)
        if k_phone:
            for idx in self.idx_phone[country].get(k_phone, []):
                cand_counts[idx] += 6

        # 3. Postal + token prefix
        if pin:
            first_tok = transliterate_indic(n_norm).split()[0][:3] if n_norm else ""
            for idx in self.idx_postal[country].get(f"{pin}_{first_tok}", []):
                cand_counts[idx] += 8

        # 4. Address Street Number + Street tokens / PIN
        if st_num:
            addr_tokens = [
                t for t in addr_norm.split()
                if t != st_num and len(t) > 2 and t not in ADDRESS_STOPS
            ]
            for t in addr_tokens[:3]:
                for idx in self.idx_addr[country].get(f"{st_num}_{t}", []):
                    cand_counts[idx] += 10
            if pin:
                for idx in self.idx_addr[country].get(f"{pin}_{st_num}", []):
                    cand_counts[idx] += 10

        # 5. Significant tokens and prefix 4-grams (capped to avoid noisy lists)
        norm_trans = transliterate_indic(n_norm)
        toks = [t for t in norm_trans.split() if t not in GENERIC_STOPS]
        for tok in toks:
            if len(tok) >= 4:
                for idx in self.idx_sigtoken[country].get(tok, []):
                    cand_counts[idx] += 5
                p_list = self.idx_prefix[country].get(tok[:4], [])
                if len(p_list) <= 500:
                    for idx in p_list:
                        cand_counts[idx] += 2

        # 6. Acronyms
        if len(toks) >= 2:
            acr = get_acronym(toks)
            if acr:
                for idx in self.idx_sigtoken[country].get(acr, []):
                    cand_counts[idx] += 6
        elif len(toks) == 1 and 2 <= len(toks[0]) <= 4:
            for idx in self.idx_sigtoken[country].get(toks[0], []):
                cand_counts[idx] += 4

        if not cand_counts:
            return []

        # Pre-select top 2x candidates using channel score, then compute exact blocking_score
        pre_selected = [idx for idx, _ in cand_counts.most_common(max(top_k * 2, 60))]

        # Fast candidate ranking with RapidFuzz
        scored = []
        n_trans = transliterate_indic(n_norm)
        for idx in pre_selected:
            c_rec = self.cand_records[idx]
            c_trans = transliterate_indic(c_rec["name_norm"])
            s_name = max(
                fuzz.token_set_ratio(n_norm, c_rec["name_norm"]),
                fuzz.token_set_ratio(n_trans, c_trans)
            )
            s_addr = fuzz.token_set_ratio(addr_norm, c_rec["addr_norm"])
            score = max(s_name, s_addr) + 0.3 * min(s_name, s_addr)
            scored.append((c_rec["entity_id"], float(score)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


def export_candidate_pairs_tsv(
    s1_ids: List[str],
    candidates_dict: Dict[str, List[str]],
    output_path: str,
):
    """Export candidate_pairs.tsv matching exact challenge format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_ids:
            cands = candidates_dict.get(s1_id, [])
            cand_str = ",".join(cands) if cands else ""
            f.write(f"{s1_id}\t{cand_str}\n")
