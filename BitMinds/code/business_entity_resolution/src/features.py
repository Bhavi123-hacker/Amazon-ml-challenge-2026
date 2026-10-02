"""Pairwise Feature Engineering Module (Stage 2).

Extracts high-discriminative string, token, address, core-stem, and structural metrics
for every (s1, candidate) pair using RapidFuzz for high performance.
"""

import re
from typing import Any, Dict, List, Optional, Set
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

LEGAL_STOPWORDS = {
    # French legal entities & stopwords
    "sas", "sarl", "sasu", "sa", "sci", "eurl", "ei", "snc", "scs", "gie",
    "france", "club", "association", "comite", "comité", "ecole", "école",
    "ets", "etablissements", "établissements", "societe", "société", "groupement",
    "centre", "center", "services", "solutions", "holdings", "group", "groupe",
    "cie", "freres", "frères", "associates", "associés", "bureau", "agence",
    # English / US / India legal entities & generic words
    "llc", "inc", "corp", "ltd", "pvt", "limited", "co", "company", "plc",
    "corporation", "incorporated", "enterprises", "industries", "trading",
    "holdings", "management", "consulting", "ventures", "india", "international",
    "national", "trust", "hospital", "clinic"
}

STREET_TYPES = {
    "street", "avenue", "road", "boulevard", "drive", "lane", "court", "place",
    "highway", "way", "circle", "rue", "cours", "chemin", "route", "impasse",
    "quai", "chaussee", "allee", "parvis", "square", "cross", "main", "marg"
}


def extract_core_name_tokens(name_norm: str) -> List[str]:
    """Extract tokens belonging to the distinctive core business name."""
    tokens = name_norm.split()
    core = [t for t in tokens if t not in LEGAL_STOPWORDS and len(t) > 1]
    return core if core else tokens


def extract_street_tokens(addr_norm: str, city_norm: str = "", state_norm: str = "") -> Set[str]:
    """Extract street name tokens excluding numbers, common street types, city, and region tokens."""
    tokens = addr_norm.split()
    city_toks = set(city_norm.split()) | set(state_norm.split())
    city_toks.update({"france", "nord", "hauts", "rhone", "alpes", "provence", "cote", "dazur", "ile", "paris", "haute", "bas"})
    st_toks = set()
    for t in tokens:
        clean_t = re.sub(r"[^\w\s]", "", t)
        if not clean_t or clean_t.isdigit() or re.match(r"^\d+[a-z]?$", clean_t):
            continue
        if clean_t in STREET_TYPES or len(clean_t) <= 2 or clean_t in city_toks:
            continue
        st_toks.add(clean_t)
    return st_toks


def compute_token_jaccard(toks1: set, toks2: set) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not toks1 or not toks2:
        return 0.0
    intersection = len(toks1 & toks2)
    union = len(toks1 | toks2)
    return intersection / union if union > 0 else 0.0


def check_acronym_match(name1: str, name2: str, toks1: List[str], toks2: List[str]) -> int:
    """Check if either name is an acronym of the other."""
    if not toks1 or not toks2:
        return 0
    if len(toks1) == 1 and 2 <= len(toks1[0]) <= 5 and len(toks2) >= 2:
        acr2 = "".join(t[0] for t in toks2 if t and t[0].isalnum())
        if toks1[0] == acr2:
            return 1
    if len(toks2) == 1 and 2 <= len(toks2[0]) <= 5 and len(toks1) >= 2:
        acr1 = "".join(t[0] for t in toks1 if t and t[0].isalnum())
        if toks2[0] == acr1:
            return 1
    return 0


def extract_pair_features(
    s1_dict: Dict[str, Any],
    cand_dict: Dict[str, Any],
    rank: int = 0,
    blocking_score: float = 0.0,
) -> Dict[str, float]:
    """Compute rich pairwise features for a single (S1, Candidate) pair."""
    s1_name = s1_dict.get("name_norm", "")
    c_name = cand_dict.get("name_norm", "")
    s1_addr = s1_dict.get("addr_norm", "")
    c_addr = cand_dict.get("addr_norm", "")

    s1_name_toks = s1_name.split()
    c_name_toks = c_name.split()
    s1_name_set = set(s1_name_toks)
    c_name_set = set(c_name_toks)

    s1_addr_toks = s1_addr.split()
    c_addr_toks = c_addr.split()
    s1_addr_set = set(s1_addr_toks)
    c_addr_set = set(c_addr_toks)

    # 1. Standard Name features
    name_lev = fuzz.ratio(s1_name, c_name) / 100.0
    name_jw = JaroWinkler.similarity(s1_name, c_name)
    name_token_sort = fuzz.token_sort_ratio(s1_name, c_name) / 100.0
    name_token_set = fuzz.token_set_ratio(s1_name, c_name) / 100.0
    name_partial = fuzz.partial_ratio(s1_name, c_name) / 100.0
    name_jaccard = compute_token_jaccard(s1_name_set, c_name_set)
    name_common = float(len(s1_name_set & c_name_set))
    name_acronym = float(check_acronym_match(s1_name, c_name, s1_name_toks, c_name_toks))

    len_s1 = len(s1_name)
    len_c = len(c_name)
    name_len_diff = float(abs(len_s1 - len_c))
    name_len_ratio = float(min(len_s1, len_c) / max(len_s1, len_c, 1))
    name_token_count_diff = float(abs(len(s1_name_toks) - len(c_name_toks)))

    # Legal suffix match
    s1_suf = s1_dict.get("legal_suffix", "")
    c_suf = cand_dict.get("legal_suffix", "")
    if s1_suf and c_suf:
        suffix_match = 1.0 if s1_suf == c_suf else -1.0
    else:
        suffix_match = 0.0

    # 2. Core Business Name Features (stripping suffix inflation)
    s1_core_toks = extract_core_name_tokens(s1_name)
    c_core_toks = extract_core_name_tokens(c_name)
    s1_core_str = " ".join(s1_core_toks)
    c_core_str = " ".join(c_core_toks)
    s1_core_set = set(s1_core_toks)
    c_core_set = set(c_core_toks)

    core_name_lev = fuzz.ratio(s1_core_str, c_core_str) / 100.0 if s1_core_str and c_core_str else name_lev
    core_name_tok_sort = fuzz.token_sort_ratio(s1_core_str, c_core_str) / 100.0 if s1_core_str and c_core_str else name_token_sort
    core_name_jaccard = compute_token_jaccard(s1_core_set, c_core_set)
    name_suffix_inflation_gap = max(0.0, name_token_sort - core_name_tok_sort)

    # 3. Standard Address features
    addr_lev = fuzz.ratio(s1_addr, c_addr) / 100.0
    addr_jw = JaroWinkler.similarity(s1_addr, c_addr)
    addr_token_sort = fuzz.token_sort_ratio(s1_addr, c_addr) / 100.0
    addr_token_set = fuzz.token_set_ratio(s1_addr, c_addr) / 100.0
    addr_partial = fuzz.partial_ratio(s1_addr, c_addr) / 100.0
    addr_jaccard = compute_token_jaccard(s1_addr_set, c_addr_set)
    addr_common = float(len(s1_addr_set & c_addr_set))

    # PIN matching
    s1_pin = s1_dict.get("pin_extracted", "")
    c_pin = cand_dict.get("pin_extracted", "")
    if s1_pin and c_pin:
        pin_match = 1.0 if s1_pin == c_pin else 0.0
        pin_mismatch = 1.0 if s1_pin != c_pin else 0.0
    else:
        pin_match = 0.0
        pin_mismatch = 0.0

    # Street number matching
    s1_st = s1_dict.get("street_number", "")
    c_st = cand_dict.get("street_number", "")
    if s1_st and c_st:
        st_match = 1.0 if s1_st == c_st else 0.0
        st_mismatch = 1.0 if s1_st != c_st else 0.0
    else:
        st_match = 0.0
        st_mismatch = 0.0

    # City matching
    s1_city = s1_dict.get("city_extracted", "")
    c_city = cand_dict.get("city_extracted", "")
    if s1_city and c_city:
        city_match = 1.0 if s1_city == c_city else 0.0
    else:
        city_match = 0.0

    addr_len_diff = float(abs(len(s1_addr) - len(c_addr)))

    # 4. Street Name Distinct Features
    s1_city = s1_dict.get("city_extracted", "")
    c_city = cand_dict.get("city_extracted", "")
    s1_st_toks = extract_street_tokens(s1_addr, s1_city)
    c_st_toks = extract_street_tokens(c_addr, c_city)
    st_name_jaccard = compute_token_jaccard(s1_st_toks, c_st_toks)
    s1_st_str = " ".join(sorted(s1_st_toks))
    c_st_str = " ".join(sorted(c_st_toks))
    street_name_sim = fuzz.token_sort_ratio(s1_st_str, c_st_str) / 100.0 if s1_st_str and c_st_str else 0.0
    # Street mismatch with shared number (critical co-location flag)
    street_mismatch_with_shared_num = 1.0 if (st_match == 1.0 and st_name_jaccard == 0.0 and street_name_sim < 0.35) else 0.0
    # Street difference flag: different numbers AND different street names
    street_diff_flag = 1.0 if (st_mismatch == 1.0 and st_name_jaccard < 0.2 and street_name_sim < 0.35) else 0.0
    exact_name_match = 1.0 if (s1_name and c_name and s1_name == c_name) else 0.0
    exact_addr_match = 1.0 if (s1_addr and c_addr and s1_addr == c_addr) else 0.0

    # 5. Structural & Derived Interactions
    country_match = 1.0 if s1_dict.get("country") == cand_dict.get("country") else 0.0
    cand_id = cand_dict.get("entity_id", "")
    source_type = 2.0 if cand_id.startswith("S2-") else (3.0 if cand_id.startswith("S3-") else 0.0)

    sim_max = max(name_jw, addr_token_sort)
    sim_mean = (name_jw + addr_token_sort) / 2.0
    interaction_name_addr = name_jw * addr_token_sort
    interaction_core_name_addr = core_name_tok_sort * addr_token_sort
    sim_ratio = core_name_tok_sort / (addr_token_sort + 0.01)
    interaction_name_pin = name_jw * pin_match
    interaction_name_street = name_jw * st_match

    return {
        "name_levenshtein": name_lev,
        "name_jaro_winkler": name_jw,
        "name_token_sort": name_token_sort,
        "name_token_set": name_token_set,
        "name_partial_ratio": name_partial,
        "name_jaccard": name_jaccard,
        "name_common_tokens": name_common,
        "name_acronym_match": name_acronym,
        "name_length_diff": name_len_diff,
        "name_length_ratio": name_len_ratio,
        "name_token_count_diff": name_token_count_diff,
        "legal_suffix_match": suffix_match,
        "core_name_lev": core_name_lev,
        "core_name_tok_sort": core_name_tok_sort,
        "core_name_jaccard": core_name_jaccard,
        "name_suffix_inflation_gap": name_suffix_inflation_gap,
        "addr_levenshtein": addr_lev,
        "addr_jaro_winkler": addr_jw,
        "addr_token_sort": addr_token_sort,
        "addr_token_set": addr_token_set,
        "addr_partial_ratio": addr_partial,
        "addr_jaccard": addr_jaccard,
        "addr_common_tokens": addr_common,
        "pin_exact_match": pin_match,
        "pin_mismatch": pin_mismatch,
        "street_number_match": st_match,
        "street_number_mismatch": st_mismatch,
        "street_name_sim": street_name_sim,
        "st_name_jaccard": st_name_jaccard,
        "street_mismatch_with_shared_num": street_mismatch_with_shared_num,
        "street_diff_flag": street_diff_flag,
        "exact_name_match": exact_name_match,
        "exact_addr_match": exact_addr_match,
        "city_exact_match": city_match,
        "addr_length_diff": addr_len_diff,
        "country_exact_match": country_match,
        "candidate_source": source_type,
        "sim_max": sim_max,
        "sim_mean": sim_mean,
        "interaction_name_addr": interaction_name_addr,
        "interaction_core_name_addr": interaction_core_name_addr,
        "sim_ratio": sim_ratio,
        "interaction_name_pin": interaction_name_pin,
        "interaction_name_street": interaction_name_street,
    }


FEATURE_COLUMNS = [
    "name_levenshtein", "name_jaro_winkler", "name_token_sort", "name_token_set",
    "name_partial_ratio", "name_jaccard", "name_common_tokens", "name_acronym_match",
    "name_length_diff", "name_length_ratio", "name_token_count_diff", "legal_suffix_match",
    "core_name_lev", "core_name_tok_sort", "core_name_jaccard", "name_suffix_inflation_gap",
    "addr_levenshtein", "addr_jaro_winkler", "addr_token_sort", "addr_token_set",
    "addr_partial_ratio", "addr_jaccard", "addr_common_tokens", "pin_exact_match",
    "pin_mismatch", "street_number_match", "street_number_mismatch", "street_name_sim",
    "st_name_jaccard", "street_mismatch_with_shared_num", "street_diff_flag",
    "exact_name_match", "exact_addr_match",
    "city_exact_match", "addr_length_diff", "country_exact_match", "candidate_source",
    "sim_max", "sim_mean", "interaction_name_addr", "interaction_core_name_addr", "sim_ratio",
    "interaction_name_pin", "interaction_name_street"
]
