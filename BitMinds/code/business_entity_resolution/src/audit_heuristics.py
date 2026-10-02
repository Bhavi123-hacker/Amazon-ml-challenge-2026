"""Expanded and Calibrated Audit Heuristic Engine.

Implements robust text cleaning, abbreviation normalization, street name extraction,
and 5 distinct failure mode detectors targeting real false-merge patterns.
"""

import re
from typing import Any, Dict, List, Set, Tuple
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

NORM_MAP = {
    # US & General
    "ave": "avenue", "st": "street", "rd": "road", "blvd": "boulevard", "dr": "drive",
    "ln": "lane", "ct": "court", "pl": "place", "hwy": "highway", "ste": "suite",
    "apt": "apartment", "no": "", "pmb": "", "box": "", "po": "",
    # French
    "r": "rue", "av": "avenue", "bd": "boulevard", "all": "allee", "allee": "allee",
    "allée": "allee", "cours": "cours", "chemin": "chemin", "route": "route",
    "imp": "impasse", "impasse": "impasse", "quai": "quai", "chaussee": "chaussee",
    "chaussée": "chaussee", "bis": "", "ter": "",
    # US States
    "ny": "new york", "tx": "texas", "ca": "california", "il": "illinois", "oh": "ohio",
    "pa": "pennsylvania", "fl": "florida", "ga": "georgia", "nc": "north carolina",
    "va": "virginia", "wa": "washington", "az": "arizona", "tn": "tennessee",
    "in": "indiana", "mo": "missouri", "md": "maryland", "wi": "wisconsin",
    "co": "colorado", "mn": "minnesota", "sc": "south carolina", "al": "alabama",
    "la": "louisiana", "ky": "kentucky", "or": "oregon", "ok": "oklahoma",
    "ct": "connecticut", "ut": "utah", "ia": "iowa", "nv": "nevada", "ar": "arkansas",
    "ms": "mississippi", "ks": "kansas", "nm": "new mexico", "ne": "nebraska",
    "wv": "west virginia", "id": "idaho", "hi": "hawaii", "nh": "new hampshire",
    "me": "maine", "ri": "rhode island", "mt": "montana", "de": "delaware",
    "sd": "south dakota", "nd": "north dakota", "ak": "alaska", "vt": "vermont",
    "wy": "wyoming"
}

LEGAL_STOPWORDS = {
    # French legal entities & stopwords
    "sas", "sarl", "sasu", "sa", "sci", "eurl", "ei", "snc", "scs", "gie",
    "france", "club", "association", "comite", "comité", "ecole", "école",
    "ets", "etablissements", "établissements", "societe", "société", "groupement",
    "centre", "center", "services", "solutions", "holdings", "group", "groupe",
    "cie", "freres", "frères", "associates", "associés", "bureau", "agence",
    "foyer", "jeunes", "sportive", "danse", "musique", "culture", "loisirs",
    "amicale", "amis", "fondation", "federation", "fédération",
    # English / US / India legal entities & generic words
    "llc", "inc", "corp", "ltd", "pvt", "limited", "co", "company", "plc",
    "corporation", "incorporated", "enterprises", "industries", "trading",
    "holdings", "management", "consulting", "enterprises", "ventures",
    "india", "international", "national", "trust", "hospital", "clinic"
}

STREET_TYPES = {
    "street", "avenue", "road", "boulevard", "drive", "lane", "court", "place",
    "highway", "way", "circle", "rue", "cours", "chemin", "route", "impasse",
    "quai", "chaussee", "allee", "parvis", "square", "cross", "main", "marg"
}

def clean_and_normalize(s: str) -> str:
    """Normalize text with token expansion and punctuation removal."""
    if not s or not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"[\#\.\,\-\/\(\)\[\]\:\;\'\"]", " ", s)
    tokens = s.split()
    tokens = [NORM_MAP.get(t, t) for t in tokens if t]
    return " ".join(tokens)

def extract_core_name(name: str) -> str:
    """Extract distinct core business stem."""
    clean = clean_and_normalize(name)
    tokens = clean.split()
    core = [t for t in tokens if t not in LEGAL_STOPWORDS and len(t) > 1]
    return " ".join(core) if core else clean

def extract_street_number(addr: str) -> str:
    """Extract street/building number."""
    m = re.search(r"\b(\d+[a-zA-Z]?)\b", addr)
    return m.group(1).lower() if m else ""

def extract_street_name_tokens(addr: str) -> Set[str]:
    """Extract distinct street tokens after removing numbers and generic street types."""
    clean = clean_and_normalize(addr)
    tokens = clean.split()
    st_toks = set()
    for t in tokens:
        if t.isdigit() or re.match(r"^\d+[a-z]?$", t):
            continue
        if t in STREET_TYPES or t in NORM_MAP.values():
            continue
        if len(t) > 2:
            st_toks.add(t)
    return st_toks

def compute_token_jaccard(toks1: Set[str], toks2: Set[str]) -> float:
    if not toks1 or not toks2:
        return 0.0
    return len(toks1 & toks2) / len(toks1 | toks2)

def evaluate_pair_audit(
    s1_name: str,
    s1_addr: str,
    c_name: str,
    c_addr: str,
) -> Dict[str, Any]:
    """Evaluate pairwise similarity metrics and robust heuristic flags."""
    s1_name_n = clean_and_normalize(s1_name)
    c_name_n = clean_and_normalize(c_name)
    s1_addr_n = clean_and_normalize(s1_addr)
    c_addr_n = clean_and_normalize(c_addr)

    name_tok_sort = fuzz.token_sort_ratio(s1_name_n, c_name_n) / 100.0
    name_lev = fuzz.ratio(s1_name_n, c_name_n) / 100.0
    s1_name_toks = set(s1_name_n.split())
    c_name_toks = set(c_name_n.split())
    name_jaccard = compute_token_jaccard(s1_name_toks, c_name_toks)

    addr_tok_sort = fuzz.token_sort_ratio(s1_addr_n, c_addr_n) / 100.0
    addr_lev = fuzz.ratio(s1_addr_n, c_addr_n) / 100.0
    s1_addr_toks = set(s1_addr_n.split())
    c_addr_toks = set(c_addr_n.split())
    addr_jaccard = compute_token_jaccard(s1_addr_toks, c_addr_toks)

    # Core names
    s1_core = extract_core_name(s1_name)
    c_core = extract_core_name(c_name)
    s1_core_toks = set(s1_core.split())
    c_core_toks = set(c_core.split())
    core_name_tok_sort = fuzz.token_sort_ratio(s1_core, c_core) / 100.0 if s1_core and c_core else name_tok_sort
    core_name_lev = fuzz.ratio(s1_core, c_core) / 100.0 if s1_core and c_core else name_lev
    core_name_jaccard = compute_token_jaccard(s1_core_toks, c_core_toks) if s1_core and c_core else name_jaccard

    # Street numbers and street names
    num1 = extract_street_number(s1_addr)
    num2 = extract_street_number(c_addr)
    num_match = (num1 == num2) if (num1 and num2) else False
    num_mismatch = (num1 != num2) if (num1 and num2) else False

    st_toks1 = extract_street_name_tokens(s1_addr)
    st_toks2 = extract_street_name_tokens(c_addr)
    st_jaccard = compute_token_jaccard(st_toks1, st_toks2)
    st_name1 = " ".join(sorted(st_toks1))
    st_name2 = " ".join(sorted(st_toks2))
    st_sim = fuzz.token_sort_ratio(st_name1, st_name2) / 100.0 if st_name1 and st_name2 else 0.0

    # 1. Old Heuristic
    old_suspicious = (addr_lev >= 0.90 or addr_tok_sort >= 0.90) and (name_lev < 0.60 and name_tok_sort < 0.60)
    old_reverse = (name_lev >= 0.85 or name_tok_sort >= 0.85) and (addr_lev < 0.40 and addr_tok_sort < 0.40)
    old_flagged = bool(old_suspicious or old_reverse)

    # 2. Expanded Heuristic Flags:
    # Mode 1: Co-located Distinct Entity (Shared street/address/number, completely distinct business)
    mode1_colocation = bool(
        (addr_tok_sort >= 0.80 or addr_jaccard >= 0.50 or (num_match and st_sim >= 0.60)) and
        (core_name_tok_sort < 0.55 and core_name_jaccard < 0.35)
    )

    # Mode 2: Suffix & Generic Word Inflation (Raw name overlap inflated by SAS, France, Club, etc., but core brand distinct)
    mode2_suffix_inflation = bool(
        (core_name_jaccard == 0.0 and core_name_tok_sort < 0.40) and
        (name_tok_sort >= 0.55)
    )

    # Mode 3: Chain / Franchise Divergence (Brand matches, but completely different city/street)
    mode3_franchise = bool(
        (core_name_tok_sort >= 0.75 or name_tok_sort >= 0.80) and
        (addr_tok_sort < 0.45 and addr_jaccard < 0.30 and st_jaccard == 0.0)
    )

    # Mode 4: Shared Number, Divergent Street (Same building number, but completely different street name)
    mode4_street_mismatch = bool(
        num_match and (st_jaccard == 0.0 and st_sim < 0.35) and (core_name_tok_sort < 0.85)
    )

    # Mode 5: Low Mutual Agreement (Both core name and address are weak)
    mode5_low_mutual = bool(
        (core_name_tok_sort < 0.65 and core_name_jaccard < 0.40) and
        (addr_tok_sort < 0.65 and addr_jaccard < 0.40)
    )

    expanded_flagged = bool(
        mode1_colocation or
        mode2_suffix_inflation or
        mode3_franchise or
        mode4_street_mismatch or
        mode5_low_mutual or
        old_flagged
    )

    return {
        "name_tok_sort": round(name_tok_sort, 4),
        "name_lev": round(name_lev, 4),
        "addr_tok_sort": round(addr_tok_sort, 4),
        "addr_lev": round(addr_lev, 4),
        "core_name_tok_sort": round(core_name_tok_sort, 4),
        "core_name_jaccard": round(core_name_jaccard, 4),
        "num_match": num_match,
        "num_mismatch": num_mismatch,
        "st_jaccard": round(st_jaccard, 4),
        "st_sim": round(st_sim, 4),
        "old_flagged": old_flagged,
        "expanded_flagged": expanded_flagged,
        "mode1_colocation": mode1_colocation,
        "mode2_suffix_inflation": mode2_suffix_inflation,
        "mode3_franchise": mode3_franchise,
        "mode4_street_mismatch": mode4_street_mismatch,
        "mode5_low_mutual": mode5_low_mutual,
    }
