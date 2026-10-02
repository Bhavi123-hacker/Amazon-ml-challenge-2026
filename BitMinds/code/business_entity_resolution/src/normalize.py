"""Module for text normalization and entity cleaning (Stage 0).

Applied strictly non-destructively to copies of raw fields.
Generates:
- name_norm: cleaned, de-accented, suffix-stripped business name
- legal_suffix: detected entity legal suffix (US, India, France, etc.)
- addr_norm: expanded abbreviations, cleaned address
- pin_extracted: extracted 5-digit or 6-digit postal code
- city_extracted: estimated city / locality string
- street_number: building or plot number
- has_landmark: indicator flag for landmark-based addresses (e.g. 'near X')
"""

import re
import unicodedata
from typing import Dict, Optional, Tuple
import pandas as pd

# Multi-jurisdiction legal suffixes (ordered from multi-word to single-word)
LEGAL_SUFFIX_PATTERNS = [
    # Multi-word English / Indian / US
    (r"\bprivate\s+limited\b", "pvt_ltd"),
    (r"\bpvt\s+ltd\b", "pvt_ltd"),
    (r"\bpvt\s+limited\b", "pvt_ltd"),
    (r"\bprivate\s+ltd\b", "pvt_ltd"),
    (r"\blimited\s+liability\s+company\b", "llc"),
    (r"\blimited\s+liability\s+partnership\b", "llp"),
    (r"\bpublic\s+limited\s+company\b", "plc"),
    # French multi-word
    (r"\bsociete\s+a\s+responsabilite\s+limitee\b", "sarl"),
    (r"\bsociete\s+par\s+actions\s+simplifiee\s+unipersonnelle\b", "sasu"),
    (r"\bsociete\s+par\s+actions\s+simplifiee\b", "sas"),
    (r"\bsociete\s+anonyme\b", "sa"),
    (r"\bentreprise\s+unipersonnelle\s+a\s+responsabilite\s+limitee\b", "eurl"),
    (r"\bsociete\s+civile\s+immobiliere\b", "sci"),
    (r"\bsociete\s+en\s+nom\s+collectif\b", "snc"),
    (r"\bgroupement\s+d\s+interet\s+economique\b", "gie"),
    # Single-word English / US
    (r"\bincorporated\b", "inc"),
    (r"\bcorporation\b", "corp"),
    (r"\bcompany\b", "co"),
    (r"\blimited\b", "ltd"),
    (r"\binc\b", "inc"),
    (r"\bcorp\b", "corp"),
    (r"\bllc\b", "llc"),
    (r"\bllp\b", "llp"),
    (r"\bplc\b", "plc"),
    (r"\bpllc\b", "pllc"),
    (r"\bltd\b", "ltd"),
    (r"\bpvt\b", "pvt"),
    (r"\bco\b", "co"),
    # French abbreviations
    (r"\bsarl\b", "sarl"),
    (r"\bsas\b", "sas"),
    (r"\bsasu\b", "sasu"),
    (r"\bsa\b", "sa"),
    (r"\beurl\b", "eurl"),
    (r"\bsci\b", "sci"),
    (r"\bsnc\b", "snc"),
    (r"\bgie\b", "gie"),
    (r"\bste\b", "ste"),
    # Hindi legal terms
    (r"प्राइवेट\s+लिमिटेड", "pvt_ltd"),
    (r"प्राइवेट", "pvt"),
    (r"लिमिटेड", "ltd"),
    (r"एलएलपी", "llp"),
]

# Address abbreviation mappings
ADDRESS_ABBR_MAP = {
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bblvd\b": "boulevard",
    r"\bct\b": "court",
    r"\bpkwy\b": "parkway",
    r"\bhwy\b": "highway",
    r"\bapt\b": "apartment",
    r"\bste\b": "suite",
    r"\bfl\b": "floor",
    r"\bdept\b": "department",
    r"\bbldg\b": "building",
    r"\bpo\s+box\b": "pobox",
    r"\bp\.o\.\s*box\b": "pobox",
    # French address terms
    r"\bav\b": "avenue",
    r"\bbd\b": "boulevard",
    r"\bpl\b": "place",
    r"\ball\b": "allee",
    r"\brte\b": "route",
}

# Landmark indicator phrases
LANDMARK_REGEX = re.compile(
    r"\b(near|opp|opposite|behind|beside|adjacent|next\s+to|in\s+front\s+of|near\s+by|close\s+to)\b",
    re.IGNORECASE,
)

# Postal code regexes
REGEX_INDIA_PIN = re.compile(r"\b([1-9][0-9]{5})\b")
REGEX_US_ZIP = re.compile(r"\b([0-9]{5})(?:-[0-9]{4})?\b")
REGEX_FRANCE_CP = re.compile(r"\b([0-9]{5})\b")
REGEX_STREET_NUM = re.compile(r"\b(\d+([a-zA-Z]|\/\d+)?)\b")


def strip_accents(text: str) -> str:
    """Unicode normalization to remove Latin accents without stripping non-Latin scripts (Devanagari, etc.)."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    out = []
    for c in nfkd:
        # 0x0300 to 0x036F are Combining Diacritical Marks for Latin (e.g. é -> e)
        if 0x0300 <= ord(c) <= 0x036F:
            continue
        out.append(c)
    return unicodedata.normalize("NFC", "".join(out))


def clean_base_text(text: str) -> str:
    """Basic cleaning: accent removal, symbol conversion, punctuation stripping while preserving Unicode marks."""
    if not text or not isinstance(text, str):
        return ""
    # Strip Latin accents
    t = strip_accents(text).lower()
    # Symbol expansion
    t = re.sub(r"&", " and ", t)
    t = re.sub(r"@", " at ", t)
    t = re.sub(r"\+", " plus ", t)
    t = re.sub(r"/", " ", t)
    t = re.sub(r"[_\-–—]", " ", t)
    t = re.sub(r"\.", " ", t)
    
    # Remove punctuation while preserving alphanumeric characters and script marks (Mn, Mc) across all languages
    out = []
    for c in t:
        cat = unicodedata.category(c)
        if c.isalnum() or c.isspace() or cat.startswith("M"):
            out.append(c)
        else:
            out.append(" ")
    
    # Collapse multiple whitespace
    return re.sub(r"\s+", " ", "".join(out)).strip()


def normalize_name(raw_name: str) -> Tuple[str, str]:
    """Normalize business name and extract legal suffix.

    Returns:
        (name_norm, legal_suffix)
    """
    cleaned = clean_base_text(raw_name)
    if not cleaned:
        return "", ""

    legal_suffix = ""
    core_name = cleaned

    for pattern, suffix_label in LEGAL_SUFFIX_PATTERNS:
        match = re.search(pattern, core_name)
        if match:
            if not legal_suffix:
                legal_suffix = suffix_label
            # Remove suffix from core name
            core_name = re.sub(pattern, " ", core_name)

    core_name = re.sub(r"\s+", " ", core_name).strip()
    # If stripping suffixes left the name empty, revert to cleaned
    if not core_name:
        core_name = cleaned

    return core_name, legal_suffix


def extract_postal_code(raw_address: str, country: str = "") -> str:
    """Extract postal code / PIN code based on country or general pattern."""
    if not raw_address or not isinstance(raw_address, str):
        return ""

    c = country.strip().upper() if country else ""
    if c == "INDIA":
        m = REGEX_INDIA_PIN.search(raw_address)
        if m:
            return m.group(1)
    elif c in ("US", "FRANCE"):
        m = REGEX_US_ZIP.search(raw_address)
        if m:
            return m.group(1)
    else:
        # Fallback: check 6-digit first, then 5-digit
        m6 = REGEX_INDIA_PIN.search(raw_address)
        if m6:
            return m6.group(1)
        m5 = REGEX_US_ZIP.search(raw_address)
        if m5:
            return m5.group(1)

    return ""


def extract_street_number(raw_address: str) -> str:
    """Extract street / building number from address."""
    if not raw_address or not isinstance(raw_address, str):
        return ""
    m = REGEX_STREET_NUM.search(raw_address)
    return m.group(1) if m else ""


def extract_city(raw_address: str) -> str:
    """Heuristic extraction of city / locality from comma-separated address parts."""
    if not raw_address or not isinstance(raw_address, str):
        return ""
    parts = [p.strip() for p in raw_address.split(",") if p.strip()]
    if len(parts) >= 2:
        # Usually city is penultimate or middle part
        candidate = parts[-2] if len(parts) >= 3 else parts[-1]
        # Clean up numbers from candidate
        candidate_clean = re.sub(r"\d+", "", candidate).strip()
        if len(candidate_clean) > 2:
            return clean_base_text(candidate_clean)
    return ""


def normalize_address(raw_address: str, country: str = "") -> Tuple[str, str, str, str, int]:
    """Normalize address string and extract components.

    Returns:
        (addr_norm, pin_extracted, city_extracted, street_number, has_landmark)
    """
    if not raw_address or not isinstance(raw_address, str):
        return "", "", "", "", 0

    has_landmark = 1 if LANDMARK_REGEX.search(raw_address) else 0
    pin_extracted = extract_postal_code(raw_address, country)
    street_number = extract_street_number(raw_address)
    city_extracted = extract_city(raw_address)

    cleaned = clean_base_text(raw_address)
    # Expand address abbreviations
    for pattern, replacement in ADDRESS_ABBR_MAP.items():
        cleaned = re.sub(pattern, replacement, cleaned)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned, pin_extracted, city_extracted, street_number, has_landmark


def apply_normalization_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply normalization to a copy of dataframe, producing derived columns."""
    df_norm = df.copy()

    has_name = "business_name" in df_norm.columns
    has_addr = "business_address" in df_norm.columns
    has_country = "country" in df_norm.columns

    names_norm = []
    legal_suffixes = []
    addrs_norm = []
    pins = []
    cities = []
    street_nums = []
    landmarks = []

    names = df_norm["business_name"].fillna("").astype(str).tolist() if has_name else [""] * len(df_norm)
    addrs = df_norm["business_address"].fillna("").astype(str).tolist() if has_addr else [""] * len(df_norm)
    countries = df_norm["country"].fillna("").astype(str).tolist() if has_country else [""] * len(df_norm)

    for i in range(len(df_norm)):
        n_norm, l_suffix = normalize_name(names[i])
        a_norm, pin, city, st_num, landmark = normalize_address(addrs[i], countries[i])

        names_norm.append(n_norm)
        legal_suffixes.append(l_suffix)
        addrs_norm.append(a_norm)
        pins.append(pin)
        cities.append(city)
        street_nums.append(st_num)
        landmarks.append(landmark)

    df_norm["name_norm"] = names_norm
    df_norm["legal_suffix"] = legal_suffixes
    df_norm["addr_norm"] = addrs_norm
    df_norm["pin_extracted"] = pins
    df_norm["city_extracted"] = cities
    df_norm["street_number"] = street_nums
    df_norm["has_landmark"] = landmarks

    return df_norm
