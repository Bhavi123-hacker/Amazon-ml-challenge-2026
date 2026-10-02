import re

clean_re1 = re.compile(r'[#№°]')
clean_re2 = re.compile(r'\b(no|n°|nº|num|number)\b\.?', re.IGNORECASE)
clean_re3 = re.compile(r'(\d+)[-/](\d+)')
# OCR typo fix: e.g. 4O -> 40 or 2O -> 20 when O is surrounded by digits or preceded by digits at word boundary
ocr_re = re.compile(r'(?<=\d)[oO]\b|\b[oO](?=\d)')
tok_re = re.compile(r'\b\d+[a-zA-Z]?\b')
non_digit_re = re.compile(r'\D')

def extract_street_numbers(addr: str) -> set:
    if not addr or not isinstance(addr, str):
        return set()
    c = ocr_re.sub('0', addr)
    c = clean_re1.sub(' ', c)
    c = clean_re2.sub(' ', c)
    c = clean_re3.sub(r'\1 \2', c)
    tokens = tok_re.findall(c)
    nums = set()
    for tok in tokens:
        bd = non_digit_re.sub('', tok).lstrip('0')
        if bd and len(bd) <= 6:
            nums.add(int(bd))
    return nums

def is_street_number_mismatch(addr1: str, addr2: str) -> bool:
    nums1 = extract_street_numbers(addr1)
    nums2 = extract_street_numbers(addr2)
    # Only a mismatch if BOTH have numbers, and sets are strictly disjoint
    if nums1 and nums2 and nums1.isdisjoint(nums2):
        return True
    return False

# Test Suite: 28 Real Confirmed Genuine Pairs from tonight's audit + known edge cases
genuine_test_cases = [
    # Zero-padding
    ("Memphis, 3365 Winchester Place, TN", "003365 WINCHESTER PL, TN, MEMPHIS", "US Zero-padding 3365 vs 003365"),
    ("204 Kingspoint Drive, Sherman, IL", "00204 Kingspoint Drive, Sherman, Illinois", "US Zero-padding 204 vs 00204"),
    ("1322 Broadway, Unit Apartment 1, Buffalo, NY", "001322 BROADWAY, BUFFALO, NY", "US Zero-padding 1322 vs 001322"),
    ("101, Manbhavan Nagar Kanadia Road, Indore", "0101, Manbhavan Nagar Kanadia Road, Indore", "India Zero-padding 101 vs 0101"),
    
    # Symbols and prefixes (#, N°, No, Nº)
    ("15 Rue Judaique, Bordeaux", "15 RUE JUDAIQUE, BORDEAUX", "France Exact 15 vs 15"),
    ("237 Rue de Pessac, Bordeaux", "#237 R De Pessac, Bordeaux", "France Prefix #237 vs 237"),
    ("237 Rue de Pessac, Bordeaux", "#237 R. De Pessac, Bordeaux", "France Prefix #237 vs 237"),
    ("253 Rue de Gand, Tourcoing", "253 R. De Gand, Tourcoing", "France Exact 253 vs 253"),
    ("26 Rue Albert Sauvage, Dunkerque", "N° 26 RUE ALBERT SAUVAGE, DUNKERQUE", "France Prefix N° 26 vs 26"),
    ("26 Rue Albert Sauvage, Dunkerque", "N° 26 RUE ALBRET SAUVAGE, DUNKERQUE", "France Prefix N° 26 vs 26"),
    ("9 Rue du Marche, Saint-Nazaire", "#9 R Du Marche, Saint-nazaire", "France Prefix #9 vs 9"),
    ("23 Rue de Misericorde, Nantes", "No 23 Rue De Misericrode, Nantes", "France Prefix No 23 vs 23"),
    ("19 Rue de Wazemmes, Lille", "NO 19 R. DE WAZEMMES, LILLE", "France Prefix NO 19 vs 19"),
    ("1929 Robertson Avenue, Albuquerque, NM", "#1929 ROBERTSON AVE, ALBUQUERQUE, NM", "US Prefix #1929 vs 1929"),
    ("Shop No. 202, D. C. Bhawan, Patna", "Shop No. 202, D. C. Bhawan, Patna", "India Prefix Shop No. 202"),
    ("Flat No. 205, Coronation Appartment 28, Park Road", "FLAT NO. 205, LUCKNOW, PARK ROAD", "India Prefix Flat No. 205"),
    ("Rz-1/91, Gali No. 2, East Sagar Pur", "Rz-#1/91, G/f, Gali No. 2, East Sagar Pur", "India Prefix Rz-#1/91 vs Rz-1/91"),

    # Unit / Sub-building suffixes & ranges
    ("982 Electric Avenue, Bigfork, MT", "BIGFORK, 982-A ELECTRIC AVE, MT", "US Unit suffix 982 vs 982-A"),
    ("B-65/1 Manoj Bhawangautam Nagar, Delhi", "B-65/1 MANOJ BHAWANGAUTAM NAGAR, NEW DELHI", "India Range/Sub-number B-65/1"),
    ("30/954, 3Rd Floor, Gali No. 30 Dda Flats", "A-30/954, 3Rd Floor, Gali No. 30 Dda Flats", "India Flat/Gali 30/954"),
    ("Unit No Floor 2308 Plan S, Plot D 108/1", "UNIT NO FLOOR 2308 PLAN S, PLOT D 108/1", "India Multi-part 2308, 108, 1"),
    ("526 California Street, Waterloo, IA", "526 CALIFORNIA STREET, WATERLOOO, IA", "US Exact 526 vs 526"),
    ("725 Pine Street, Fl 3rd, Baraboo, WI", "725 PINE STREET, BARABOO, WI", "US Exact 725 with Fl 3rd"),
    ("1828 Fauver Avenue, Dayton, OH", "1828 FAUVER AVE, DAYTON, OH", "US Exact 1828 vs 1828"),

    # Missing numbers in one source (should NOT reject)
    ("204 Kingspoint Drive, Sherman, IL", "KINGSPOINT DRIVE, SHERMAN, IL", "US Missing number in S2"),
    ("8 Rue des Pommiers, Pornic", "R. DES POMMIERS, PORNIC", "France Missing number in S2"),
    ("C/O Sudesh Kalupur West, Ramnagar", "#41 C/O SUDESH KALUPUR WEST, RAMNAGAR", "India Missing number in S1"),
    ("Unit No Floor 2308 Plan S, Plot D 108/1", "Maharashtra, THANE, UNIT NO FLOOR 2308 PLAN S", "India Partial numbers shared (2308)"),
    
    # OCR letter O in digits
    ("1500 7th Street, Unit APT 4O, Sacramento, CA", "1500 7th Street, APT 40, Sacramento, CA", "US OCR O in 4O vs 40")
]

print("="*80)
print("TESTING STREET NUMBER CONSISTENCY GUARD ON 29 REAL CONFIRMED GENUINE CASES")
print("="*80)

passed_genuine = 0
for idx, (a1, a2, desc) in enumerate(genuine_test_cases, 1):
    mismatch = is_street_number_mismatch(a1, a2)
    n1 = extract_street_numbers(a1)
    n2 = extract_street_numbers(a2)
    status = "FAIL (REJECTED GENUINE!)" if mismatch else "PASS (ACCEPTED)"
    if not mismatch:
        passed_genuine += 1
    print(f"[{status:24s}] #{idx:02d} {desc}")
    print(f"   Addr 1: {a1} -> {n1}")
    print(f"   Addr 2: {a2} -> {n2}")

print(f"\nGENUINE MATCH ACCURACY: {passed_genuine} / {len(genuine_test_cases)} ({passed_genuine/len(genuine_test_cases)*100:.1f}%)")

# Now Test on Known False Distractors (Must be REJECTED!)
distractor_cases = [
    ("Memphis, 3365 Winchester Place, TN", "3378-3380 WINCHESTER PL, MEMPHIS, TN", "US Distractor: 3365 vs 3378-3380"),
    ("OH, Dayton, 1828 Fauver Avenue", "1831 Fauver Ave, Dayton, Ohio", "US Distractor: 1828 vs 1831"),
    ("1322 Broadway, Unit Apartment 1, Buffalo, NY", "1343 BROADWAY, BUFFALO, NY", "US Distractor: 1322 vs 1343"),
    ("3530 Grant Avenue, Groves, TX", "03533 Grant Ave, Groves CTY, Texas", "US Distractor: 3530 vs 3533"),
    ("1929 Robertson Avenue, Albuquerque, NM", "1930 ROBERTSON AVE, ALBUQUERQUE, NM", "US Distractor: 1929 vs 1930"),
    ("452 Isleta Boulevard, Albuquerque, NM", "459 Isleta Blvd, SOUTH VALLEY, NM", "US Distractor: 452 vs 459"),
    ("452 Isleta Boulevard, Albuquerque, NM", "52 ISLETA BLVD, SOUTH VALLEY, NM", "US Distractor: 452 vs 52"),
    ("526 California Street, Waterloo, IA", "533 CALIFORNIA ST, WATERLOO, IA", "US Distractor: 526 vs 533"),
    ("725 Pine Street, Fl 3rd, Village Of West Baraboo, WI", "Wisconsin, Baraboo, Fl 3rd, 738 Pine Street", "US Distractor: 725 vs 738"),
    ("204 Kingspoint Drive, Sherman, IL", "09 Kingspoint Dr, Sherman, Illinois", "US Distractor: 204 vs 9"),
    ("253 Rue de Gand, Tourcoing", "# 56 R. DE BAPAUME, TOURCOING", "France Distractor: 253 vs 56"),
    ("101, Manbhavan Nagar Kanadia Road, Indore", "108, Manbhavan Nagar Kanadia Road, Indore", "India Distractor: 101 vs 108")
]

print("\n" + "="*80)
print("TESTING STREET NUMBER CONSISTENCY GUARD ON 12 KNOWN FALSE DISTRACTORS")
print("="*80)

rejected_distractors = 0
for idx, (a1, a2, desc) in enumerate(distractor_cases, 1):
    mismatch = is_street_number_mismatch(a1, a2)
    n1 = extract_street_numbers(a1)
    n2 = extract_street_numbers(a2)
    status = "PASS (REJECTED DISTRACTOR)" if mismatch else "FAIL (ACCEPTED DISTRACTOR!)"
    if mismatch:
        rejected_distractors += 1
    print(f"[{status:26s}] #{idx:02d} {desc}")
    print(f"   Addr 1: {a1} -> {n1}")
    print(f"   Addr 2: {a2} -> {n2}")

print(f"\nDISTRACTOR REJECTION RATE: {rejected_distractors} / {len(distractor_cases)} ({rejected_distractors/len(distractor_cases)*100:.1f}%)")
