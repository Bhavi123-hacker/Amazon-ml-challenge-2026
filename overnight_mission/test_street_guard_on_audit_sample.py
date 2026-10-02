"""Test Street Number & Guard D on all 30 audited entities from Step 1."""

import re
import pandas as pd
from rapidfuzz import fuzz

def extract_street_numbers(addr: str) -> set:
    if not addr:
        return set()
    clean = re.sub(r'[#№°]', ' ', addr)
    clean = re.sub(r'\b(no|n°|nº|num|number)\b\.?', ' ', clean, flags=re.IGNORECASE)
    tokens = re.findall(r'\b\d+[a-zA-Z]?\b', clean)
    nums = set()
    for tok in tokens:
        base_digits = re.sub(r'\D', '', tok).lstrip('0')
        if base_digits and len(base_digits) <= 6:
            nums.add(base_digits)
    return nums

def is_street_number_mismatch(addr1: str, addr2: str) -> bool:
    nums1 = extract_street_numbers(addr1)
    nums2 = extract_street_numbers(addr2)
    # If both have extracted street numbers, and there is ZERO overlap, it's a mismatch
    if nums1 and nums2 and nums1.isdisjoint(nums2):
        return True
    return False

def is_multi_tenant_collision(name1: str, name2: str, addr1: str, addr2: str) -> bool:
    ns = fuzz.token_sort_ratio(name1.lower(), name2.lower())
    asim = fuzz.token_sort_ratio(addr1.lower(), addr2.lower())
    return (ns < 45.0 and asim > 70.0)

def should_reject(name1, addr1, name2, addr2) -> tuple:
    if is_street_number_mismatch(addr1, addr2):
        return True, "STREET_NUM_MISMATCH"
    if is_multi_tenant_collision(name1, name2, addr1, addr2):
        return True, "MULTI_TENANT_COLLISION"
    return False, "ACCEPT"

# Load the audit results from Step 1
print("Testing guard rules on the 30 audited entities from Step 1...\n")

with open("output/step1_audit_results.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

current_s1 = None
current_country = None
cur_s1_name = ""
cur_s1_addr = ""

accepted_genuine = 0
rejected_genuine = 0
rejected_false = 0
accepted_false = 0

for line in lines:
    line = line.strip()
    if line.startswith("[France #") or line.startswith("[US #") or line.startswith("[India #"):
        current_s1 = line
    elif line.startswith("Source 1 :"):
        parts = line.replace("Source 1 :", "").split("|")
        cur_s1_name = parts[0].strip()
        cur_s1_addr = parts[1].strip() if len(parts) > 1 else ""
    elif line.startswith("* S2-") or line.startswith("* S3-"):
        parts = line.split(":", 1)
        cid = parts[0].split()[1]
        c_text = parts[1].split("|")
        c_name = c_text[0].strip()
        c_addr = c_text[1].strip() if len(c_text) > 1 else ""
        
        rej, reason = should_reject(cur_s1_name, cur_s1_addr, c_name, c_addr)
        
        nums1 = extract_street_numbers(cur_s1_addr)
        nums2 = extract_street_numbers(c_addr)
        ns = fuzz.token_sort_ratio(cur_s1_name.lower(), c_name.lower())
        
        status = "REJECTED" if rej else "ACCEPTED"
        print(f"[{status:8s}] {cid:12s} Reason={reason:22s} | S1={cur_s1_name[:20]:20s} ({nums1}) vs C={c_name[:20]:20s} ({nums2}) [ns={ns:2.0f}]")

