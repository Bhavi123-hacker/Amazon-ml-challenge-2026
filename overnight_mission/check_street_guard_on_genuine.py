import re
import sys
from rapidfuzz import fuzz

def extract_street_numbers(addr: str) -> set:
    if not addr:
        return set()
    # Normalize special symbols
    clean = re.sub(r'[#№°]', ' ', addr)
    # Remove 'no.', 'num.', 'n°', etc.
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

# Parse output/step1_audit_results.txt
with open("output/step1_audit_results.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Let's split into entity sections
sections = text.split("================================================================================")

genuine_pairs = []
false_pairs = []
ambig_pairs = []

for sec in sections:
    if not sec.strip():
        continue
    lines = [l.strip() for l in sec.strip().split("\n") if l.strip()]
    
    header = lines[0] if lines else ""
    country = "Unknown"
    if "France" in header:
        country = "France"
    elif "US" in header:
        country = "US"
    elif "India" in header:
        country = "India"
        
    s1_name, s1_addr = "", ""
    current_match = None
    
    for l in lines:
        if l.startswith("Source 1 :"):
            content = l.replace("Source 1 :", "").strip()
            parts = content.split("|", 1)
            s1_name = parts[0].strip()
            s1_addr = parts[1].strip() if len(parts) > 1 else ""
        elif l.startswith("* S2-") or l.startswith("* S3-"):
            parts = l.split(":", 1)
            cid = parts[0].replace("*", "").strip()
            c_content = parts[1].strip()
            c_parts = c_content.split("|", 1)
            c_name = c_parts[0].strip()
            c_addr = c_parts[1].strip() if len(c_parts) > 1 else ""
            current_match = {
                "cid": cid, "name": c_name, "addr": c_addr,
                "s1_name": s1_name, "s1_addr": s1_addr,
                "country": country, "verdict": "Unknown", "notes": ""
            }
        elif l.startswith("-> Verdict:"):
            v_text = l.replace("-> Verdict:", "").strip()
            if current_match:
                if "GENUINE" in v_text:
                    current_match["verdict"] = "GENUINE"
                    genuine_pairs.append(current_match)
                elif "FALSE" in v_text:
                    current_match["verdict"] = "FALSE"
                    false_pairs.append(current_match)
                elif "AMBIGUOUS" in v_text:
                    current_match["verdict"] = "AMBIGUOUS"
                    ambig_pairs.append(current_match)
                current_match["notes"] = v_text
                current_match = None

print(f"Parsed from step1_audit_results.txt:")
print(f"  Genuine matches: {len(genuine_pairs)}")
print(f"  False matches:   {len(false_pairs)}")
print(f"  Ambiguous:       {len(ambig_pairs)}")

# Now test the guard on all genuine matches!
print("\n--- TESTING GUARD ON ALL CONFIRMED GENUINE MATCHES ---")
false_rejections_street = []
false_rejections_tenant = []

for p in genuine_pairs:
    s_mismatch = is_street_number_mismatch(p["s1_addr"], p["addr"])
    t_collision = is_multi_tenant_collision(p["s1_name"], p["name"], p["s1_addr"], p["addr"])
    
    nums1 = extract_street_numbers(p["s1_addr"])
    nums2 = extract_street_numbers(p["addr"])
    
    if s_mismatch:
        false_rejections_street.append((p, nums1, nums2))
    if t_collision:
        false_rejections_tenant.append((p, nums1, nums2))

print(f"Total Genuine pairs tested: {len(genuine_pairs)}")
print(f"False Rejections by Street Number Guard: {len(false_rejections_street)}")
print(f"False Rejections by Multi-Tenant Guard:  {len(false_rejections_tenant)}")

if false_rejections_street:
    print("\nDETAILS OF FALSE REJECTIONS BY STREET NUMBER GUARD:")
    for p, n1, n2 in false_rejections_street:
        print(f"  [{p['country']}] S1: {p['s1_name']} | {p['s1_addr']} (Nums: {n1})")
        print(f"         C:  {p['name']} | {p['addr']} (Nums: {n2})")
        print(f"         Verdict Notes: {p['notes']}")
        print()

if false_rejections_tenant:
    print("\nDETAILS OF FALSE REJECTIONS BY MULTI-TENANT GUARD:")
    for p, n1, n2 in false_rejections_tenant:
        print(f"  [{p['country']}] S1: {p['s1_name']} vs C: {p['name']}")
        print(f"         S1 Addr: {p['s1_addr']}")
        print(f"         C  Addr: {p['addr']}")
        print()

# Now test the guard on all confirmed false matches!
print("\n--- TESTING GUARD ON ALL CONFIRMED FALSE MATCHES ---")
true_rejections_street = 0
true_rejections_tenant = 0
total_rejected_false = 0

for p in false_pairs:
    s_mismatch = is_street_number_mismatch(p["s1_addr"], p["addr"])
    t_collision = is_multi_tenant_collision(p["s1_name"], p["name"], p["s1_addr"], p["addr"])
    
    nums1 = extract_street_numbers(p["s1_addr"])
    nums2 = extract_street_numbers(p["addr"])
    
    if s_mismatch:
        true_rejections_street += 1
    if t_collision:
        true_rejections_tenant += 1
    if s_mismatch or t_collision:
        total_rejected_false += 1
        print(f"  [CORRECT REJECTION] [{p['country']}] S1: {p['s1_name'][:25]} vs C: {p['name'][:25]} | Reason: {'STREET_NUM ' if s_mismatch else ''}{'TENANT' if t_collision else ''}")

print(f"\nTotal False pairs tested: {len(false_pairs)}")
print(f"Correctly Caught by Street Number Guard: {true_rejections_street}")
print(f"Correctly Caught by Multi-Tenant Guard:  {true_rejections_tenant}")
print(f"Total False Pairs Caught: {total_rejected_false} / {len(false_pairs)} ({total_rejected_false/len(false_pairs)*100:.1f}%)")
