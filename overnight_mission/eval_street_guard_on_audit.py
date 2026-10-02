import re
import sys
from rapidfuzz import fuzz

def extract_street_numbers(addr: str) -> set:
    if not addr:
        return set()
    clean = re.sub(r'[#№°]', ' ', addr)
    clean = re.sub(r'\b(no|n°|nº|num|number)\b\.?', ' ', clean, flags=re.IGNORECASE)
    # Split on hyphens/slashes between digits if any (e.g. 3378-3380 or 108/1)
    clean = re.sub(r'(\d+)[-/](\d+)', r'\1 \2', clean)
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

# Read step1_audit_results.txt line by line
with open("output/step1_audit_results.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

log_lines = []
log_lines.append("="*80)
log_lines.append("TESTING STREET NUMBER & GUARD D ON ALL 101 PAIRS FROM 30 AUDITED ENTITIES")
log_lines.append("="*80)

cur_entity = None
cur_s1_name = ""
cur_s1_addr = ""
cur_country = ""

total_pairs = 0
rejected_count = 0
accepted_count = 0

for line in lines:
    line_s = line.strip()
    if line_s.startswith("==================== COUNTRY:"):
        cur_country = line_s.split(":")[1].replace("=", "").strip()
        log_lines.append(f"\n>>> COUNTRY: {cur_country} <<<\n")
    elif line_s.startswith("[France #") or line_s.startswith("[US #") or line_s.startswith("[India #"):
        cur_entity = line_s
        log_lines.append(f"\n{cur_entity}")
    elif line_s.startswith("Source 1 :"):
        content = line_s.replace("Source 1 :", "").strip()
        parts = content.split("|", 1)
        cur_s1_name = parts[0].strip()
        cur_s1_addr = parts[1].strip() if len(parts) > 1 else ""
        s1_nums = extract_street_numbers(cur_s1_addr)
        log_lines.append(f"  S1: {cur_s1_name} | {cur_s1_addr} (Nums: {s1_nums})")
    elif line_s.startswith("* S2-") or line_s.startswith("* S3-"):
        total_pairs += 1
        parts = line_s.split(":", 1)
        cid = parts[0].split()[1]
        c_content = parts[1].strip()
        c_parts = c_content.split("|", 1)
        c_name = c_parts[0].strip()
        c_addr = c_parts[1].strip() if len(c_parts) > 1 else ""
        c_nums = extract_street_numbers(c_addr)
        
        rej, reason = should_reject(cur_s1_name, cur_s1_addr, c_name, c_addr)
        ns = fuzz.token_sort_ratio(cur_s1_name.lower(), c_name.lower())
        
        if rej:
            rejected_count += 1
            status = "REJECTED"
        else:
            accepted_count += 1
            status = "ACCEPTED"
            
        log_lines.append(f"    [{status:8s}] {cid:12s} | {reason:22s} | Nums: {c_nums} | ns={ns:2.0f} | {c_name} | {c_addr}")

log_lines.append("\n" + "="*80)
log_lines.append(f"TOTAL AUDITED PAIRS: {total_pairs}")
log_lines.append(f"ACCEPTED:            {accepted_count} ({accepted_count/total_pairs*100:.1f}%)")
log_lines.append(f"REJECTED:            {rejected_count} ({rejected_count/total_pairs*100:.1f}%)")
log_lines.append("="*80)

output_text = "\n".join(log_lines)
with open("output/street_guard_audit_evaluation.txt", "w", encoding="utf-8") as f:
    f.write(output_text)

print(f"Evaluation complete. Total pairs: {total_pairs}, Accepted: {accepted_count}, Rejected: {rejected_count}")
print("Saved details to output/street_guard_audit_evaluation.txt")
