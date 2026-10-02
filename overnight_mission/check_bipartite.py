import collections

s2_to_s1 = collections.defaultdict(list)
with open('student_resource/dataset/train/train_ground_truth.tsv', 'r', encoding='utf-8') as f:
    next(f)
    for i, line in enumerate(f):
        # Check all lines

        parts = line.strip().split('\t')
        s1 = parts[0]
        if len(parts) > 1 and parts[1]:
            for s2 in parts[1].split(','):
                s2 = s2.strip()
                if s2:
                    s2_to_s1[s2].append(s1)

shared = sum(1 for s2, s1s in s2_to_s1.items() if len(s1s) > 1)
total_s2 = len(s2_to_s1)
print(f"Total S2 checked in 100k lines: {total_s2}")
print(f"S2 claimed by more than 1 S1: {shared} ({shared/max(1, total_s2)*100:.4f}%)")
if shared > 0:
    sample_shared = [(s2, s1s) for s2, s1s in s2_to_s1.items() if len(s1s) > 1][:5]
    print(f"Sample shared S2: {sample_shared}")
