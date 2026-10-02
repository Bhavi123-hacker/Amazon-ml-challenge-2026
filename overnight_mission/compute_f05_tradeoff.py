"""Step 2: Explicit F0.5 Trade-off Analysis between Base tau=0.75 and Guard D."""

def calc_f05(p, r):
    return (1.25 * p * r) / (0.25 * p + r)

def main():
    print("=" * 80)
    print("STEP 2: EXPLICIT F0.5 TRADE-OFF CALCULATION (FRANCE CONFIGURATIONS)")
    print("=" * 80)

    configs = [
        ("Old Filter (street_name_sim >= 0.70)", 0.88, 0.49, "High prec, but wiped 51% recall (26.6% singletons)"),
        ("Base tau=0.70 (No Guard)", 0.60, 0.98, "High recall, but flooded with multi-tenant collisions"),
        ("Base tau=0.75 (No Guard)", 0.65, 0.97, "High recall, precision capped by multi-tenant collisions"),
        ("Base tau=0.80 (No Guard)", 0.75, 0.92, "Moderate precision improvement, still has collisions"),
        ("Guard D: Conservative (Strict Prec)", 0.89, 0.92, "Measured strict prec 88-92%, natural singletons (1.9%)"),
        ("Guard D: Expected (Balanced)", 0.91, 0.94, "Strict/effective midpoint, strong cluster preservation"),
        ("Guard D: Effective Prec (Best Est.)", 0.96, 0.94, "Measured effective prec 94-98%, natural singletons (1.9%)"),
    ]

    print(f"{'Configuration':<38} | {'Precision':<10} | {'Recall':<10} | {'Estimated F0.5':<15} | {'Notes'}")
    print("-" * 110)
    for name, p, r, note in configs:
        f05 = calc_f05(p, r)
        print(f"{name:<38} | {p*100:>8.1f}% | {r*100:>8.1f}% | {f05:>14.4f}  | {note}")

    print("=" * 80)
    print("CONCLUSION:")
    print("Guard D's +25-30% precision boost decisively outweighs the ~3-5% recall cost.")
    print("Under F0.5 (which weights precision 2x over recall), Guard D moves France from ~0.65 to ~0.90-0.95.")
    print("=" * 80)

if __name__ == "__main__":
    main()
