"""Evaluation module for Business Entity Resolution Challenge.

Calculates Macro-averaged F_0.5 score:
    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

Singletons:
- If ground truth is empty (no matches):
    - Predicted empty -> 1.0
    - Predicted non-empty -> 0.0
- If ground truth is non-empty:
    - Predicted empty or TP == 0 -> 0.0
    - Otherwise -> standard F_0.5 formula
"""

import argparse
import sys
from typing import Dict, List, Set, Tuple, Union
import pandas as pd


def compute_entity_f05(
    true_matches: Union[Set[str], List[str]],
    pred_matches: Union[Set[str], List[str]],
) -> float:
    """Compute F_0.5 score for a single Source 1 entity."""
    true_set = set(true_matches) if not isinstance(true_matches, set) else true_matches
    pred_set = set(pred_matches) if not isinstance(pred_matches, set) else pred_matches

    # Filter out empty or whitespace tokens
    true_set = {x.strip() for x in true_set if x and x.strip()}
    pred_set = {x.strip() for x in pred_set if x and x.strip()}

    # Case 1: True singleton (no matches in ground truth)
    if len(true_set) == 0:
        return 1.0 if len(pred_set) == 0 else 0.0

    # Case 2: Non-singleton ground truth, but predicted empty
    if len(pred_set) == 0:
        return 0.0

    # Case 3: Overlap calculation
    tp = len(true_set & pred_set)
    if tp == 0:
        return 0.0

    precision = tp / len(pred_set)
    recall = tp / len(true_set)

    denom = 0.25 * precision + recall
    if denom == 0.0:
        return 0.0

    return (1.25 * precision * recall) / denom


def compute_macro_f05(
    ground_truth: Dict[str, Union[Set[str], List[str]]],
    predictions: Dict[str, Union[Set[str], List[str]]],
) -> Tuple[float, float, float]:
    """Compute Macro-averaged F_0.5, macro precision, and macro recall.

    Returns:
        (macro_f05, macro_precision, macro_recall)
    """
    total_f05 = 0.0
    total_prec = 0.0
    total_rec = 0.0
    n = len(ground_truth)
    if n == 0:
        return 0.0, 0.0, 0.0

    for s1_id, true_matches in ground_truth.items():
        true_set = {x.strip() for x in true_matches if x and x.strip()}
        pred_matches = predictions.get(s1_id, set())
        pred_set = {x.strip() for x in pred_matches if x and x.strip()}

        f05 = compute_entity_f05(true_set, pred_set)
        total_f05 += f05

        if len(true_set) == 0:
            if len(pred_set) == 0:
                total_prec += 1.0
                total_rec += 1.0
            else:
                total_prec += 0.0
                total_rec += 1.0
        else:
            if len(pred_set) == 0:
                total_prec += 0.0
                total_rec += 0.0
            else:
                tp = len(true_set & pred_set)
                total_prec += tp / len(pred_set)
                total_rec += tp / len(true_set)

    return total_f05 / n, total_prec / n, total_rec / n


def parse_tsv_matches(path: str, id_col: str, match_col: str) -> Dict[str, Set[str]]:
    """Parse TSV file with comma-separated match lists."""
    result = {}
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    for _, row in df.iterrows():
        s1 = row[id_col].strip()
        m_raw = row[match_col].strip()
        matches = {m.strip() for m in m_raw.split(",") if m.strip()} if m_raw else set()
        result[s1] = matches
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate Macro F_0.5 for Business Entity Resolution.")
    parser.add_argument("--ground-truth", required=True, help="Path to ground truth TSV file")
    parser.add_argument("--predictions", required=True, help="Path to predictions TSV file")
    args = parser.parse_args()

    gt = parse_tsv_matches(args.ground_truth, "source1_entity_id", "matched_entity_ids")
    preds = parse_tsv_matches(args.predictions, "source1_entity_id", "matched_entity_ids")

    f05, prec, rec = compute_macro_f05(gt, preds)
    print(f"Entities Evaluated: {len(gt):,}")
    print(f"Macro F_0.5 Score : {f05:.5f}")
    print(f"Macro Precision   : {prec:.5f}")
    print(f"Macro Recall      : {rec:.5f}")


if __name__ == "__main__":
    main()
