"""
Milestone 2 Track B: Rules-Based SBDH Span & Category Extraction Baseline.

Features:
- Transparent, manually specified regular expression & keyword patterns for all 15 SBDH categories.
- Full support for multi-label assignment on single or overlapping spans.
- Completely independent of the reviewed test set (patterns derived from domain clinical taxonomies
  and training split patterns, frozen prior to evaluation).
- Evaluates against synth_sbdh_test_reviewed.csv (1,732 notes).
- Reports:
  1. Per-category Precision, Recall, F1 (note-level).
  2. Aggregate Micro, Macro, and Weighted Classification metrics.
  3. Span-level Exact Matching Precision, Recall, F1.
  4. Span-level Relaxed / Overlap Matching Precision, Recall, F1.
- Generates: reports/sbdh_rules_metrics.json
"""
import os
import sys
import re
import json
from typing import Dict, Any, List, Tuple, Set
import pandas as pd
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loaders import load_synth_sbdh
from src.span_alignment import align_spans_in_note, parse_delimited_field
from src.utils import NpEncoder, save_json


# Frozen, manually specified domain-knowledge rules for 15 SBDH categories.
# Built using clinical screening guidelines (PRAPARE, AHC-HRSN, ICD-10 Z-codes) and training patterns.
SBDH_RULES_TAXONOMY: Dict[str, List[Tuple[str, List[str]]]] = {
    "housing_rules": [
        # Pattern, List of SBDH Categories assigned to the match
        (r"\b(currently\s+)?homeless(ness)?\b", ["housing insecurity"]),
        (r"\b(living|staying)\s+(in\s+a\s+shelter|on\s+the\s+streets?)\b", ["housing insecurity"]),
        (r"\b(facing\s+)?evict(ion|ed)?\b", ["housing insecurity"]),
        (r"\bforeclos(ure|ed)\s+on\s+(his|her|their)?\s*home\b", ["housing insecurity", "financial insecurity"]),
        (r"\b(unstable|insecure)\s+housing\b", ["housing insecurity"]),
        (r"\bhousing\s+insecurity\b", ["housing insecurity"]),
        (r"\bcouch[\s-]surfing\b", ["housing insecurity"]),
        (r"\bno\s+permanent\s+address\b", ["housing insecurity"]),
    ],
    "food_rules": [
        (r"\bfood\s+insecurity\b", ["food insecurity"]),
        (r"\b(skipping|skipped|skips)\s+meals?\b", ["food insecurity"]),
        (r"\b(frequently\s+)?visits?\s+(local\s+)?food\s+bank\b", ["food insecurity"]),
        (r"\b(unable|cannot|struggling)\s+to\s+afford\s+groceries\b", ["food insecurity", "financial insecurity"]),
        (r"\bdifficulty\s+affording\s+groceries\b", ["food insecurity", "financial insecurity"]),
        (r"\bfood\s+pantry\b", ["food insecurity"]),
        (r"\blacks?\s+sufficient\s+(healthy\s+)?food\b", ["food insecurity"]),
    ],
    "financial_rules": [
        (r"\b(struggling|difficulty)\s+financially\b", ["financial insecurity"]),
        (r"\bfinancial\s+(difficulties|stress|instability|constraints|hardship|struggles?)\b", ["financial insecurity"]),
        (r"\bcannot\s+afford\s+to\s+continue\s+physical\s+therapy\b", ["financial insecurity", "barriers to care"]),
        (r"\b(unable|cannot)\s+to\s+afford\s+medications?\b", ["financial insecurity", "barriers to care"]),
        (r"\bliving\s+paycheck\s+to\s+paycheck\b", ["financial insecurity"]),
        (r"\bdeclared\s+bankruptcy\b", ["financial insecurity"]),
        (r"\bheavy\s+debt\b", ["financial insecurity"]),
        (r"\bextreme\s+poverty\b", ["financial insecurity"]),
    ],
    "barriers_rules": [
        (r"\b(lack|no|lacks?)\s+of\s+transportation\b", ["barriers to care"]),
        (r"\brelies?\s+on\s+public\s+transportation\b", ["barriers to care"]),
        (r"\b(often\s+)?skips?\s+(medication\s+)?doses?\b", ["barriers to care"]),
        (r"\bmissed\s+appointments?\s+due\s+to\s+transportation\b", ["barriers to care"]),
        (r"\bdelays?\s+seeking\s+care\b", ["barriers to care"]),
        (r"\b(no|uninsured|lacks?)\s+(health\s+)?insurance\b", ["barriers to care"]),
        (r"\bbarriers?\s+to\s+care\b", ["barriers to care"]),
    ],
    "job_rules": [
        (r"\b(recent\s+)?job\s+loss\b", ["job insecurity"]),
        (r"\blost\s+(his|her|their)?\s*job\b", ["job insecurity"]),
        (r"\blosing\s+(his|her|their)?\s*job\b", ["job insecurity"]),
        (r"\bunemployed\b", ["job insecurity"]),
        (r"\bunemployment\b", ["job insecurity"]),
        (r"\blaid\s+off\b", ["job insecurity"]),
        (r"\bterminated\s+from\s+employment\b", ["job insecurity"]),
        (r"\bseeking\s+employment\b", ["job insecurity"]),
    ],
    "pain_rules": [
        (r"\bchronic\s+back\s+pain\b", ["pain"]),
        (r"\bchronic\s+pain(\s+due\s+to\s+\w+)?\b", ["pain"]),
        (r"\bsevere\s+back\s+pain\b", ["pain"]),
        (r"\bsevere\s+(abdominal\s+)?pain\b", ["pain"]),
        (r"\babdominal\s+pain\b", ["pain"]),
        (r"\bsevere\s+migraines?\b", ["pain"]),
        (r"\bfrequent\s+headaches?\b", ["pain"]),
        (r"\bjoint\s+pain\b", ["pain"]),
        (r"\bneuropathic\s+pain\b", ["pain"]),
        (r"\bpain\s+management\b", ["pain"]),
        (r"\bexperienc(ing|es)\s+pain\b", ["pain"]),
    ],
    "substance_rules": [
        (r"\bdrinking\s+heavily\b", ["substance abuse"]),
        (r"\balcohol\s+addiction\b", ["substance abuse"]),
        (r"\bhistory\s+of\s+alcohol\s+abuse\b", ["substance abuse"]),
        (r"\bstruggling\s+with\s+alcohol\s+addiction\b", ["substance abuse"]),
        (r"\brecovering\s+alcoholic\b", ["substance abuse"]),
        (r"\bheavy\s+alcohol\s+use\b", ["substance abuse"]),
        (r"\bopioid\s+usage\b", ["substance abuse"]),
        (r"\billicit\s+drug\s+use\b", ["substance abuse"]),
        (r"\bsubstance\s+abuse\b", ["substance abuse"]),
    ],
    "psychiatric_rules": [
        (r"\bptsd\b", ["psychiatric symptoms or disorders"]),
        (r"\bpost[\s-]traumatic\s+stress\s+disorder\b", ["psychiatric symptoms or disorders"]),
        (r"\bsevere\s+depression\b", ["psychiatric symptoms or disorders"]),
        (r"\bfeeling\s+depressed\b", ["psychiatric symptoms or disorders"]),
        (r"\bdepression\b", ["psychiatric symptoms or disorders"]),
        (r"\bsevere\s+anxiety\b", ["psychiatric symptoms or disorders"]),
        (r"\banxiety\b", ["psychiatric symptoms or disorders"]),
        (r"\bpanic\s+attacks?\b", ["psychiatric symptoms or disorders"]),
        (r"\bbipolar\s+disorder\b", ["psychiatric symptoms or disorders"]),
        (r"\bschizophrenia\b", ["psychiatric symptoms or disorders"]),
    ],
    "social_isolation_rules": [
        (r"\blives\s+alone\b", ["social isolation", "physical isolation"]),
        (r"\bliving\s+alone\b", ["social isolation", "physical isolation"]),
        (r"\bfeeling\s+(socially\s+)?isolated\b", ["social isolation"]),
        (r"\bfeeling\s+lonely\b", ["social isolation"]),
        (r"\bloneliness\b", ["social isolation"]),
        (r"\bhas\s+no\s+(close\s+)?(friends|family)\b", ["social isolation"]),
        (r"\blacks?\s+social\s+support\b", ["social isolation"]),
    ],
    "physical_isolation_rules": [
        (r"\bunable\s+to\s+leave\s+(his|her|their)?\s*(house|home)\b", ["physical isolation"]),
        (r"\brarely\s+leaves?\s+(the\s+)?house\b", ["physical isolation"]),
        (r"\bhomebound\b", ["physical isolation"]),
        (r"\bhousebound\b", ["physical isolation"]),
        (r"\bgeographically\s+isolated\b", ["physical isolation"]),
    ],
    "disability_rules": [
        (r"\bwheelchair[\s-]bound\b", ["patient disability"]),
        (r"\buses?\s+a\s+wheelchair\b", ["patient disability"]),
        (r"\bin\s+a\s+wheelchair\b", ["patient disability"]),
        (r"\bphysical\s+disability\b", ["patient disability"]),
        (r"\bdisabled\b", ["patient disability"]),
        (r"\bpermanent\s+impairment\b", ["patient disability"]),
        (r"\bamputee\b", ["patient disability"]),
        (r"\bparalyzed\b", ["patient disability"]),
    ],
    "loss_rules": [
        (r"\brecently\s+divorced\b", ["loss of relationship"]),
        (r"\brecent\s+divorce\b", ["loss of relationship"]),
        (r"\bdivorce(d)?\b", ["loss of relationship"]),
        (r"\b(husband|wife|spouse|partner)\s+passed\s+away\b", ["loss of relationship"]),
        (r"\bdeath\s+of\s+(his|her|their)?\s*(husband|wife|spouse|partner)\b", ["loss of relationship"]),
        (r"\bwidow(ed)?\b", ["loss of relationship"]),
    ],
    "violence_rules": [
        (r"\bvictim\s+of\s+domestic\s+violence\b", ["violence"]),
        (r"\bdomestic\s+violence\b", ["violence"]),
        (r"\bvictim\s+of\s+physical\s+abuse\b", ["violence"]),
        (r"\bphysical\s+abuse\b", ["violence"]),
        (r"\bphysical\s+altercation\b", ["violence"]),
        (r"\bpartner\s+violence\b", ["violence"]),
        (r"\brestraining\s+order\b", ["violence"]),
    ],
    "legal_rules": [
        (r"\bpending\s+court\s+case\b", ["legal problems"]),
        (r"\bupcoming\s+court\s+date\b", ["legal problems"]),
        (r"\breleased\s+from\s+prison\b", ["legal problems"]),
        (r"\blegal\s+(issues|problems?|trouble)\b", ["legal problems"]),
        (r"\bseeking\s+legal\s+help\b", ["legal problems"]),
        (r"\barrested\b", ["legal problems"]),
        (r"\bin\s+jail\b", ["legal problems"]),
    ],
    "transitions_rules": [
        (r"\bdischarged\s+from\s+the\s+hospital\b", ["transitions of care"]),
        (r"\brecently\s+discharged(\s+from\s+the\s+hospital)?\b", ["transitions of care"]),
        (r"\badmitted\s+to\s+the\s+hospital\b", ["transitions of care"]),
        (r"\bhospital\s+admission\b", ["transitions of care"]),
        (r"\brecently\s+moved\s+from\s+a\s+nursing\s+home\b", ["transitions of care"]),
        (r"\btransferred\s+to\s+rehab\b", ["transitions of care"]),
    ],
}


def extract_spans_with_rules(text: str) -> List[Dict[str, Any]]:
    """
    Apply frozen rules against text note.
    Returns list of predicted span dicts with start_char, end_char, categories, and matched_text.
    Supports multiple SBDH labels on a single span.
    """
    predicted_spans: List[Dict[str, Any]] = []

    for rule_group, patterns in SBDH_RULES_TAXONOMY.items():
        for pat_str, cats in patterns:
            pattern = re.compile(pat_str, re.IGNORECASE)
            for match in pattern.finditer(text):
                start = match.start()
                end = match.end()
                matched_snippet = match.group(0)

                # Check if this exact interval already exists in predicted_spans
                existing = next((p for p in predicted_spans if p["start_char"] == start and p["end_char"] == end), None)
                if existing:
                    for c in cats:
                        if c not in existing["categories"]:
                            existing["categories"].append(c)
                else:
                    predicted_spans.append({
                        "start_char": start,
                        "end_char": end,
                        "span_text": matched_snippet,
                        "categories": list(cats),
                    })

    # Sort by start_char
    predicted_spans.sort(key=lambda x: (x["start_char"], x["end_char"]))
    return predicted_spans


def match_spans_exact(
    pred_tuples: Set[Tuple[int, int, str]],
    gold_tuples: Set[Tuple[int, int, str]],
) -> Tuple[int, int, int]:
    """
    Exact span matching: start_char, end_char, and category must match exactly.
    Returns: (tp, fp, fn)
    """
    tp = len(pred_tuples & gold_tuples)
    fp = len(pred_tuples - gold_tuples)
    fn = len(gold_tuples - pred_tuples)
    assert tp + fp == len(pred_tuples), f"Exact match math failure: {tp} + {fp} != {len(pred_tuples)}"
    assert tp + fn == len(gold_tuples), f"Exact match math failure: {tp} + {fn} != {len(gold_tuples)}"
    return tp, fp, fn


def find_maximum_bipartite_matching(
    pred_list: List[Tuple[int, int, str]],
    gold_list: List[Tuple[int, int, str]],
) -> Set[Tuple[int, int]]:
    """
    Compute maximum cardinality bipartite matching between predictions and gold spans
    under category agreement and non-empty character overlap.
    Uses Kuhn's augmenting path algorithm (O(V * E)), guaranteeing strict prediction
    order-invariance and global maximum 1-to-1 matching.
    Returns set of matched (pred_idx, gold_idx) tuples.
    """
    n_preds = len(pred_list)
    n_golds = len(gold_list)

    # Build bipartite adjacency list
    adj: List[List[int]] = [[] for _ in range(n_preds)]
    for u, p in enumerate(pred_list):
        p_s, p_e, p_cat = p
        for v, g in enumerate(gold_list):
            g_s, g_e, g_cat = g
            if p_cat == g_cat and max(p_s, g_s) < min(p_e, g_e):
                adj[u].append(v)

    match_to_pred = [-1] * n_golds

    def dfs(u_idx: int, visited: List[bool]) -> bool:
        for v_idx in adj[u_idx]:
            if not visited[v_idx]:
                visited[v_idx] = True
                if match_to_pred[v_idx] < 0 or dfs(match_to_pred[v_idx], visited):
                    match_to_pred[v_idx] = u_idx
                    return True
        return False

    for u in range(n_preds):
        visited = [False] * n_golds
        dfs(u, visited)

    return {(u_idx, v_idx) for v_idx, u_idx in enumerate(match_to_pred) if u_idx >= 0}


def match_spans_relaxed(
    pred_tuples: Set[Tuple[int, int, str]],
    gold_tuples: Set[Tuple[int, int, str]],
) -> Tuple[int, int, int, Set[Tuple[int, int, str]]]:
    """
    Relaxed span matching: overlapping character interval (max(p_s, g_s) < min(p_e, g_e))
    with matching SBDH category. Solved via maximum cardinality bipartite matching (augmenting paths),
    guaranteeing strict order-invariance, global maximum 1-to-1 matches, and prevention of double-counting.
    Returns: (tp, fp, fn, matched_gold)
    """
    sorted_preds = sorted(list(pred_tuples), key=lambda x: (x[0], x[1], x[2]))
    sorted_golds = sorted(list(gold_tuples), key=lambda x: (x[0], x[1], x[2]))

    matched_pairs = find_maximum_bipartite_matching(sorted_preds, sorted_golds)
    tp = len(matched_pairs)
    fp = len(pred_tuples) - tp
    fn = len(gold_tuples) - tp

    matched_gold = {sorted_golds[v] for (_, v) in matched_pairs}

    assert tp + fp == len(pred_tuples), f"Relaxed match math failure: {tp} + {fp} != {len(pred_tuples)}"
    assert tp + fn == len(gold_tuples), f"Relaxed match math failure: {tp} + {fn} != {len(gold_tuples)}"
    return tp, fp, fn, matched_gold


def evaluate_rules_on_test_reviewed(test_reviewed_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluate the frozen rules baseline against synth_sbdh_test_reviewed.csv.
    Computes:
    - Resolvable subset evaluation (3,340 annotations across 3,303 unique intervals)
    - Full accounting of unresolved ambiguous cases (5 annotations, 0.15%)
    - Worst-case sensitivity / lower-bound metrics treating ambiguous cases as false negatives
    - Note-level category classification metrics (micro, macro, weighted)
    - Span-level exact matching and relaxed/overlap matching
    """
    all_categories = sorted(list({c for group in SBDH_RULES_TAXONOMY.values() for _, cats in group for c in cats}))

    # Note-level category classification accumulators
    cat_tp = {c: 0 for c in all_categories}
    cat_fp = {c: 0 for c in all_categories}
    cat_fn = {c: 0 for c in all_categories}
    cat_support = {c: 0 for c in all_categories}

    # Span-level match accumulators
    exact_span_tp = 0
    exact_span_fp = 0
    exact_span_fn = 0

    relaxed_span_tp = 0
    relaxed_span_fp = 0
    relaxed_span_fn = 0

    total_raw_annotations = 0
    total_resolvable_gold_spans = 0
    total_unique_gold_intervals = 0
    total_pred_spans = 0
    all_ambiguous_cases: List[Dict[str, Any]] = []

    for idx, row in test_reviewed_df.iterrows():
        ex_no = int(row["ex_no"]) if pd.notna(row.get("ex_no")) else None
        text = str(row["Text"])
        spans = parse_delimited_field(row.get("Textspan"))
        sbdhs = parse_delimited_field(row.get("SBDH"))
        presences = parse_delimited_field(row.get("Presence"))
        periods = parse_delimited_field(row.get("Period"))

        total_raw_annotations += len(spans)

        # Align gold spans using validated span alignment module
        gold_aligned = align_spans_in_note(
            text=text,
            spans=spans,
            sbdhs=sbdhs,
            presences=presences,
            periods=periods,
            ex_no=ex_no,
            disambiguate_context=True,
        )

        gold_intervals = gold_aligned["resolved_intervals"]
        total_unique_gold_intervals += len(gold_intervals)

        # Collect ambiguous cases if any
        if gold_aligned["ambiguous_cases_count"] > 0:
            for amb in gold_aligned["ambiguous_cases"]:
                all_ambiguous_cases.append({
                    "ex_no": amb["ex_no"],
                    "span": amb["span"],
                    "sbdh": amb["sbdh"],
                    "ann_idx": amb["ann_idx"],
                    "candidates_count": amb["num_candidates"],
                    "reason": amb["reason"],
                })

        # Decompose gold intervals into (start, end, category)
        # Note: identical intervals with multiple categories contribute multiple tuples
        gold_tuples: Set[Tuple[int, int, str]] = set()
        gold_cats_in_note: Set[str] = set()
        for g in gold_intervals:
            for cat in g["categories"]:
                gold_tuples.add((g["start_char"], g["end_char"], cat))
                gold_cats_in_note.add(cat)

        total_resolvable_gold_spans += len(gold_tuples)

        # Predict spans using frozen rules
        preds = extract_spans_with_rules(text)
        pred_tuples: Set[Tuple[int, int, str]] = set()
        pred_cats_in_note: Set[str] = set()
        for p in preds:
            for cat in p["categories"]:
                pred_tuples.add((p["start_char"], p["end_char"], cat))
                pred_cats_in_note.add(cat)

        total_pred_spans += len(pred_tuples)

        # 1. Note-level Category Classification Accumulators
        for c in all_categories:
            in_gold = c in gold_cats_in_note
            in_pred = c in pred_cats_in_note
            if in_gold and in_pred:
                cat_tp[c] += 1
            elif in_pred and not in_gold:
                cat_fp[c] += 1
            elif in_gold and not in_pred:
                cat_fn[c] += 1
            if in_gold:
                cat_support[c] += 1

        # 2. Span-level Exact Matching
        e_tp, e_fp, e_fn = match_spans_exact(pred_tuples, gold_tuples)
        exact_span_tp += e_tp
        exact_span_fp += e_fp
        exact_span_fn += e_fn

        # 3. Span-level Relaxed / Overlap Matching
        r_tp, r_fp, r_fn, _ = match_spans_relaxed(pred_tuples, gold_tuples)
        relaxed_span_tp += r_tp
        relaxed_span_fp += r_fp
        relaxed_span_fn += r_fn

    # Compute Per-Category Metrics
    per_category_metrics = {}
    macro_precisions = []
    macro_recalls = []
    macro_f1s = []

    total_tp = sum(cat_tp.values())
    total_fp = sum(cat_fp.values())
    total_fn = sum(cat_fn.values())
    total_support = sum(cat_support.values())

    weighted_f1_sum = 0.0

    for c in all_categories:
        tp = cat_tp[c]
        fp = cat_fp[c]
        fn = cat_fn[c]
        sup = cat_support[c]

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        per_category_metrics[c] = {
            "support": sup,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f1, 4),
        }
        macro_precisions.append(p)
        macro_recalls.append(r)
        macro_f1s.append(f1)
        weighted_f1_sum += f1 * sup

    # Micro, Macro, Weighted Classification
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    macro_p = float(np.mean(macro_precisions))
    macro_r = float(np.mean(macro_recalls))
    macro_f1 = float(np.mean(macro_f1s))
    weighted_f1 = weighted_f1_sum / total_support if total_support > 0 else 0.0

    # Span-level Exact Metrics (Resolvable Subset)
    exact_p = exact_span_tp / (exact_span_tp + exact_span_fp) if (exact_span_tp + exact_span_fp) > 0 else 0.0
    exact_r = exact_span_tp / (exact_span_tp + exact_span_fn) if (exact_span_tp + exact_span_fn) > 0 else 0.0
    exact_f1 = (2 * exact_p * exact_r) / (exact_p + exact_r) if (exact_p + exact_r) > 0 else 0.0

    # Span-level Relaxed Metrics (Resolvable Subset)
    relaxed_p = relaxed_span_tp / (relaxed_span_tp + relaxed_span_fp) if (relaxed_span_tp + relaxed_span_fp) > 0 else 0.0
    relaxed_r = relaxed_span_tp / (relaxed_span_tp + relaxed_span_fn) if (relaxed_span_tp + relaxed_span_fn) > 0 else 0.0
    relaxed_f1 = (2 * relaxed_p * relaxed_r) / (relaxed_p + relaxed_r) if (relaxed_p + relaxed_r) > 0 else 0.0

    # Worst-Case Sensitivity Bounds: count all unresolved ambiguous cases as False Negatives (FN)
    unresolved_count = len(all_ambiguous_cases)
    wc_exact_fn = exact_span_fn + unresolved_count
    wc_exact_r = exact_span_tp / (exact_span_tp + wc_exact_fn) if (exact_span_tp + wc_exact_fn) > 0 else 0.0
    wc_exact_f1 = (2 * exact_p * wc_exact_r) / (exact_p + wc_exact_r) if (exact_p + wc_exact_r) > 0 else 0.0

    wc_relaxed_fn = relaxed_span_fn + unresolved_count
    wc_relaxed_r = relaxed_span_tp / (relaxed_span_tp + wc_relaxed_fn) if (relaxed_span_tp + wc_relaxed_fn) > 0 else 0.0
    wc_relaxed_f1 = (2 * relaxed_p * wc_relaxed_r) / (relaxed_p + wc_relaxed_r) if (relaxed_p + wc_relaxed_r) > 0 else 0.0

    # Mathematical assertions
    assert exact_span_tp + exact_span_fp == total_pred_spans
    assert exact_span_tp + exact_span_fn == total_resolvable_gold_spans
    assert relaxed_span_tp + relaxed_span_fp == total_pred_spans
    assert relaxed_span_tp + relaxed_span_fn == total_resolvable_gold_spans
    assert total_resolvable_gold_spans + unresolved_count == total_raw_annotations

    return {
        "dataset": "Synth-SBDH (synth_sbdh_test_reviewed.csv)",
        "total_test_notes": len(test_reviewed_df),
        "total_raw_gold_annotations": total_raw_annotations,
        "resolvable_gold_annotations_count": total_resolvable_gold_spans,
        "unique_resolvable_gold_intervals": total_unique_gold_intervals,
        "unresolved_ambiguous_annotations_count": unresolved_count,
        "unresolved_ambiguous_cases": all_ambiguous_cases,
        "total_pred_spans": total_pred_spans,
        "tasks_evaluated": {
            "task_1_span_location": "EVALUATED (exact boundary character offsets and relaxed character overlap)",
            "task_2_sbdh_category": "EVALUATED (15 social determinant categories assigned to matched intervals)",
            "task_3_presence": "UNEVALUATED (Polarity yes/no assertion not predicted by regex rules baseline)",
            "task_4_temporal_period": "UNEVALUATED (Period current/history not predicted by regex rules baseline)",
        },
        "rules_independence": (
            "Extraction rules were derived exclusively from domain clinical guidelines (PRAPARE, AHC-HRSN, "
            "ICD-10 Z-codes) and training partition patterns. The expert-reviewed test set was frozen and was NOT "
            "used to develop, tune, or optimize extraction patterns."
        ),
        "aggregate_classification_metrics": {
            "micro": {
                "precision": round(micro_p, 4),
                "recall": round(micro_r, 4),
                "f1_score": round(micro_f1, 4),
            },
            "macro": {
                "precision": round(macro_p, 4),
                "recall": round(macro_r, 4),
                "f1_score": round(macro_f1, 4),
            },
            "weighted": {
                "f1_score": round(weighted_f1, 4),
            },
        },
        "span_level_exact_matching": {
            "resolvable_subset": {
                "tp": exact_span_tp,
                "fp": exact_span_fp,
                "fn": exact_span_fn,
                "precision": round(exact_p, 4),
                "recall": round(exact_r, 4),
                "f1_score": round(exact_f1, 4),
                "total_eval_gold": total_resolvable_gold_spans,
                "definition": "Exact character start, end, and SBDH category match.",
            },
            "worst_case_bound_with_ambiguous_as_fn": {
                "tp": exact_span_tp,
                "fp": exact_span_fp,
                "fn": wc_exact_fn,
                "precision": round(exact_p, 4),
                "recall": round(wc_exact_r, 4),
                "f1_score": round(wc_exact_f1, 4),
                "total_eval_gold": total_raw_annotations,
                "definition": "Worst-case sensitivity bound treating all 5 unresolved ambiguous annotations as False Negatives.",
            },
        },
        "span_level_relaxed_matching": {
            "resolvable_subset": {
                "tp": relaxed_span_tp,
                "fp": relaxed_span_fp,
                "fn": relaxed_span_fn,
                "precision": round(relaxed_p, 4),
                "recall": round(relaxed_r, 4),
                "f1_score": round(relaxed_f1, 4),
                "total_eval_gold": total_resolvable_gold_spans,
                "definition": "Overlapping character interval (intersection > 0) and matching SBDH category (1-to-1 matching).",
            },
            "worst_case_bound_with_ambiguous_as_fn": {
                "tp": relaxed_span_tp,
                "fp": relaxed_span_fp,
                "fn": wc_relaxed_fn,
                "precision": round(relaxed_p, 4),
                "recall": round(wc_relaxed_r, 4),
                "f1_score": round(wc_relaxed_f1, 4),
                "total_eval_gold": total_raw_annotations,
                "definition": "Worst-case sensitivity bound treating all 5 unresolved ambiguous annotations as False Negatives.",
            },
        },
        "per_category_metrics": per_category_metrics,
        "methodology_and_regulatory_notice": (
            "This rules-based baseline is a deterministic heuristic keyword/regex system, NOT a machine learning "
            "or deep-learning model. Performance on synthetic clinical notes generated via LLM prompt templates "
            "does NOT establish real-world clinical validity or generalizability to authentic EHR notes. "
            "Rules were frozen prior to evaluation and were not tuned against test errors."
        ),
    }


def main():
    print("=" * 80)
    print("TRACK B: RULES-BASED SBDH BASELINE EVALUATION (SYNTH-SBDH)")
    print("=" * 80)

    dfs, data_dir = load_synth_sbdh()
    test_reviewed_df = dfs["test_reviewed"]
    print(f"Loaded gold-standard test_reviewed set: {len(test_reviewed_df):,d} instances")

    print("\nRunning rules-based extraction against gold-standard evaluation set...")
    results = evaluate_rules_on_test_reviewed(test_reviewed_df)

    # Save artifact
    os.makedirs("reports", exist_ok=True)
    out_file = "reports/sbdh_rules_metrics.json"
    save_json(results, out_file)
    print(f"\n[EXPORT] Rules baseline metrics saved to: {out_file}")

    # Display results
    print("\n" + "=" * 80)
    print("AGGREGATE CLASSIFICATION METRICS (Note-Level)")
    print("=" * 80)
    agg = results["aggregate_classification_metrics"]
    print(f"Micro-Average: Precision={agg['micro']['precision']:.4f} | Recall={agg['micro']['recall']:.4f} | F1={agg['micro']['f1_score']:.4f}")
    print(f"Macro-Average: Precision={agg['macro']['precision']:.4f} | Recall={agg['macro']['recall']:.4f} | F1={agg['macro']['f1_score']:.4f}")
    print(f"Weighted F1:   {agg['weighted']['f1_score']:.4f}")

    print("\n" + "=" * 80)
    print("SPAN-LEVEL EXTRACTION METRICS")
    print("=" * 80)
    ex_res = results["span_level_exact_matching"]["resolvable_subset"]
    ex_wc = results["span_level_exact_matching"]["worst_case_bound_with_ambiguous_as_fn"]
    rel_res = results["span_level_relaxed_matching"]["resolvable_subset"]
    rel_wc = results["span_level_relaxed_matching"]["worst_case_bound_with_ambiguous_as_fn"]

    print(f"Total Raw Gold Annotations:         {results['total_raw_gold_annotations']:,d}")
    print(f"Resolvable Gold Annotations Subset: {results['resolvable_gold_annotations_count']:,d} (across {results['unique_resolvable_gold_intervals']:,d} unique intervals)")
    print(f"Unresolved Ambiguous Annotations:   {results['unresolved_ambiguous_annotations_count']} ({results['unresolved_ambiguous_annotations_count']/results['total_raw_gold_annotations']*100:.2f}%)")
    print(f"\nExact Matching (Resolvable):  Precision={ex_res['precision']:.4f} | Recall={ex_res['recall']:.4f} | F1={ex_res['f1_score']:.4f} (TP={ex_res['tp']}, FP={ex_res['fp']}, FN={ex_res['fn']})")
    print(f"Exact Matching (Worst-Case):  Precision={ex_wc['precision']:.4f} | Recall={ex_wc['recall']:.4f} | F1={ex_wc['f1_score']:.4f} (TP={ex_wc['tp']}, FP={ex_wc['fp']}, FN={ex_wc['fn']})")
    print(f"Relaxed Matching (Resolvable): Precision={rel_res['precision']:.4f} | Recall={rel_res['recall']:.4f} | F1={rel_res['f1_score']:.4f} (TP={rel_res['tp']}, FP={rel_res['fp']}, FN={rel_res['fn']})")
    print(f"Relaxed Matching (Worst-Case): Precision={rel_wc['precision']:.4f} | Recall={rel_wc['recall']:.4f} | F1={rel_wc['f1_score']:.4f} (TP={rel_wc['tp']}, FP={rel_wc['fp']}, FN={rel_wc['fn']})")

    print("\n" + "=" * 80)
    print("PER-CATEGORY BREAKDOWN (15 SBDH Categories)")
    print("=" * 80)
    print(f"{'Category':<35} | {'Support':<8} | {'Precision':<10} | {'Recall':<8} | {'F1-Score':<8}")
    print("-" * 80)
    for cat, m in results["per_category_metrics"].items():
        print(f"{cat:<35} | {m['support']:<8d} | {m['precision']:<10.4f} | {m['recall']:<8.4f} | {m['f1_score']:<8.4f}")

    print("\n[NOTE] " + results["methodology_and_regulatory_notice"])


if __name__ == "__main__":
    main()
