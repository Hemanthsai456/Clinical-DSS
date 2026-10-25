"""
Comprehensive audit and validation logic for Synth-SBDH (bio-nlp-umass/Synth-SBDH).
Audits multi-split schemas, annotation alignments, span validity, expert review operations,
character offset recovery, multi-label overlapping spans, and synthetic limitations.
Updated for Milestone 1.1 Modeling Readiness Gate with contextual span recovery.
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional


EXPECTED_SPLIT_COLUMNS = {
    "standard": ["ex_no", "Text", "Textspan", "SBDH", "Presence", "Period", "Reasoning"],
    "reviewed": ["ex_no", "Text", "Textspan", "Reasoning", "SBDH", "Presence", "Period", "Operation"],
}


def parse_delimited_field(val: Any, delimiter: str = "||") -> List[str]:
    """Safely parse a ||-delimited field into a list of stripped strings."""
    if pd.isna(val) or val is None:
        return []
    s = str(val).strip()
    if not s:
        return []
    return [item.strip() for item in s.split(delimiter)]


def recover_spans_contextual(
    text: str,
    spans: List[str],
    sbdhs: Optional[List[str]] = None,
    presences: Optional[List[str]] = None,
    periods: Optional[List[str]] = None,
    operations: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Contextually recover 0-indexed character offsets (start_char, end_char) for all spans in a row.
    Handles:
    1. Verbatim exact matches.
    2. Case-insensitive normalizations (e.g. expert lowercasing in reviewed test).
    3. Multi-label identical spans (same text phrase annotated for multiple SBDH categories).
    4. Sequential repeated distinct mentions of the same word (e.g. 'pain' current vs 'pain' history).
    """
    results = []
    text_lower = text.lower()

    # Track frequency of each span string in the annotation list
    span_counts_in_list: Dict[str, int] = {}
    for s in spans:
        clean = s.strip().lower()
        span_counts_in_list[clean] = span_counts_in_list.get(clean, 0) + 1

    last_end_by_span: Dict[str, int] = {}

    for i, span in enumerate(spans):
        s_clean = span.strip()
        s_lower = s_clean.lower()
        sbdh = sbdhs[i] if sbdhs and i < len(sbdhs) else "N/A"
        pres = presences[i] if presences and i < len(presences) else "N/A"
        per = periods[i] if periods and i < len(periods) else "N/A"
        op = operations[i] if operations and i < len(operations) else "N/A"

        if not s_clean:
            results.append({
                "span": "",
                "start_char": -1,
                "end_char": -1,
                "match_type": "empty_span",
                "is_multi_label": False,
                "is_ambiguous": False,
                "sbdh": sbdh,
                "presence": pres,
                "period": per,
                "operation": op,
                "matched_text": "",
            })
            continue

        total_occurrences_in_text = text_lower.count(s_lower)
        is_exact = s_clean in text

        # Check if this is a repeated distinct mention in text or a multi-label overlapping span
        is_multi_label = (total_occurrences_in_text == 1 and span_counts_in_list.get(s_lower, 0) > 1)
        is_sequential_repeat = (total_occurrences_in_text > 1 and span_counts_in_list.get(s_lower, 0) > 1)

        if is_sequential_repeat:
            start_search = last_end_by_span.get(s_lower, 0)
            idx = text_lower.find(s_lower, start_search)
            if idx == -1:
                # Wrap around if search ran past the end
                idx = text_lower.find(s_lower)
        else:
            idx = text_lower.find(s_lower)

        start_char = idx
        end_char = start_char + len(s_clean) if start_char != -1 else -1

        if start_char != -1:
            last_end_by_span[s_lower] = end_char

        match_type = "exact" if is_exact else ("case_insensitive" if start_char != -1 else "unmatched")
        matched_snippet = text[start_char:end_char] if start_char != -1 else ""

        results.append({
            "span": s_clean,
            "start_char": start_char,
            "end_char": end_char,
            "match_type": match_type,
            "total_occurrences_in_text": total_occurrences_in_text,
            "is_multi_label": is_multi_label,
            "is_ambiguous": total_occurrences_in_text > 1,
            "sbdh": sbdh,
            "presence": pres,
            "period": per,
            "operation": op,
            "matched_text": matched_snippet,
        })

    return results


def recover_character_offsets(text: str, span: str) -> Dict[str, Any]:
    """
    Single-span helper for standalone queries.
    """
    clean_span = span.strip()
    if not clean_span:
        return {
            "start_char": -1,
            "end_char": -1,
            "match_type": "empty_span",
            "occurrences": 0,
            "is_ambiguous": False,
            "matched_text": "",
        }

    exact_count = text.count(clean_span)
    if exact_count > 0:
        start_char = text.find(clean_span)
        end_char = start_char + len(clean_span)
        return {
            "start_char": start_char,
            "end_char": end_char,
            "match_type": "exact",
            "occurrences": exact_count,
            "is_ambiguous": exact_count > 1,
            "matched_text": text[start_char:end_char],
        }

    text_lower = text.lower()
    span_lower = clean_span.lower()
    ci_count = text_lower.count(span_lower)
    if ci_count > 0:
        start_char = text_lower.find(span_lower)
        end_char = start_char + len(clean_span)
        return {
            "start_char": start_char,
            "end_char": end_char,
            "match_type": "case_insensitive",
            "occurrences": ci_count,
            "is_ambiguous": ci_count > 1,
            "matched_text": text[start_char:end_char],
        }

    return {
        "start_char": -1,
        "end_char": -1,
        "match_type": "unmatched",
        "occurrences": 0,
        "is_ambiguous": False,
        "matched_text": "",
    }


def audit_single_split(df: pd.DataFrame, split_name: str) -> Dict[str, Any]:
    """Audit schema, missing values, duplicates, and annotation alignment for one split."""
    total_rows = len(df)
    unique_ids = int(df["ex_no"].nunique())
    duplicate_ids = int(df["ex_no"].duplicated().sum())
    duplicate_texts = int(df["Text"].duplicated().sum())

    missing_counts = {col: int(cnt) for col, cnt in df.isna().sum().items() if cnt > 0}

    # Column verification
    is_reviewed = "Operation" in df.columns
    expected_cols = EXPECTED_SPLIT_COLUMNS["reviewed"] if is_reviewed else EXPECTED_SPLIT_COLUMNS["standard"]
    col_order_match = list(df.columns) == expected_cols

    # Multi-label alignment & span verification
    length_mismatches = []
    exact_span_matches = 0
    case_insensitive_span_matches = 0
    span_not_in_text = []
    ambiguous_spans = []
    multi_label_spans_count = 0
    
    total_annotations = 0
    sbdh_dist: Dict[str, int] = {}
    presence_dist: Dict[str, int] = {}
    period_dist: Dict[str, int] = {}
    operation_dist: Dict[str, int] = {}

    detailed_casing_cases = []

    for idx, row in df.iterrows():
        ex_no = row["ex_no"]
        text = str(row["Text"])
        spans = parse_delimited_field(row["Textspan"])
        sbdhs = parse_delimited_field(row["SBDH"])
        presences = parse_delimited_field(row["Presence"])
        periods = parse_delimited_field(row["Period"])
        reasonings = parse_delimited_field(row["Reasoning"])
        operations = parse_delimited_field(row["Operation"]) if is_reviewed else []

        n_spans = len(spans)
        n_sbdh = len(sbdhs)
        n_pres = len(presences)
        n_per = len(periods)
        
        # Check alignment of tuple components
        base_lens = [n_spans, n_sbdh, n_pres, n_per]
        if is_reviewed:
            base_lens.append(len(operations))

        if len(set(base_lens)) > 1:
            length_mismatches.append({
                "row_index": int(idx),
                "ex_no": int(ex_no) if pd.notna(ex_no) else None,
                "span_count": n_spans,
                "sbdh_count": n_sbdh,
                "presence_count": n_pres,
                "period_count": n_per,
                "operation_count": len(operations) if is_reviewed else None,
            })
            continue

        total_annotations += n_spans

        # Contextual span recovery
        contextual_spans = recover_spans_contextual(
            text, spans, sbdhs, presences, periods, operations if is_reviewed else None
        )

        for rec in contextual_spans:
            sbdh_cat = rec["sbdh"]
            pres = rec["presence"]
            per = rec["period"]
            op = rec["operation"]

            sbdh_dist[sbdh_cat] = sbdh_dist.get(sbdh_cat, 0) + 1
            presence_dist[pres] = presence_dist.get(pres, 0) + 1
            period_dist[per] = period_dist.get(per, 0) + 1

            if is_reviewed:
                operation_dist[op] = operation_dist.get(op, 0) + 1

            if rec["is_multi_label"]:
                multi_label_spans_count += 1

            if rec["is_ambiguous"]:
                ambiguous_spans.append({
                    "ex_no": int(ex_no) if pd.notna(ex_no) else None,
                    "span": rec["span"],
                    "occurrences": rec["total_occurrences_in_text"],
                    "match_type": rec["match_type"],
                    "sbdh": sbdh_cat,
                })

            if rec["match_type"] == "exact":
                exact_span_matches += 1
            elif rec["match_type"] == "case_insensitive":
                case_insensitive_span_matches += 1
                if is_reviewed:
                    detailed_casing_cases.append({
                        "case_number": len(detailed_casing_cases) + 1,
                        "ex_no": int(ex_no) if pd.notna(ex_no) else None,
                        "annotated_span": rec["span"],
                        "matched_text_verbatim": rec["matched_text"],
                        "start_char": rec["start_char"],
                        "end_char": rec["end_char"],
                        "exact_match": False,
                        "case_insensitive_match": True,
                        "occurrences_in_text": rec["total_occurrences_in_text"],
                        "is_ambiguous": rec["is_ambiguous"],
                        "is_multi_label": rec["is_multi_label"],
                        "operation": op,
                        "sbdh_category": sbdh_cat,
                        "presence": pres,
                        "period": per,
                        "text": text,
                    })
            else:
                span_not_in_text.append({
                    "row_index": int(idx),
                    "ex_no": int(ex_no) if pd.notna(ex_no) else None,
                    "span": rec["span"],
                    "sbdh": sbdh_cat,
                })

    # Automated assertions
    assert len(length_mismatches) == 0, f"Split '{split_name}' contains {len(length_mismatches)} rows with tuple length mismatches!"
    assert len(span_not_in_text) == 0, f"Split '{split_name}' contains {len(span_not_in_text)} unrecoverable spans!"

    return {
        "split_name": split_name,
        "total_rows": total_rows,
        "columns": list(df.columns),
        "columns_match_expected": col_order_match,
        "has_operation_column": is_reviewed,
        "unique_ex_no": unique_ids,
        "duplicate_ex_no": duplicate_ids,
        "duplicate_texts": duplicate_texts,
        "missing_values": missing_counts,
        "annotation_metrics": {
            "total_extracted_annotations": total_annotations,
            "length_mismatch_rows_count": len(length_mismatches),
            "exact_span_matches": exact_span_matches,
            "case_insensitive_only_span_matches": case_insensitive_span_matches,
            "unmatched_spans_count": len(span_not_in_text),
            "multi_label_overlapping_spans_count": multi_label_spans_count,
            "ambiguous_repeated_spans_count": len(ambiguous_spans),
            "ambiguous_repeated_spans_samples": ambiguous_spans[:5],
        },
        "distributions": {
            "sbdh_categories": dict(sorted(sbdh_dist.items(), key=lambda x: x[1], reverse=True)),
            "presence": dict(sorted(presence_dist.items(), key=lambda x: x[1], reverse=True)),
            "period": dict(sorted(period_dist.items(), key=lambda x: x[1], reverse=True)),
            "operation": dict(sorted(operation_dist.items(), key=lambda x: x[1], reverse=True)) if is_reviewed else None,
        },
        "detailed_casing_cases": detailed_casing_cases if is_reviewed else [],
    }


def audit_synth_sbdh_all(dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Audit all 4 splits and generate comparative cross-split evaluation."""
    split_audits = {}
    for split_name, df in dfs.items():
        split_audits[split_name] = audit_single_split(df, split_name)

    # 1. Independent splits text overlap verification
    train_texts = set(dfs["train"]["Text"]) if "train" in dfs else set()
    val_texts = set(dfs["val"]["Text"]) if "val" in dfs else set()
    test_texts = set(dfs["test"]["Text"]) if "test" in dfs else set()

    train_val_text_leak = len(train_texts.intersection(val_texts))
    train_test_text_leak = len(train_texts.intersection(test_texts))
    val_test_text_leak = len(val_texts.intersection(test_texts))

    assert train_val_text_leak == 0, f"Text leakage between Train and Val: {train_val_text_leak}"
    assert train_test_text_leak == 0, f"Text leakage between Train and Test: {train_test_text_leak}"
    assert val_test_text_leak == 0, f"Text leakage between Val and Test: {val_test_text_leak}"

    # 2. Strict evaluation contract: Test vs Reviewed-Test subset check
    test_ex = set(dfs["test"]["ex_no"]) if "test" in dfs else set()
    rev_ex = set(dfs["test_reviewed"]["ex_no"]) if "test_reviewed" in dfs else set()
    
    is_reviewed_subset = rev_ex.issubset(test_ex)
    omitted_in_reviewed = sorted(list(test_ex - rev_ex))
    extra_in_reviewed = sorted(list(rev_ex - test_ex))

    assert is_reviewed_subset, f"Reviewed test contains {len(extra_in_reviewed)} IDs not in standard test: {extra_in_reviewed}"
    assert len(omitted_in_reviewed) == 23, f"Expected exactly 23 omitted IDs in reviewed test, got {len(omitted_in_reviewed)}"

    # Check text encoding discrepancies between test and reviewed-test
    test_text_map = dict(zip(dfs["test"]["ex_no"], dfs["test"]["Text"]))
    rev_text_map = dict(zip(dfs["test_reviewed"]["ex_no"], dfs["test_reviewed"]["Text"]))
    encoding_discrepancies = []
    for ex in rev_ex:
        t_text = test_text_map[ex]
        r_text = rev_text_map[ex]
        if t_text != r_text:
            encoding_discrepancies.append({
                "ex_no": int(ex),
                "standard_text": t_text,
                "reviewed_text": r_text,
                "reason": "Character encoding normalization (e.g. fixed corrupted apostrophe 'doesnt' -> 'doesn\\'t')"
            })

    test_vs_reviewed_contract = {
        "standard_test_rows": len(test_ex),
        "reviewed_test_rows": len(rev_ex),
        "is_reviewed_subset_of_standard": is_reviewed_subset,
        "omitted_records_count": len(omitted_in_reviewed),
        "omitted_ex_no_list": omitted_in_reviewed,
        "encoding_corrections_count": len(encoding_discrepancies),
        "encoding_corrections": encoding_discrepancies,
        "evaluation_contract_role": (
            "The expert-reviewed test set (synth_sbdh_test_reviewed.csv, 1,732 instances) is formally "
            "designated as the PRIMARY FINAL EVALUATION SET for the SDoH extraction model. "
            "It must NEVER be used for model training, preprocessing fitting, or hyperparameter tuning. "
            "The 23 records omitted from standard test represent machine-generated synthetic items that "
            "expert clinical annotators determined contained no valid SBDH cues or were unannotatable."
        ),
        "operation_column_contract": (
            "The 'Operation' column (values: 'keep', 'add', 'correct') is strictly an audit and curation "
            "provenance metadata field. It must NEVER be used as a feature, input, or prediction target "
            "in any modeling pipeline."
        ),
        "multi_label_architecture_requirement": (
            "In 38 rows of test_reviewed and 33 rows of train, identical text phrases are annotated for multiple "
            "SBDH categories (e.g. 'cannot afford to continue physical therapy' -> 'financial insecurity' AND 'barriers to care'). "
            "Standard single-label token classification (flat BIO tagging) will cause label collisions. "
            "The NLP extraction pipeline must employ multi-label token classification or a span-based/seq2seq extraction architecture."
        ),
    }

    cross_split_clarification = {
        "independent_splits_overlap": {
            "train_val_overlap": train_val_text_leak,
            "train_test_overlap": train_test_text_leak,
            "val_test_overlap": val_test_text_leak,
            "status": "PASSED - ZERO TEXT OVERLAP ACROSS INDEPENDENT SPLITS",
        },
        "reviewed_test_relationship": {
            "status": "INTENTIONALLY RELATED EXPERT-CURATED SUBSET",
            "detail": (
                "The expert-reviewed test set is NOT an independent fourth dataset. "
                "It is an expert-curated ground-truth version of the test split (1,732 of 1,755 examples). "
                "Zero training or validation text appears in either test set."
            ),
        },
    }

    synthetic_caveats = [
        "Synthetic clinical notes are generated via LLM prompt templates and do not exhibit authentic clinical noise (e.g. shorthand, acronyms, dictation typos).",
        "Label co-occurrence patterns in synthetic text may reflect prompting biases rather than genuine clinical SDoH comorbidities.",
        "Models tuned exclusively on synthetic notes risk catastrophic generalization failure when deployed on genuine EHR notes.",
        "Synth-SBDH is strictly an educational NLP benchmark for testing extraction pipelines, NOT a clinical deployment asset.",
    ]

    return {
        "dataset_name": "Synth-SBDH (bio-nlp-umass/Synth-SBDH)",
        "source_url": "https://huggingface.co/datasets/bio-nlp-umass/Synth-SBDH",
        "split_audits": split_audits,
        "test_vs_reviewed_contract": test_vs_reviewed_contract,
        "cross_split_overlap_audit": cross_split_clarification,
        "synthetic_data_limitations": synthetic_caveats,
        "roles_and_protocols": {
            "train": "Model training and parameter optimization only.",
            "val": "Validation, hyperparameter tuning, model selection, and early stopping.",
            "test": "Standard benchmark evaluation (machine synthetic baseline).",
            "test_reviewed": "Primary final gold-standard evaluation set (expert-reviewed). NEVER train or tune on this set.",
        },
    }
