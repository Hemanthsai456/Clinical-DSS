"""
Script to execute full SBDH span alignment audit across all Synth-SBDH splits.
Produces:
1. reports/sbdh_alignment/alignment_audit_report.json
2. reports/sbdh_alignment/unresolved_alignment_cases.json
3. Console summary reporting exact counts of exact, multi-label, casing, contextual, and unresolved cases.
"""
import os
import sys
import json

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loaders import load_synth_sbdh
from src.span_alignment import audit_dataset_span_alignment
from src.utils import NpEncoder, save_json


def main():
    print("=" * 80)
    print("SYNTH-SBDH SPAN ALIGNMENT & VALIDATION GATE AUDIT")
    print("=" * 80)

    dfs, data_dir = load_synth_sbdh()
    print(f"Loaded Synth-SBDH dataset from: {data_dir}")

    os.makedirs("reports/sbdh_alignment", exist_ok=True)

    split_reports = {}
    all_unresolved = []

    for split_name in ["train", "val", "test", "test_reviewed"]:
        if split_name not in dfs:
            continue
        df = dfs[split_name]
        rep = audit_dataset_span_alignment(df, split_name)
        split_reports[split_name] = rep

        print(f"\n--- Split: {split_name:13s} ({rep['total_notes']:,} notes) ---")
        print(f"  Total Extracted Annotations:          {rep['total_annotations']:,}")
        print(f"  Resolved Unique Intervals:            {rep['resolved_intervals_total']:,}")
        print(f"  Multi-Label Identical Intervals:      {rep['multi_label_identical_intervals_count']:,}")
        print(f"  Casing-Only Normalization Intervals:  {rep['case_insensitive_intervals_count']:,}")
        print(f"  Contextually Disambiguated Intervals: {rep['contextual_resolved_intervals_count']:,}")
        print(f"  Ambiguous Unresolved Annotations:     {rep['ambiguous_annotations_count']:,}")
        print(f"  Unmatched / Missing Spans:            {rep['unmatched_annotations_count']:,}")

        if rep["unresolved_cases"]:
            for u in rep["unresolved_cases"]:
                u["split"] = split_name
                all_unresolved.append(u)

    # Export machine-readable reports
    report_path = "reports/sbdh_alignment/alignment_audit_report.json"
    unresolved_path = "reports/sbdh_alignment/unresolved_alignment_cases.json"

    # Don't duplicate full unresolved list inside report_path to keep it clean
    summary_export = {
        split: {k: v for k, v in rep.items() if k != "unresolved_cases"}
        for split, rep in split_reports.items()
    }
    save_json(summary_export, report_path)
    save_json(all_unresolved, unresolved_path)

    print("\n" + "=" * 80)
    print(f"[EXPORT] Alignment Audit Summary exported to: {report_path}")
    print(f"[EXPORT] Total Unresolved Ambiguous Cases: {len(all_unresolved)} exported to: {unresolved_path}")
    print("=" * 80)

    if all_unresolved:
        print("\nBreakdown of Unresolved Ambiguous Cases across splits:")
        for idx, u in enumerate(all_unresolved, 1):
            print(f"  {idx}. Split [{u.get('split')}], ex_no {u.get('ex_no')}: Span '{u.get('span')}' ({u.get('num_candidates')} candidates in text)")
            print(f"     Reason: {u.get('reason')}")


if __name__ == "__main__":
    main()
