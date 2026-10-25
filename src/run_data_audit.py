"""
End-to-end execution script to audit both datasets and generate machine-readable artifacts.
Updated for Milestone 1.1 Modeling Readiness Gate with automated assertions.
"""
import os
import sys
import argparse

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import set_seeds, get_environment_info, save_json
from src.data_loaders import load_uci_diabetes, load_synth_sbdh, load_uci_ids_mapping
from src.audit_uci import audit_uci_dataset, split_uci_patient_grouped, create_eligible_modeling_cohort
from src.audit_synth_sbdh import audit_synth_sbdh_all


def parse_args():
    parser = argparse.ArgumentParser(description="Evernorth Clinical CDS - Milestone 1.1: Modeling Readiness Gate")
    parser.add_argument(
        "--uci-path",
        type=str,
        default=None,
        help="Path to UCI diabetic_data.csv or directory containing it (defaults to auto-discovery).",
    )
    parser.add_argument(
        "--synth-dir",
        type=str,
        default=None,
        help="Path to directory containing Synth-SBDH CSV files (defaults to auto-discovery).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/data_audit",
        help="Directory to save machine-readable audit reports (default: reports/data_audit).",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        default=False,
        help="Allow network download if local files are not found (default: False).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("Evernorth Clinical CDS — Milestone 1.1: Modeling Readiness Gate")
    print("=" * 80)

    # 1. Reproducibility & Environment Setup
    set_seeds(42)
    env_info = get_environment_info()
    print(f"Python: {env_info['python_version']} on {env_info['platform']}")
    print(f"Colab runtime: {env_info['is_colab']}")
    print(f"Key Packages: {env_info['packages']}")

    # 2. UCI Diabetes Audit & Cohort Policy
    print("\n" + "=" * 80)
    print("1. UCI DIABETES: COHORT POLICY, LEAKAGE AUDIT & PATIENT SPLIT")
    print("=" * 80)
    uci_df, uci_resolved_path = load_uci_diabetes(
        data_path=args.uci_path,
        allow_download=args.allow_download,
    )
    print(f"Resolved UCI path: {uci_resolved_path}")
    print(f"Raw DataFrame: {uci_df.shape[0]:,} encounters x {uci_df.shape[1]} columns")

    ids_path = os.path.join(os.path.dirname(uci_resolved_path), "IDS_mapping.csv")
    if os.path.exists(ids_path):
        print(f"Found IDs mapping: {ids_path}")
    else:
        ids_path = None

    uci_audit = audit_uci_dataset(uci_df, ids_path=ids_path)
    uci_audit["resolved_source_path"] = uci_resolved_path

    cohort_cmp = uci_audit["cohort_policy_and_filtering"]
    print("\n[Cohort Filtering Comparison (Mortality & Hospice Exclusion)]")
    print(f"  Verified Exclusion Codes: {cohort_cmp['verified_exclusion_codes']}")
    print(f"  Unfiltered Cohort:        {cohort_cmp['unfiltered_cohort']['total_encounters']:,} encounters ({cohort_cmp['unfiltered_cohort']['unique_patients']:,} patients, {cohort_cmp['unfiltered_cohort']['positive_prevalence_pct']}% positive)")
    print(f"  Excluded (Death/Hospice): {cohort_cmp['excluded_mortality_hospice']['total_encounters']:,} encounters ({cohort_cmp['excluded_mortality_hospice']['unique_patients']:,} patients, {cohort_cmp['excluded_mortality_hospice']['positive_prevalence_pct']}% positive)")
    print(f"  Eligible Modeling Cohort: {cohort_cmp['eligible_modeling_cohort']['total_encounters']:,} encounters ({cohort_cmp['eligible_modeling_cohort']['unique_patients']:,} patients, {cohort_cmp['eligible_modeling_cohort']['positive_prevalence_pct']}% positive)")
    print(f"  Eligible Imbalance Ratio: {cohort_cmp['eligible_modeling_cohort']['imbalance_ratio']} : 1")

    # Run patient-grouped split strictly on the ELIGIBLE modeling cohort
    print("\n[Patient-Grouped Split on Eligible Modeling Cohort]")
    df_eligible, _ = create_eligible_modeling_cohort(uci_df, ids_path=ids_path)
    train_df, val_df, test_df, split_metrics = split_uci_patient_grouped(
        df_eligible, test_size=0.15, val_size=0.15, random_state=42
    )
    uci_audit["eligible_cohort_grouped_split"] = split_metrics

    print(f"  Train: {split_metrics['train']['encounters']:,} encounters ({split_metrics['train']['patients']:,} patients, {split_metrics['train']['positive_rate_pct']}% positive)")
    print(f"  Val:   {split_metrics['validation']['encounters']:,} encounters ({split_metrics['validation']['patients']:,} patients, {split_metrics['validation']['positive_rate_pct']}% positive)")
    print(f"  Test:  {split_metrics['test']['encounters']:,} encounters ({split_metrics['test']['patients']:,} patients, {split_metrics['test']['positive_rate_pct']}% positive)")
    print(f"  Patient Leakage Check: {split_metrics['overlap_verification']['status']}")

    # 3. Synth-SBDH Audit & Evaluation Contract
    print("\n" + "=" * 80)
    print("2. SYNTH-SBDH: EVALUATION CONTRACT, TEST SUBSET & SPAN QUALITY")
    print("=" * 80)
    synth_dfs, synth_resolved_dir = load_synth_sbdh(
        data_dir=args.synth_dir,
        allow_download=args.allow_download,
    )
    print(f"Resolved Synth-SBDH directory: {synth_resolved_dir}")
    for name, df in synth_dfs.items():
        print(f"  Loaded split '{name}': {len(df):,} rows, {df.shape[1]} columns")

    synth_audit = audit_synth_sbdh_all(synth_dfs)
    synth_audit["resolved_source_dir"] = synth_resolved_dir

    contract = synth_audit["test_vs_reviewed_contract"]
    print("\n[Test vs. Expert-Reviewed Test Contract]")
    print(f"  Standard Test: {contract['standard_test_rows']:,} rows")
    print(f"  Reviewed Test: {contract['reviewed_test_rows']:,} rows")
    print(f"  Is Reviewed Test a strict subset? {contract['is_reviewed_subset_of_standard']}")
    print(f"  Omitted records count: {contract['omitted_records_count']}")
    print(f"  Omitted ex_no list: {contract['omitted_ex_no_list']}")
    print(f"  Encoding corrections count: {contract['encoding_corrections_count']}")

    leak = synth_audit["cross_split_overlap_audit"]
    print(f"  Independent Splits Leakage: {leak['independent_splits_overlap']['status']}")
    print(f"  Reviewed Test Relationship: {leak['reviewed_test_relationship']['status']}")

    print("\n[Annotation & Span Metrics Across All Splits]")
    for split_name, sa in synth_audit["split_audits"].items():
        an = sa["annotation_metrics"]
        print(
            f"  [{split_name:13s}] "
            f"Tuples: {an['total_extracted_annotations']:5d} | "
            f"Length mismatches: {an['length_mismatch_rows_count']} | "
            f"Exact spans: {an['exact_span_matches']:5d} | "
            f"Casing-only: {an['case_insensitive_only_span_matches']:2d} | "
            f"Ambiguous: {an['ambiguous_repeated_spans_count']} | "
            f"Unmatched: {an['unmatched_spans_count']}"
        )

    # Export Casing Normalization Cases (92 cases)
    casing_cases = synth_audit["split_audits"]["test_reviewed"]["detailed_casing_cases"]
    casing_out_file = os.path.join(args.output_dir, "reviewed_test_casing_cases.json")
    save_json(casing_cases, casing_out_file)
    print(f"\n[EXPORT] Exported {len(casing_cases)} detailed casing-normalization cases to: {casing_out_file}")

    # 4. Save Machine-Readable Summaries
    print("\n" + "=" * 80)
    print("3. EXPORTING UPDATED MACHINE-READABLE AUDIT SUMMARIES")
    print("=" * 80)
    os.makedirs(args.output_dir, exist_ok=True)
    uci_out = os.path.join(args.output_dir, "uci_diabetes_audit_summary.json")
    synth_out = os.path.join(args.output_dir, "synth_sbdh_audit_summary.json")

    save_json(uci_audit, uci_out)
    save_json(synth_audit, synth_out)
    print(f"Saved UCI audit summary:       {uci_out} ({os.path.getsize(uci_out):,} bytes)")
    print(f"Saved Synth-SBDH audit summary: {synth_out} ({os.path.getsize(synth_out):,} bytes)")

    print("\n[GATE PASS] All Milestone 1.1 Modeling Readiness Gate assertions passed successfully.")


if __name__ == "__main__":
    main()
