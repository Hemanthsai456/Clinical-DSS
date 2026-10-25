"""
Standalone verification script for patient-grouped partitioning.
Validates zero patient leakage across train, val, and test splits.
"""
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loaders import load_uci_diabetes
from src.audit_uci import split_uci_patient_grouped


def main():
    print("Loading UCI Diabetes dataset...")
    df, resolved_path = load_uci_diabetes()
    print(f"Loaded from: {resolved_path} ({len(df):,} encounters)")

    print("Executing patient-grouped split (70% train / 15% val / 15% test)...")
    train_df, val_df, test_df, metrics = split_uci_patient_grouped(df, test_size=0.15, val_size=0.15, random_state=42)

    for split_name in ["train", "validation", "test"]:
        m = metrics[split_name]
        print(f"  [{split_name.upper():10s}] Encounters: {m['encounters']:6,d} | Patients: {m['patients']:6,d} | Readmit <30d: {m['positive_rate_pct']:.2f}%")

    print(f"\nOverlap Verification: {metrics['overlap_verification']['status']}")


if __name__ == "__main__":
    main()
