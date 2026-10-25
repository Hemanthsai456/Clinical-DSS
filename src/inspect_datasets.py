"""
Exploratory dataset profiling utility using official loaders.
"""
import os
import sys
import pandas as pd

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loaders import load_uci_diabetes, load_synth_sbdh, load_uci_ids_mapping
from src.audit_uci import audit_uci_dataset
from src.audit_synth_sbdh import audit_synth_sbdh_all


def profile_datasets():
    print("=" * 60)
    print("PROFILING UCI DIABETES")
    print("=" * 60)
    uci_df, uci_path = load_uci_diabetes()
    print(f"Path: {uci_path}")
    print(f"Shape: {uci_df.shape}")
    ids_df = load_uci_ids_mapping(uci_path)
    if ids_df is not None:
        print(f"IDs Mapping: {len(ids_df)} rows")

    print("\n" + "=" * 60)
    print("PROFILING SYNTH-SBDH")
    print("=" * 60)
    synth_dfs, synth_dir = load_synth_sbdh()
    print(f"Directory: {synth_dir}")
    for name, df in synth_dfs.items():
        print(f"  {name:13s}: {df.shape[0]:,d} rows x {df.shape[1]} columns")


if __name__ == "__main__":
    profile_datasets()
