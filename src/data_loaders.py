"""
Reproducible data fetchers and loaders for:
1. UCI Diabetes 130-US Hospitals (Dataset ID 296)
2. Synth-SBDH (bio-nlp-umass/Synth-SBDH)

Designed for robust offline and online execution:
- Prioritizes existing local files on disk.
- Never silently overwrites raw source data.
- Avoids network downloads when local datasets exist.
- Accepts explicit file/directory paths via function arguments, CLI flags, or environment variables.
- Emits detailed, actionable error messages with exact paths if required files are missing.
"""
import os
import sys
import urllib.request
import pandas as pd
from typing import Dict, Optional, Tuple, List


UCI_DATA_URL = "https://archive.ics.uci.edu/static/public/296/data.csv"
SYNTH_SBDH_BASE_URL = "https://huggingface.co/datasets/bio-nlp-umass/Synth-SBDH/raw/main"
SYNTH_SBDH_FILES = [
    "synth_sbdh_train.csv",
    "synth_sbdh_val.csv",
    "synth_sbdh_test.csv",
    "synth_sbdh_test_reviewed.csv",
]


def _is_colab() -> bool:
    return "google.colab" in sys.modules


def find_uci_diabetes_path(explicit_path: Optional[str] = None) -> Tuple[Optional[str], List[str]]:
    """
    Locate diabetic_data.csv or equivalent on disk in priority order:
    1. explicit_path (if supplied)
    2. Environment variable UCI_DATA_PATH or UCI_DATA_DIR
    3. Standard local directories (e.g. D:/Evernorth/UCI-Diabetes, ../UCI-Diabetes, data/uci_diabetes)
    Returns (found_path, searched_paths).
    """
    searched: List[str] = []

    candidates: List[Optional[str]] = [
        explicit_path,
        os.environ.get("UCI_DATA_PATH"),
        os.environ.get("UCI_DATA_DIR"),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "UCI-Diabetes", "diabetic_data.csv")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "UCI-Diabetes")),
        "D:/Evernorth/UCI-Diabetes/diabetic_data.csv",
        "D:/Evernorth/UCI-Diabetes",
        "../UCI-Diabetes/diabetic_data.csv",
        "../UCI-Diabetes",
        "data/uci_diabetes/diabetic_data.csv",
        "data/uci_diabetes",
        "data/diabetic_data.csv",
        "data/data.csv",
    ]

    for cand in candidates:
        if not cand:
            continue
        norm = os.path.normpath(os.path.abspath(cand))
        if norm not in searched:
            searched.append(norm)

        # Check if candidate is directly a valid CSV file
        if os.path.isfile(norm) and os.path.getsize(norm) > 0:
            return norm, searched

        # Check if candidate is a directory containing diabetic_data.csv or data.csv
        if os.path.isdir(norm):
            for sub_name in ["diabetic_data.csv", "data.csv"]:
                sub_path = os.path.normpath(os.path.join(norm, sub_name))
                if sub_path not in searched:
                    searched.append(sub_path)
                if os.path.isfile(sub_path) and os.path.getsize(sub_path) > 0:
                    return sub_path, searched

    return None, searched


def find_synth_sbdh_dir(explicit_dir: Optional[str] = None) -> Tuple[Optional[str], List[str]]:
    """
    Locate Synth-SBDH directory containing the 4 required CSV files in priority order:
    1. explicit_dir (if supplied)
    2. Environment variable SYNTH_SBDH_DATA_DIR or SYNTH_SBDH_DIR
    3. Standard local directories (e.g. D:/Evernorth/Synth-SBDH, ../Synth-SBDH, data/synth_sbdh)
    Returns (found_dir, searched_paths).
    """
    searched: List[str] = []

    candidates: List[Optional[str]] = [
        explicit_dir,
        os.environ.get("SYNTH_SBDH_DATA_DIR"),
        os.environ.get("SYNTH_SBDH_DIR"),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Synth-SBDH")),
        "D:/Evernorth/Synth-SBDH",
        "../Synth-SBDH",
        "data/synth_sbdh",
        "data",
    ]

    for cand in candidates:
        if not cand:
            continue
        norm = os.path.normpath(os.path.abspath(cand))
        if norm not in searched:
            searched.append(norm)

        if os.path.isdir(norm):
            # Check if at least the train and test files exist here
            has_train = os.path.isfile(os.path.join(norm, "synth_sbdh_train.csv"))
            has_test = os.path.isfile(os.path.join(norm, "synth_sbdh_test.csv"))
            if has_train and has_test:
                return norm, searched

    return None, searched


def download_file(url: str, dest_path: str, user_agent: str = "Mozilla/5.0 (Clinical-DSS-Audit)") -> str:
    """Download a file with user-agent headers without overwriting existing files."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"File already exists on disk: {dest_path}. Skipping network download.")
        return dest_path

    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    print(f"Downloading from {url} to {dest_path}...")
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
        out_file.write(response.read())
    print(f"Successfully downloaded {dest_path} ({os.path.getsize(dest_path):,} bytes).")
    return dest_path


def load_uci_diabetes(
    data_path: Optional[str] = None,
    allow_download: Optional[bool] = None,
    low_memory: bool = False
) -> Tuple[pd.DataFrame, str]:
    """
    Load UCI Diabetes 130-US Hospitals dataset.
    Prioritizes local file discovery. Never overwrites raw data.
    Treats '?' as missing values.

    Returns:
        (df, resolved_filepath)
    """
    resolved_path, searched = find_uci_diabetes_path(data_path)

    if resolved_path is not None:
        print(f"[UCI-Diabetes] Found local dataset at: {resolved_path} ({os.path.getsize(resolved_path):,} bytes)")
        df = pd.read_csv(
            resolved_path,
            na_values=["?", ""],
            keep_default_na=False,
            low_memory=low_memory
        )
        # Verify literal 'None' in lab tests is preserved as valid category, not NaN
        assert df["max_glu_serum"].isna().sum() == 0, "max_glu_serum contains NaNs; 'None' was incorrectly converted to missing!"
        assert "None" in df["max_glu_serum"].values, "'None' category missing from max_glu_serum!"
        assert df["A1Cresult"].isna().sum() == 0, "A1Cresult contains NaNs; 'None' was incorrectly converted to missing!"
        assert "None" in df["A1Cresult"].values, "'None' category missing from A1Cresult!"
        return df, resolved_path

    # If allow_download is None, default to True ONLY if running in Google Colab
    if allow_download is None:
        allow_download = _is_colab()

    if not allow_download:
        error_msg = (
            "\n[UCI-Diabetes ERROR] Could not find 'diabetic_data.csv' in any searched location:\n"
            + "\n".join(f"  - {p}" for p in searched)
            + "\n\nHow to supply the dataset:\n"
            + "1. Pass the path explicitly: load_uci_diabetes(data_path='/path/to/diabetic_data.csv')\n"
            + "2. Set the environment variable: set UCI_DATA_PATH=D:\\Evernorth\\UCI-Diabetes\\diabetic_data.csv\n"
            + "3. Or enable download if permitted: load_uci_diabetes(allow_download=True)\n"
        )
        raise FileNotFoundError(error_msg)

    # Fallback download to local cache
    fallback_dest = os.path.abspath("data/uci_diabetes/diabetic_data.csv")
    print(f"[UCI-Diabetes] Local file not found. Downloading to: {fallback_dest}...")
    download_file(UCI_DATA_URL, fallback_dest)
    df = pd.read_csv(
        fallback_dest,
        na_values=["?", ""],
        keep_default_na=False,
        low_memory=low_memory
    )
    # Verify literal 'None' in lab tests is preserved as valid category, not NaN
    assert df["max_glu_serum"].isna().sum() == 0, "max_glu_serum contains NaNs; 'None' was incorrectly converted to missing!"
    assert "None" in df["max_glu_serum"].values, "'None' category missing from max_glu_serum!"
    assert df["A1Cresult"].isna().sum() == 0, "A1Cresult contains NaNs; 'None' was incorrectly converted to missing!"
    assert "None" in df["A1Cresult"].values, "'None' category missing from A1Cresult!"
    return df, fallback_dest


def load_uci_ids_mapping(data_dir_or_file: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Load IDS_mapping.csv if available.
    Returns mapping DataFrame or None if not found.
    """
    candidates = []
    if data_dir_or_file:
        if os.path.isfile(data_dir_or_file):
            parent = os.path.dirname(data_dir_or_file)
            candidates.append(os.path.join(parent, "IDS_mapping.csv"))
        else:
            candidates.append(os.path.join(data_dir_or_file, "IDS_mapping.csv"))

    candidates.extend([
        "D:/Evernorth/UCI-Diabetes/IDS_mapping.csv",
        "../UCI-Diabetes/IDS_mapping.csv",
        "data/uci_diabetes/IDS_mapping.csv",
    ])

    for cand in candidates:
        norm = os.path.normpath(os.path.abspath(cand))
        if os.path.isfile(norm) and os.path.getsize(norm) > 0:
            print(f"[UCI-Diabetes] Found IDs mapping file at: {norm}")
            return pd.read_csv(norm)

    return None


def load_synth_sbdh(
    data_dir: Optional[str] = None,
    allow_download: Optional[bool] = None
) -> Tuple[Dict[str, pd.DataFrame], str]:
    """
    Load all 4 official splits of the Synth-SBDH dataset:
    - train: synth_sbdh_train.csv
    - val: synth_sbdh_val.csv
    - test: synth_sbdh_test.csv
    - test_reviewed: synth_sbdh_test_reviewed.csv

    Returns:
        (dict_of_dfs, resolved_directory_path)
    """
    resolved_dir, searched = find_synth_sbdh_dir(data_dir)

    split_keys = {
        "synth_sbdh_train.csv": "train",
        "synth_sbdh_val.csv": "val",
        "synth_sbdh_test.csv": "test",
        "synth_sbdh_test_reviewed.csv": "test_reviewed",
    }

    if resolved_dir is not None:
        print(f"[Synth-SBDH] Found local dataset directory at: {resolved_dir}")
        dfs = {}
        missing_files = []
        for filename, key in split_keys.items():
            fpath = os.path.join(resolved_dir, filename)
            if os.path.isfile(fpath) and os.path.getsize(fpath) > 0:
                dfs[key] = pd.read_csv(fpath)
            else:
                missing_files.append((filename, fpath))

        if not missing_files:
            return dfs, resolved_dir

        print(f"[Synth-SBDH] Directory found but missing {len(missing_files)} files: {[m[0] for m in missing_files]}")

    # If allow_download is None, default to True ONLY if running in Google Colab
    if allow_download is None:
        allow_download = _is_colab()

    if not allow_download:
        error_msg = (
            "\n[Synth-SBDH ERROR] Could not find Synth-SBDH CSV files in any searched location:\n"
            + "\n".join(f"  - {p}" for p in searched)
            + "\n\nRequired files:\n"
            + "  - synth_sbdh_train.csv\n"
            + "  - synth_sbdh_val.csv\n"
            + "  - synth_sbdh_test.csv\n"
            + "  - synth_sbdh_test_reviewed.csv\n\n"
            + "How to supply the dataset:\n"
            + "1. Pass the path explicitly: load_synth_sbdh(data_dir='/path/to/Synth-SBDH')\n"
            + "2. Set the environment variable: set SYNTH_SBDH_DATA_DIR=D:\\Evernorth\\Synth-SBDH\n"
            + "3. Or enable download if permitted: load_synth_sbdh(allow_download=True)\n"
        )
        raise FileNotFoundError(error_msg)

    # Download missing files to fallback cache
    target_dir = os.path.abspath(resolved_dir if resolved_dir else "data/synth_sbdh")
    os.makedirs(target_dir, exist_ok=True)
    dfs = {}
    for filename, key in split_keys.items():
        fpath = os.path.join(target_dir, filename)
        if not (os.path.isfile(fpath) and os.path.getsize(fpath) > 0):
            url = f"{SYNTH_SBDH_BASE_URL}/{filename}"
            download_file(url, fpath)
        dfs[key] = pd.read_csv(fpath)

    return dfs, target_dir
