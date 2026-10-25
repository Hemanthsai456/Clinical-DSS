"""
Utility functions for reproducibility, environment profiling, and serialization.
"""
import os
import sys
import json
import random
import numpy as np
import pandas as pd


def set_seeds(seed: int = 42) -> None:
    """Set random seeds across standard library, numpy, and environment."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def is_colab() -> bool:
    """Detect if code is executing in a Google Colab runtime."""
    return 'google.colab' in sys.modules


def get_environment_info() -> dict:
    """Gather runtime environment details and core package versions."""
    info = {
        'python_version': sys.version.split()[0],
        'platform': sys.platform,
        'is_colab': is_colab(),
    }
    packages = ['pandas', 'numpy', 'sklearn', 'requests', 'matplotlib', 'seaborn', 'huggingface_hub', 'ucimlrepo', 'nbformat']
    pkg_versions = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            pkg_versions[pkg] = getattr(mod, '__version__', 'unknown')
        except ImportError:
            pkg_versions[pkg] = 'not_installed'
    info['packages'] = pkg_versions
    return info


class NpEncoder(json.JSONEncoder):
    """JSON encoder handling NumPy scalar and array types, plus sets."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return sorted(list(obj))
        if pd.isna(obj):
            return None
        return super().default(obj)


def save_json(data: dict, filepath: str) -> str:
    """Safely serialize a dictionary to formatted JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, cls=NpEncoder)
    return filepath
