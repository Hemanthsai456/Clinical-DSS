"""
Inventory script to compute SHA256 hashes and file sizes for all key artifacts in Clinical-DSS.
"""
import os
import hashlib
import json

ARTIFACTS = [
    # Track A Models & Reports
    "models/baseline_a_logreg.joblib",
    "models/baseline_b_logreg.joblib",
    "models/baseline_c_logreg.joblib",
    "reports/readmission_baselines_metrics.json",
    "reports/readmission_calibration_metrics.json",
    "reports/track_a_readmission_audit.md",
    "reports/data_audit/uci_diabetes_audit_summary.json",
    "reports/milestone_2_1_evaluation_integrity.md",
    
    # Track B Models & Reports
    "models/sbdh_transformer/pytorch_model.bin",
    "models/sbdh_transformer/config.json",
    "models/sbdh_transformer/tokenizer.json",
    "models/sbdh_transformer/tokenizer_config.json",
    "models/sbdh_transformer/training_curves.json",
    "evernorth_m3_results/models/sbdh_transformer/best_model.pt",
    "evernorth_m3_results/models/sbdh_transformer/config.json",
    "evernorth_m3_results/models/sbdh_transformer/tokenizer.json",
    "evernorth_m3_results/models/sbdh_transformer/tokenizer_config.json",
    "evernorth_m3_results/reports/sbdh_transformer_metrics.json",
    "evernorth_m3_results/reports/validation_threshold_config.json",
    "evernorth_m3_results/reports/unresolved_alignment_audit.csv",
    "evernorth_m3_results/reports/per_category_metrics.csv",
    "reports/sbdh_rules_metrics.json",
    "reports/sbdh_rules_metrics_corrected.json",
    "reports/sbdh_transformer_metrics.json",
    "reports/sbdh_transformer_metrics_corrected.json",
    "reports/sbdh_colab_transformer_metrics_corrected.json",
    "reports/final_evidence_consistency_audit.md",
    
    # Notebook & Scripts
    "notebooks/02_sbdh_transformer.ipynb",
    "scripts/validate_colab_notebook.py",
    "scripts/generate_colab_notebook.py",
    "scripts/audit_track_a_readmission.py",
    "scripts/evaluate_colab_checkpoint.py",
    "scripts/run_full_alignment_audit.py",
    "src/data_loaders.py",
    "src/audit_uci.py",
    "src/train_readmission_baseline.py",
    "src/span_alignment.py",
    "src/transformer_dataset.py",
    "src/transformer_model.py",
    "src/evaluate_sbdh_transformer.py",
    "src/sbdh_rules_baseline.py",
    "requirements.txt",
]

def hash_file(path):
    if not os.path.exists(path):
        return None, None
    size = os.path.getsize(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return size, h.hexdigest()

results = {}
for p in ARTIFACTS:
    size, sha = hash_file(p)
    results[p] = {"size_bytes": size, "sha256": sha, "exists": (size is not None)}

print(json.dumps(results, indent=2))
