"""
Script to create a timestamped baseline freeze archive containing:
- Baseline manifest and reproduction guide
- Audit reports and metrics JSONs
- Configuration files and training curves
- Checkpoint provenance hashes
(Excludes huge binary weights and raw datasets to keep archive lightweight and portable).
"""
import os
import zipfile
import hashlib

ARCHIVE_PATH = "reports/baseline_freeze_archive_20261026.zip"

FILES_TO_INCLUDE = [
    # Manifest & Guides
    ("reports/baseline_freeze_manifest.md", "reports/baseline_freeze_manifest.md"),
    ("reports/reproduction_guide.md", "reports/reproduction_guide.md"),
    ("reports/final_evidence_consistency_audit.md", "reports/final_evidence_consistency_audit.md"),
    ("reports/track_a_readmission_audit.md", "reports/track_a_readmission_audit.md"),
    ("reports/milestone_3_report.md", "reports/milestone_3_report.md"),
    ("reports/milestone_2_1_evaluation_integrity.md", "reports/milestone_2_1_evaluation_integrity.md"),
    ("reports/milestone_2_report.md", "reports/milestone_2_report.md"),
    
    # Track A Metrics & Audits
    ("reports/readmission_baselines_metrics.json", "reports/readmission_baselines_metrics.json"),
    ("reports/readmission_calibration_metrics.json", "reports/readmission_calibration_metrics.json"),
    ("reports/data_audit/uci_diabetes_audit_summary.json", "reports/data_audit/uci_diabetes_audit_summary.json"),
    ("reports/data_audit/modeling_readiness_report.md", "reports/data_audit/modeling_readiness_report.md"),
    
    # Track B Metrics & Audits
    ("reports/sbdh_rules_metrics.json", "reports/sbdh_rules_metrics.json"),
    ("reports/sbdh_rules_metrics_corrected.json", "reports/sbdh_rules_metrics_corrected.json"),
    ("reports/sbdh_transformer_metrics.json", "reports/sbdh_transformer_metrics.json"),
    ("reports/sbdh_transformer_metrics_corrected.json", "reports/sbdh_transformer_metrics_corrected.json"),
    ("reports/sbdh_colab_transformer_metrics_corrected.json", "reports/sbdh_colab_transformer_metrics_corrected.json"),
    ("evernorth_m3_results/reports/sbdh_transformer_metrics.json", "evernorth_m3_results/reports/sbdh_transformer_metrics.json"),
    ("evernorth_m3_results/reports/validation_threshold_config.json", "evernorth_m3_results/reports/validation_threshold_config.json"),
    ("evernorth_m3_results/reports/unresolved_alignment_audit.csv", "evernorth_m3_results/reports/unresolved_alignment_audit.csv"),
    ("evernorth_m3_results/reports/per_category_metrics.csv", "evernorth_m3_results/reports/per_category_metrics.csv"),
    ("evernorth_m3_results/reports/milestone_3_colab_report.md", "evernorth_m3_results/reports/milestone_3_colab_report.md"),
    
    # Model Configurations & Tokenizer Metadata (Excluding large .bin/.pt files)
    ("models/sbdh_transformer/config.json", "models/sbdh_transformer/config.json"),
    ("models/sbdh_transformer/training_curves.json", "models/sbdh_transformer/training_curves.json"),
    ("models/sbdh_transformer/tokenizer_config.json", "models/sbdh_transformer/tokenizer_config.json"),
    ("evernorth_m3_results/models/sbdh_transformer/config.json", "evernorth_m3_results/models/sbdh_transformer/config.json"),
    ("evernorth_m3_results/models/sbdh_transformer/tokenizer_config.json", "evernorth_m3_results/models/sbdh_transformer/tokenizer_config.json"),
    
    # Dependencies
    ("requirements.txt", "requirements.txt"),
]

def main():
    print(f"Creating baseline archive: {ARCHIVE_PATH}...")
    with zipfile.ZipFile(ARCHIVE_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arcname in FILES_TO_INCLUDE:
            assert os.path.exists(src), f"Missing file: {src}"
            zf.write(src, arcname=arcname)
            print(f"  Added: {arcname} ({os.path.getsize(src):,} bytes)")

    size = os.path.getsize(ARCHIVE_PATH)
    h = hashlib.sha256()
    with open(ARCHIVE_PATH, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)

    print(f"\n[ARCHIVE COMPLETE] Successfully archived {len(FILES_TO_INCLUDE)} files to {ARCHIVE_PATH}")
    print(f"Size: {size:,} bytes | SHA256: {h.hexdigest()}")

if __name__ == "__main__":
    main()
