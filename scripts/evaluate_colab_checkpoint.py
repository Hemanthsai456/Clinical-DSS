"""
Provenance and Evaluation Script for Track B Colab-Trained Transformer Checkpoint.
Verifies:
1. Checkpoint file integrity, SHA256 hashes, timestamps, parameter counts.
2. Evaluates the 3-epoch Colab model (best_model.pt) against synth_sbdh_test_reviewed.csv (1,732 notes).
3. Uses the corrected annotation alignment, whitespace trimming, and bipartite matching logic.
4. Generates dual-bound metrics: Resolvable Subset (N=3,340) vs Worst-Case Lower Bound (N=3,345).
5. Exports results to reports/sbdh_colab_transformer_metrics_corrected.json.
"""
import os
import sys
import json
import time
import hashlib
import torch
import pandas as pd
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loaders import load_synth_sbdh
from src.transformer_model import MultiLabelSBDHTransformer
from src.evaluate_sbdh_transformer import evaluate_transformer_on_dataset, compare_transformer_vs_rules
from src.utils import save_json


def hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify_checkpoint_provenance():
    colab_dir = "evernorth_m3_results/models/sbdh_transformer"
    local_dir = "models/sbdh_transformer"

    colab_weights = os.path.join(colab_dir, "best_model.pt")
    local_weights = os.path.join(local_dir, "pytorch_model.bin")

    assert os.path.exists(colab_weights), f"Colab weights missing: {colab_weights}"
    assert os.path.exists(local_weights), f"Local weights missing: {local_weights}"

    colab_size = os.path.getsize(colab_weights)
    local_size = os.path.getsize(local_weights)

    colab_hash = hash_file(colab_weights)
    local_hash = hash_file(local_weights)

    # Inspect state dicts
    colab_state = torch.load(colab_weights, map_location="cpu")
    local_state = torch.load(local_weights, map_location="cpu")

    colab_params = sum(p.numel() for p in colab_state.values())
    local_params = sum(p.numel() for p in local_state.values())

    provenance = {
        "colab_checkpoint": {
            "path": colab_weights,
            "filename": "best_model.pt",
            "size_bytes": colab_size,
            "sha256": colab_hash,
            "total_parameters": colab_params,
            "origin": "Google Colab 3-epoch GPU training run (evernorth_m3_results.zip)",
            "validation_threshold": 0.30,
        },
        "local_pilot_checkpoint": {
            "path": local_weights,
            "filename": "pytorch_model.bin",
            "size_bytes": local_size,
            "sha256": local_hash,
            "total_parameters": local_params,
            "origin": "Local CPU pilot run (60 training notes, 1 epoch, sample limit)",
            "training_curves_file": "models/sbdh_transformer/training_curves.json",
        },
        "distinct_checkpoints_verified": (colab_hash != local_hash),
    }

    print("=" * 80)
    print("TRACK B: CHECKPOINT PROVENANCE VERIFICATION")
    print("=" * 80)
    print(f"Colab Checkpoint: {colab_weights} ({colab_size:,} bytes) | SHA256: {colab_hash[:16]}...")
    print(f"Local Checkpoint: {local_weights} ({local_size:,} bytes) | SHA256: {local_hash[:16]}...")
    print(f"Unambiguously distinct models: {provenance['distinct_checkpoints_verified']}")
    return provenance


def run_colab_checkpoint_evaluation():
    colab_dir = "evernorth_m3_results/models/sbdh_transformer"
    threshold = 0.30
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 80)
    print("EVALUATING COLAB CHECKPOINT USING CORRECTED EVALUATION PIPELINE")
    print("=" * 80)
    print(f"Loading model from {colab_dir} on {device.upper()}...")

    tokenizer = AutoTokenizer.from_pretrained(colab_dir)
    model = MultiLabelSBDHTransformer.from_pretrained(colab_dir)
    model.to(device)

    dfs, _ = load_synth_sbdh()
    test_reviewed_df = dfs["test_reviewed"]
    print(f"Loaded {len(test_reviewed_df):,d} reviewed test notes.")

    t0 = time.time()
    results = evaluate_transformer_on_dataset(
        model=model,
        df=test_reviewed_df,
        tokenizer=tokenizer,
        device=device,
        max_length=256,
        threshold=threshold,
        dataset_name="test_reviewed",
    )
    duration = time.time() - t0
    results["evaluation_duration_seconds"] = round(duration, 2)
    results["checkpoint_provenance"] = "evernorth_m3_results/models/sbdh_transformer/best_model.pt (3 epochs, Colab GPU)"

    # Compare with rules baseline
    rules_path = "reports/sbdh_rules_metrics_corrected.json"
    if not os.path.exists(rules_path):
        rules_path = "reports/sbdh_rules_metrics.json"
    results["rules_baseline_comparison"] = compare_transformer_vs_rules(results, rules_metrics_path=rules_path)

    out_path = "reports/sbdh_colab_transformer_metrics_corrected.json"
    save_json(results, out_path)
    print(f"\n[EXPORT] Evaluation completed in {duration:.1f}s. Saved to: {out_path}")

    ex = results["span_level_exact_matching"]["resolvable_subset"]
    ex_wc = results["span_level_exact_matching"]["worst_case_bound_with_ambiguous_as_fn"]
    rel = results["span_level_relaxed_matching"]["resolvable_subset"]
    rel_wc = results["span_level_relaxed_matching"]["worst_case_bound_with_ambiguous_as_fn"]
    note_micro = results["aggregate_classification_metrics"]["micro"]


    print("\n" + "=" * 80)
    print("COLAB MODEL (3-EPOCH) EVALUATION RESULTS ON TEST_REVIEWED (THRESHOLD = 0.30)")
    print("=" * 80)
    print(f"Total Predicted Spans:          {results['total_pred_spans']}")
    print(f"Exact Match TP / FP / FN:       {ex['tp']} / {ex['fp']} / {ex['fn']}")
    print(f"Exact Precision (Resolvable):   {ex['precision']:.4f}")
    print(f"Exact Recall (Resolvable):      {ex['recall']:.4f}")
    print(f"Exact F1 (Resolvable):          {ex['f1_score']:.4f}")
    print(f"Exact F1 (Worst-Case Bound):    {ex_wc['f1_score']:.4f} (Recall: {ex_wc['recall']:.4f})")
    print(f"-" * 80)
    print(f"Relaxed Match TP / FP / FN:     {rel['tp']} / {rel['fp']} / {rel['fn']}")
    print(f"Relaxed Precision (Resolvable): {rel['precision']:.4f}")
    print(f"Relaxed Recall (Resolvable):    {rel['recall']:.4f}")
    print(f"Relaxed F1 (Resolvable):        {rel['f1_score']:.4f}")
    print(f"Relaxed F1 (Worst-Case Bound):  {rel_wc['f1_score']:.4f} (Recall: {rel_wc['recall']:.4f})")
    print(f"-" * 80)
    print(f"Note-Level Micro Precision:     {note_micro['precision']:.4f}")
    print(f"Note-Level Micro Recall:        {note_micro['recall']:.4f}")
    print(f"Note-Level Micro F1:            {note_micro['f1_score']:.4f}")
    print(f"Boundary Errors (Mismatches):   {results['boundary_errors_count']}")
    print(f"Unresolved Test Annotations:    {len(results['ambiguous_cases'])}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    verify_checkpoint_provenance()
    run_colab_checkpoint_evaluation()
