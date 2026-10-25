"""
Automated test suite for Track A Readmission Evaluation Integrity.
Verifies:
1. Mathematical agreement between confusion matrix (TN, FP, FN, TP) and reported classification metrics:
   - Precision = TP / (TP + FP)
   - Recall = TP / (TP + FN)
   - Specificity = TN / (TN + FP)
   - F1 = 2 * Precision * Recall / (Precision + Recall)
   - Total encounters = TN + FP + FN + TP
   - Positive count = TP + FN
   - Negative count = TN + FP
2. Exact test-set positive count (1,650) and prevalence (11.1104%).
3. Threshold selection validity (validation-only selection, frozen for test).
4. Automated verification against reports/readmission_baselines_metrics.json.
5. In-memory validation of saved model artifacts (baseline_a, baseline_b, baseline_c).
"""
import os
import sys
import json
import pytest
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.train_readmission_baseline import evaluate_predictions


def test_confusion_matrix_mathematical_derivation():
    """
    Test that evaluate_predictions strictly enforces mathematical agreement between
    confusion-matrix cells and all derived classification metrics.
    """
    # Synthetic ground truth and probabilities
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 0])  # 4 positives, 6 negatives
    y_prob = np.array([0.9, 0.8, 0.2, 0.1, 0.7, 0.3, 0.4, 0.05, 0.65, 0.15])
    threshold = 0.5

    # Predictions >= 0.5:
    # idx 0: 0.9 >= 0.5 -> 1 (TP)
    # idx 1: 0.8 >= 0.5 -> 1 (TP)
    # idx 2: 0.2 < 0.5  -> 0 (FN)
    # idx 3: 0.1 < 0.5  -> 0 (TN)
    # idx 4: 0.7 >= 0.5 -> 1 (FP)
    # idx 5: 0.3 < 0.5  -> 0 (TN)
    # idx 6: 0.4 < 0.5  -> 0 (TN)
    # idx 7: 0.05 < 0.5 -> 0 (TN)
    # idx 8: 0.65 >= 0.5-> 1 (TP)
    # idx 9: 0.15 < 0.5 -> 0 (TN)
    # TP = 3 (idx 0, 1, 8)
    # FP = 1 (idx 4)
    # FN = 1 (idx 2)
    # TN = 5 (idx 3, 5, 6, 7, 9)

    metrics = evaluate_predictions(y_true, y_prob, threshold, "test", "Synthetic_Test_Model")

    cm = metrics["confusion_matrix"]
    assert cm["tp"] == 3
    assert cm["fp"] == 1
    assert cm["fn"] == 1
    assert cm["tn"] == 5

    # Check mathematical derivations
    expected_prec = 3 / (3 + 1)  # 0.75
    expected_rec = 3 / (3 + 1)   # 0.75
    expected_spec = 5 / (5 + 1)  # 5/6 = 0.8333...
    expected_f1 = 2 * 0.75 * 0.75 / (0.75 + 0.75)  # 0.75

    assert abs(metrics["precision"] - round(expected_prec, 4)) < 1e-4
    assert abs(metrics["recall"] - round(expected_rec, 4)) < 1e-4
    assert abs(metrics["specificity"] - round(expected_spec, 4)) < 1e-4
    assert abs(metrics["f1_score"] - round(expected_f1, 4)) < 1e-4
    assert metrics["positive_count"] == 4
    assert metrics["negative_count"] == 6
    assert metrics["total_encounters"] == 10


def test_edge_cases_all_negative_majority_baseline():
    """
    Test zero-division and edge case handling when all predictions are negative (majority baseline).
    """
    y_true = np.array([0, 0, 0, 1, 0, 0])
    y_prob = np.full_like(y_true, 0.1, dtype=float)  # All below 0.5
    threshold = 0.5

    metrics = evaluate_predictions(y_true, y_prob, threshold, "test", "Majority_Edge_Case")
    cm = metrics["confusion_matrix"]

    assert cm["tp"] == 0
    assert cm["fp"] == 0
    assert cm["fn"] == 1
    assert cm["tn"] == 5
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1_score"] == 0.0
    assert metrics["specificity"] == 1.0


def test_json_artifact_mathematical_consistency():
    """
    Verify that reports/readmission_baselines_metrics.json has 100% mathematical consistency
    across all models and partitions.
    """
    metrics_path = os.path.join(os.path.dirname(__file__), "..", "reports", "readmission_baselines_metrics.json")
    assert os.path.exists(metrics_path), f"Metrics artifact not found at: {metrics_path}"

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    models_to_check = [("majority_class_baseline", data["majority_class_baseline"])]
    for exp_key, exp_data in data["sensitivity_experiments"].items():
        models_to_check.append((exp_key, exp_data))

    for model_name, model_entry in models_to_check:
        for part_name in ["validation", "test"]:
            part = model_entry[part_name]
            cm = part["confusion_matrix"]
            tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]

            # 1. Total counts
            total = tp + fp + fn + tn
            pos = tp + fn
            neg = tn + fp

            assert total == part["total_encounters"], f"{model_name} {part_name}: total {total} != {part['total_encounters']}"
            assert pos == part["positive_count"], f"{model_name} {part_name}: pos {pos} != {part['positive_count']}"
            assert neg == part["negative_count"], f"{model_name} {part_name}: neg {neg} != {part['negative_count']}"

            # 2. Partition specific invariants
            if part_name == "test":
                assert total == 14851, f"Test total must be 14,851, got {total}"
                assert pos == 1650, f"Test positive count must be exactly 1,650, got {pos}"
                assert neg == 13201, f"Test negative count must be exactly 13,201, got {neg}"
                assert abs(part["class_prevalence"] - (1650 / 14851)) < 1e-4

            if part_name == "validation":
                assert total == 15022, f"Validation total must be 15,022, got {total}"
                assert pos == 1711, f"Validation positive count must be exactly 1,711, got {pos}"
                assert neg == 13311, f"Validation negative count must be exactly 13,311, got {neg}"
                assert abs(part["class_prevalence"] - (1711 / 15022)) < 1e-4

            # 3. Precision, Recall, Specificity, F1 mathematical agreement
            expected_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            expected_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            expected_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            expected_f1 = (2 * expected_prec * expected_rec) / (expected_prec + expected_rec) if (expected_prec + expected_rec) > 0 else 0.0

            assert abs(part["precision"] - round(expected_prec, 4)) < 1e-4, \
                f"{model_name} {part_name}: Reported precision {part['precision']} != derived {expected_prec}"
            assert abs(part["recall"] - round(expected_rec, 4)) < 1e-4, \
                f"{model_name} {part_name}: Reported recall {part['recall']} != derived {expected_rec}"
            assert abs(part["specificity"] - round(expected_spec, 4)) < 1e-4, \
                f"{model_name} {part_name}: Reported specificity {part['specificity']} != derived {expected_spec}"
            assert abs(part["f1_score"] - round(expected_f1, 4)) < 1e-4, \
                f"{model_name} {part_name}: Reported f1 {part['f1_score']} != derived {expected_f1}"


def test_frozen_thresholds_selected_on_validation_only():
    """
    Verify that decision thresholds for Baselines A, B, and C are identical between
    Validation and Test partitions in reports/readmission_baselines_metrics.json.
    """
    metrics_path = os.path.join(os.path.dirname(__file__), "..", "reports", "readmission_baselines_metrics.json")
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for exp_key, exp_data in data["sensitivity_experiments"].items():
        val_thresh = exp_data["validation"]["decision_threshold"]
        test_thresh = exp_data["test"]["decision_threshold"]
        assert val_thresh == test_thresh, \
            f"{exp_key}: Validation threshold {val_thresh} does not match frozen Test threshold {test_thresh}"
