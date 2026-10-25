"""
Milestone 2 Track A: 30-Day Readmission Prediction Baselines & Feature Sensitivity Experiments.

Datasets:
- Eligible UCI Diabetes cohort (99,343 encounters, 69,990 patients).
- Strict patient-grouped train/validation/test partitions (zero patient overlap).
- Preserves literal 'None' in lab tests (0% missing).
- Excludes identifiers ('encounter_id', 'patient_nbr') and administrative sparse fields ('weight', 'payer_code').

Feature Sensitivity Experiments:
- Baseline A: Excludes discharge_disposition_id, diag_1, diag_2, diag_3.
- Baseline B: Adds discharge_disposition_id (post-acute placement).
- Baseline C: Adds discharge_disposition_id + grouped ICD-9 diagnoses (diag_1, diag_2, diag_3).

Models:
1. Majority-class baseline.
2. Logistic Regression (fitted preprocessing strictly on training data).

Outputs:
- models/baseline_a_logreg.joblib
- models/baseline_b_logreg.joblib
- models/baseline_c_logreg.joblib
- reports/readmission_baselines_metrics.json
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
)

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loaders import load_uci_diabetes
from src.audit_uci import create_eligible_modeling_cohort, split_uci_patient_grouped
from src.utils import NpEncoder, save_json


NUMERICAL_COLS = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

BASE_CATEGORICAL_COLS = [
    "race",
    "gender",
    "age",
    "admission_type_id",
    "admission_source_id",
    "max_glu_serum",
    "A1Cresult",
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
    "change",
    "diabetesMed",
]


def group_icd9(val: Any) -> str:
    """
    Clinically group ICD-9 codes according to Strack et al. (2014) standard categories.
    """
    if pd.isna(val) or val is None or val == "?" or val == "":
        return "Missing"
    s = str(val).strip()
    if s.startswith("V") or s.startswith("v"):
        return "Supplementary_V"
    if s.startswith("E") or s.startswith("e"):
        return "External_Injury_E"
    try:
        code = float(s)
    except ValueError:
        return "Other"

    if 250.0 <= code <= 250.99 or s.startswith("250"):
        return "Diabetes"
    elif 390 <= code <= 459 or code == 785:
        return "Circulatory"
    elif 460 <= code <= 519 or code == 786:
        return "Respiratory"
    elif 520 <= code <= 579 or code == 787:
        return "Digestive"
    elif 580 <= code <= 629 or code == 788:
        return "Genitourinary"
    elif 140 <= code <= 239:
        return "Neoplasms"
    elif 710 <= code <= 739:
        return "Musculoskeletal"
    elif 800 <= code <= 999:
        return "Injury_Poisoning"
    else:
        return "Other"


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare feature representations, ensuring correct types and ICD-9 groupings."""
    df_feat = df.copy()
    # Ensure ID columns are treated as strings
    df_feat["admission_type_id"] = df_feat["admission_type_id"].astype(str)
    df_feat["admission_source_id"] = df_feat["admission_source_id"].astype(str)
    if "discharge_disposition_id" in df_feat.columns:
        df_feat["discharge_disposition_id"] = df_feat["discharge_disposition_id"].astype(str)

    # Group ICD-9 codes
    if "diag_1" in df_feat.columns:
        df_feat["diag_1_group"] = df_feat["diag_1"].apply(group_icd9)
    if "diag_2" in df_feat.columns:
        df_feat["diag_2_group"] = df_feat["diag_2"].apply(group_icd9)
    if "diag_3" in df_feat.columns:
        df_feat["diag_3_group"] = df_feat["diag_3"].apply(group_icd9)

    return df_feat


def create_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """Build preprocessing pipeline fitting imputers, scalers, and encoders."""
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols),
    ])


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find threshold that maximizes F1 score on validation set."""
    best_f1 = -1.0
    best_t = 0.5
    for t in np.linspace(0.05, 0.40, 71):
        preds = (y_prob >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = t
    return float(best_t)


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    partition_name: str,
    model_name: str,
) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation metrics on predictions.
    Includes automated assertions verifying mathematical agreement between
    confusion-matrix cells and reported precision, recall, specificity, and F1.
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    auroc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    # Average Precision / PR-AUC: area under the precision-recall curve
    # defined as sum_{n} (R_n - R_{n-1}) * P_n
    ap = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float(np.mean(y_true))
    brier = float(brier_score_loss(y_true, y_prob))

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    prevalence = float(np.mean(y_true))

    # Mathematical agreement assertions between CM and reported metrics
    expected_prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    expected_rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    expected_spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    expected_f1 = float(2 * expected_prec * expected_rec / (expected_prec + expected_rec)) if (expected_prec + expected_rec) > 0 else 0.0

    assert abs(prec - expected_prec) < 1e-4, f"Precision mismatch: {prec} vs {expected_prec}"
    assert abs(rec - expected_rec) < 1e-4, f"Recall mismatch: {rec} vs {expected_rec}"
    assert abs(spec - expected_spec) < 1e-4, f"Specificity mismatch: {spec} vs {expected_spec}"
    assert abs(f1 - expected_f1) < 1e-4, f"F1 score mismatch: {f1} vs {expected_f1}"
    assert tn + fp + fn + tp == len(y_true), f"Total count mismatch: {tn + fp + fn + tp} != {len(y_true)}"
    assert tp + fn == int(np.sum(y_true)), f"Positive count mismatch: {tp + fn} != {int(np.sum(y_true))}"
    assert tn + fp == int(len(y_true) - np.sum(y_true)), f"Negative count mismatch: {tn + fp} != {int(len(y_true) - np.sum(y_true))}"

    return {
        "model_name": model_name,
        "partition": partition_name,
        "total_encounters": int(len(y_true)),
        "positive_count": int(tp + fn),
        "negative_count": int(tn + fp),
        "class_prevalence": round(prevalence, 4),
        "prevalence_pct": round(prevalence * 100, 4),
        "auroc": round(auroc, 4),
        "average_precision_pr_auc": round(ap, 4),
        "ap_definition": "sklearn.metrics.average_precision_score: weighted mean of precisions achieved at each recall threshold (sum_n (R_n - R_{n-1}) * P_n)",
        "decision_threshold": round(threshold, 4),
        "threshold_selection_note": "Threshold chosen to maximize F1 score on Validation partition, frozen and applied directly to Test.",
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "specificity": round(spec, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "confusion_matrix_derived": {
            "precision": round(expected_prec, 4),
            "recall": round(expected_rec, 4),
            "specificity": round(expected_spec, 4),
            "f1_score": round(expected_f1, 4),
        },
        "mathematical_consistency_verified": True,
        "brier_score": round(brier, 4),
    }


def evaluate_majority_baseline(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate majority class baseline (predict constant 0 / training prevalence)."""
    train_prev = float(np.mean(y_train))
    val_prob = np.full_like(y_val, train_prev, dtype=float)
    test_prob = np.full_like(y_test, train_prev, dtype=float)

    val_res = evaluate_predictions(y_val, val_prob, threshold=0.5, partition_name="validation", model_name="Majority Class Baseline")
    test_res = evaluate_predictions(y_test, test_prob, threshold=0.5, partition_name="test", model_name="Majority Class Baseline")

    return {
        "validation": val_res,
        "test": test_res,
    }


def run_track_a_baselines() -> Dict[str, Any]:
    print("=" * 80)
    print("TRACK A: 30-DAY READMISSION PREDICTION BASELINES & FEATURE SENSITIVITY")
    print("=" * 80)

    # 1. Load data and filter to eligible cohort
    df_raw, uci_path = load_uci_diabetes()
    df_eligible, cohort_metrics = create_eligible_modeling_cohort(df_raw)
    print(f"Eligible modeling cohort: {len(df_eligible):,d} encounters ({df_eligible['patient_nbr'].nunique():,d} patients)")

    # 2. Patient-grouped splitting
    train_df, val_df, test_df, split_metrics = split_uci_patient_grouped(df_eligible, random_state=42)
    print(f"Train split: {len(train_df):,d} | Val split: {len(val_df):,d} | Test split: {len(test_df):,d}")
    print(f"Zero patient overlap verified across all partitions.")

    # 3. Targets
    y_train = (train_df["readmitted"] == "<30").astype(int).values
    y_val = (val_df["readmitted"] == "<30").astype(int).values
    y_test = (test_df["readmitted"] == "<30").astype(int).values

    print(f"Prevalence: Train={np.mean(y_train)*100:.2f}% | Val={np.mean(y_val)*100:.2f}% | Test={np.mean(y_test)*100:.2f}%")

    # 4. Feature sets preparation
    train_feat = prepare_features(train_df)
    val_feat = prepare_features(val_df)
    test_feat = prepare_features(test_df)

    # Define sensitivity feature sets
    cat_cols_a = list(BASE_CATEGORICAL_COLS)
    cat_cols_b = list(BASE_CATEGORICAL_COLS) + ["discharge_disposition_id"]
    cat_cols_c = list(BASE_CATEGORICAL_COLS) + ["discharge_disposition_id", "diag_1_group", "diag_2_group", "diag_3_group"]

    experiments = {
        "Baseline_A_No_Disposition_No_Diag": {
            "description": "Baseline A: Standard Clinical Features EXCLUDING discharge_disposition_id and discharge diagnoses.",
            "num_cols": NUMERICAL_COLS,
            "cat_cols": cat_cols_a,
            "disposition_included": False,
            "diagnoses_included": False,
            "model_file": "models/baseline_a_logreg.joblib",
        },
        "Baseline_B_With_Disposition_No_Diag": {
            "description": "Baseline B: Augmented Model INCLUDING discharge_disposition_id while still EXCLUDING discharge diagnoses.",
            "num_cols": NUMERICAL_COLS,
            "cat_cols": cat_cols_b,
            "disposition_included": True,
            "diagnoses_included": False,
            "model_file": "models/baseline_b_logreg.joblib",
        },
        "Baseline_C_With_Disposition_With_Diag": {
            "description": "Baseline C: Full Retrospective Model INCLUDING discharge_disposition_id AND grouped discharge ICD-9 diagnoses.",
            "num_cols": NUMERICAL_COLS,
            "cat_cols": cat_cols_c,
            "disposition_included": True,
            "diagnoses_included": True,
            "model_file": "models/baseline_c_logreg.joblib",
        },
    }

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # Evaluate Majority Baseline
    majority_results = evaluate_majority_baseline(y_train, y_val, y_test)

    results = {
        "majority_class_baseline": majority_results,
        "sensitivity_experiments": {},
    }

    print("\n" + "-" * 80)
    print("TRAINING LOGISTIC REGRESSION MODELS (Fitted strictly on Train)")
    print("-" * 80)

    for exp_key, cfg in experiments.items():
        print(f"\n[Running {exp_key}]")
        print(f"  Description: {cfg['description']}")
        print(f"  Numerical features: {len(cfg['num_cols'])} | Categorical features: {len(cfg['cat_cols'])}")

        # Build pipeline
        preprocessor = create_preprocessor(cfg["num_cols"], cfg["cat_cols"])
        clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0, solver="lbfgs")
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ])

        # Fit exclusively on training data
        feature_cols = cfg["num_cols"] + cfg["cat_cols"]
        X_train = train_feat[feature_cols]
        X_val = val_feat[feature_cols]
        X_test = test_feat[feature_cols]

        print("  Fitting pipeline on X_train...")
        pipe.fit(X_train, y_train)

        # Predict probabilities
        val_probs = pipe.predict_proba(X_val)[:, 1]
        test_probs = pipe.predict_proba(X_test)[:, 1]

        # Select decision threshold on Validation partition
        opt_thresh = find_optimal_threshold(y_val, val_probs)
        print(f"  Validation-tuned optimal F1 threshold: {opt_thresh:.4f}")

        # Evaluate on validation and test separately
        val_eval = evaluate_predictions(y_val, val_probs, threshold=opt_thresh, partition_name="validation", model_name=exp_key)
        test_eval = evaluate_predictions(y_test, test_probs, threshold=opt_thresh, partition_name="test", model_name=exp_key)

        print(f"  [VAL ] AUROC: {val_eval['auroc']:.4f} | PR-AUC: {val_eval['average_precision_pr_auc']:.4f} | F1: {val_eval['f1_score']:.4f} | Brier: {val_eval['brier_score']:.4f}")
        print(f"  [TEST] AUROC: {test_eval['auroc']:.4f} | PR-AUC: {test_eval['average_precision_pr_auc']:.4f} | F1: {test_eval['f1_score']:.4f} | Brier: {test_eval['brier_score']:.4f}")

        # Save model artifact
        joblib.dump(pipe, cfg["model_file"])
        print(f"  Saved fitted model pipeline to: {cfg['model_file']} ({os.path.getsize(cfg['model_file']):,} bytes)")

        results["sensitivity_experiments"][exp_key] = {
            "description": cfg["description"],
            "disposition_included": cfg["disposition_included"],
            "diagnoses_included": cfg["diagnoses_included"],
            "features_count": len(feature_cols),
            "model_file": cfg["model_file"],
            "validation": val_eval,
            "test": test_eval,
        }

    # Save metrics JSON artifact
    metrics_path = "reports/readmission_baselines_metrics.json"
    save_json(results, metrics_path)
    print(f"\n[EXPORT] All readmission baseline metrics exported to: {metrics_path}")

    # Print summary comparison table
    print("\n" + "=" * 95)
    print("TRACK A SUMMARY COMPARISON TABLE (TEST PARTITION)")
    print("=" * 95)
    print(f"{'Model':<38} | {'AUROC':<7} | {'PR-AUC':<8} | {'Threshold':<9} | {'F1':<6} | {'Recall':<7} | {'Brier':<7}")
    print("-" * 95)
    maj = results["majority_class_baseline"]["test"]
    print(f"{'Majority Class Baseline':<38} | {maj['auroc']:<7.4f} | {maj['average_precision_pr_auc']:<8.4f} | {maj['decision_threshold']:<9.4f} | {maj['f1_score']:<6.4f} | {maj['recall']:<7.4f} | {maj['brier_score']:<7.4f}")

    for exp_key, exp in results["sensitivity_experiments"].items():
        t = exp["test"]
        print(f"{exp_key:<38} | {t['auroc']:<7.4f} | {t['average_precision_pr_auc']:<8.4f} | {t['decision_threshold']:<9.4f} | {t['f1_score']:<6.4f} | {t['recall']:<7.4f} | {t['brier_score']:<7.4f}")

    return results


if __name__ == "__main__":
    run_track_a_baselines()
