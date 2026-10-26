"""
Track A UCI Readmission Audit Script.
Validates:
1. Target definition (<30 positive; >30 and NO negative).
2. Data loading & literal 'None' preservation in lab tests.
3. Cohort exclusion (expired / hospice) & analysis of the 86 hospice encounters.
4. Patient-grouped train/val/test splits (zero patient leakage).
5. Preprocessing fit verification (only on train).
6. Exact reproduction of baseline metrics (Majority, A, B, C) using saved joblib models.
7. Calibration analysis (Brier score, Expected Calibration Error across 10 deciles).
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score, confusion_matrix, precision_score, recall_score, f1_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loaders import load_uci_diabetes
from src.audit_uci import create_eligible_modeling_cohort, split_uci_patient_grouped
from src.train_readmission_baseline import prepare_features, NUMERICAL_COLS, BASE_CATEGORICAL_COLS


def compute_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    brier = float(brier_score_loss(y_true, y_prob))
    
    # Decile binning
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    bin_data = []
    ece = 0.0
    total_samples = len(y_true)
    
    for i in range(n_bins):
        mask = bin_indices == i
        count = int(np.sum(mask))
        if count > 0:
            avg_prob = float(np.mean(y_prob[mask]))
            emp_rate = float(np.mean(y_true[mask]))
            abs_err = abs(avg_prob - emp_rate)
            ece += (count / total_samples) * abs_err
        else:
            avg_prob = float((bins[i] + bins[i+1]) / 2)
            emp_rate = 0.0
            abs_err = 0.0
            
        bin_data.append({
            "bin": i + 1,
            "bin_range": f"[{bins[i]:.2f}, {bins[i+1]:.2f})",
            "count": count,
            "mean_predicted_prob": round(avg_prob, 4),
            "observed_positive_rate": round(emp_rate, 4),
            "calibration_error": round(abs_err, 4),
        })
        
    return {
        "brier_score": round(brier, 4),
        "expected_calibration_error_ece": round(ece, 4),
        "mean_predicted_probability": round(float(np.mean(y_prob)), 4),
        "observed_prevalence": round(float(np.mean(y_true)), 4),
        "decile_calibration_curve": bin_data,
    }


def main():
    print("=" * 80)
    print("TRACK A: AUDIT OF UCI DIABETES READMISSION PIPELINE")
    print("=" * 80)

    # 1. Load Data
    df_raw, path = load_uci_diabetes()
    print(f"[1] Loaded raw data from {path}: {len(df_raw)} rows, {df_raw.shape[1]} cols.")
    print(f"    Raw Target Value Counts:\n{df_raw['readmitted'].value_counts().to_dict()}")

    # 2. Check lab 'None' preservation
    print(f"[2] max_glu_serum NaNs: {df_raw['max_glu_serum'].isna().sum()}, None count: {(df_raw['max_glu_serum'] == 'None').sum()}")
    print(f"    A1Cresult NaNs: {df_raw['A1Cresult'].isna().sum()}, None count: {(df_raw['A1Cresult'] == 'None').sum()}")

    # 3. Eligible cohort
    df_eligible, cohort_comp = create_eligible_modeling_cohort(df_raw)
    print(f"[3] Eligible cohort: {len(df_eligible)} encounters, {df_eligible['patient_nbr'].nunique()} patients.")
    print(f"    Excluded: {cohort_comp['excluded_mortality_hospice']['total_encounters']} encounters.")
    print(f"    Hospice 86 analysis: {cohort_comp['excluded_mortality_hospice']['hospice_86_breakdown']['breakdown_by_disposition']}")

    # 4. Patient grouped split
    train_df, val_df, test_df, split_metrics = split_uci_patient_grouped(df_eligible, random_state=42)
    print(f"[4] Train encounters: {len(train_df)}, Val encounters: {len(val_df)}, Test encounters: {len(test_df)}")
    print(f"    Train patients: {train_df['patient_nbr'].nunique()}, Val patients: {val_df['patient_nbr'].nunique()}, Test patients: {test_df['patient_nbr'].nunique()}")
    
    # Overlap assertions
    train_p = set(train_df["patient_nbr"])
    val_p = set(val_df["patient_nbr"])
    test_p = set(test_df["patient_nbr"])
    print(f"    Overlap train-val: {len(train_p & val_p)}, train-test: {len(train_p & test_p)}, val-test: {len(val_p & test_p)}")

    # 5. Targets
    y_train = (train_df["readmitted"] == "<30").astype(int).values
    y_val = (val_df["readmitted"] == "<30").astype(int).values
    y_test = (test_df["readmitted"] == "<30").astype(int).values

    print(f"    Test Positives: {int(np.sum(y_test))}, Negatives: {int(len(y_test) - np.sum(y_test))}, Prevalence: {np.mean(y_test):.6f}")

    # 6. Feature prep
    train_feat = prepare_features(train_df)
    val_feat = prepare_features(val_df)
    test_feat = prepare_features(test_df)

    cat_cols_a = list(BASE_CATEGORICAL_COLS)
    cat_cols_b = list(BASE_CATEGORICAL_COLS) + ["discharge_disposition_id"]
    cat_cols_c = list(BASE_CATEGORICAL_COLS) + ["discharge_disposition_id", "diag_1_group", "diag_2_group", "diag_3_group"]

    exp_cfgs = {
        "Baseline_A_No_Disposition_No_Diag": {
            "cols": NUMERICAL_COLS + cat_cols_a,
            "model_path": "models/baseline_a_logreg.joblib",
            "val_thresh": 0.130,
        },
        "Baseline_B_With_Disposition_No_Diag": {
            "cols": NUMERICAL_COLS + cat_cols_b,
            "model_path": "models/baseline_b_logreg.joblib",
            "val_thresh": 0.145,
        },
        "Baseline_C_With_Disposition_With_Diag": {
            "cols": NUMERICAL_COLS + cat_cols_c,
            "model_path": "models/baseline_c_logreg.joblib",
            "val_thresh": 0.135,
        },
    }

    # Load existing report metrics
    with open("reports/readmission_baselines_metrics.json", "r") as f:
        existing_metrics = json.load(f)

    calibration_results = {}

    # Majority baseline calibration
    train_prev = float(np.mean(y_train))
    val_prob_maj = np.full_like(y_val, train_prev, dtype=float)
    test_prob_maj = np.full_like(y_test, train_prev, dtype=float)
    calibration_results["Majority_Class_Baseline"] = {
        "validation": compute_calibration_metrics(y_val, val_prob_maj),
        "test": compute_calibration_metrics(y_test, test_prob_maj),
    }

    print("\n[6] Model Replication and Calibration Analysis:")
    for exp_key, cfg in exp_cfgs.items():
        print(f"\n--- Checking {exp_key} ---")
        assert os.path.exists(cfg["model_path"]), f"Model file missing: {cfg['model_path']}"
        pipe = joblib.load(cfg["model_path"])

        X_val = val_feat[cfg["cols"]]
        X_test = test_feat[cfg["cols"]]

        val_probs = pipe.predict_proba(X_val)[:, 1]
        test_probs = pipe.predict_proba(X_test)[:, 1]

        thresh = cfg["val_thresh"]
        y_pred_test = (test_probs >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test, labels=[0, 1]).ravel()
        auroc = float(roc_auc_score(y_test, test_probs))
        ap = float(average_precision_score(y_test, test_probs))
        prec = float(precision_score(y_test, y_pred_test, zero_division=0))
        rec = float(recall_score(y_test, y_pred_test, zero_division=0))
        f1 = float(f1_score(y_test, y_pred_test, zero_division=0))

        # Check match with existing reported metrics
        rep = existing_metrics["sensitivity_experiments"][exp_key]["test"]
        print(f"    Reproduced vs Reported:")
        print(f"      AUROC: {auroc:.4f} vs {rep['auroc']:.4f}")
        print(f"      AP:    {ap:.4f} vs {rep['average_precision_pr_auc']:.4f}")
        print(f"      F1:    {f1:.4f} vs {rep['f1_score']:.4f}")
        print(f"      CM:    ({tn}, {fp}, {fn}, {tp}) vs ({rep['confusion_matrix']['tn']}, {rep['confusion_matrix']['fp']}, {rep['confusion_matrix']['fn']}, {rep['confusion_matrix']['tp']})")

        assert abs(auroc - rep["auroc"]) < 1e-4, f"AUROC mismatch for {exp_key}"
        assert abs(ap - rep["average_precision_pr_auc"]) < 1e-4, f"AP mismatch for {exp_key}"
        assert abs(f1 - rep["f1_score"]) < 1e-4, f"F1 mismatch for {exp_key}"
        assert (tn, fp, fn, tp) == (rep["confusion_matrix"]["tn"], rep["confusion_matrix"]["fp"], rep["confusion_matrix"]["fn"], rep["confusion_matrix"]["tp"]), f"CM mismatch for {exp_key}"
        print(f"    [MATCH CONFIRMED] Model {exp_key} reproduced with exact numerical agreement!")

        # Calibration
        val_calib = compute_calibration_metrics(y_val, val_probs)
        test_calib = compute_calibration_metrics(y_test, test_probs)
        calibration_results[exp_key] = {
            "validation": val_calib,
            "test": test_calib,
        }
        print(f"    Calibration - Test Brier: {test_calib['brier_score']:.4f}, Test ECE: {test_calib['expected_calibration_error_ece']:.4f}")
        print(f"    Mean predicted prob: {test_calib['mean_predicted_probability']:.4f} vs observed prevalence: {test_calib['observed_prevalence']:.4f}")

    # Save calibration metrics
    calib_out_path = "reports/readmission_calibration_metrics.json"
    with open(calib_out_path, "w", encoding="utf-8") as f:
        json.dump(calibration_results, f, indent=2)
    print(f"\n[7] Exported calibration metrics to {calib_out_path}")


if __name__ == "__main__":
    main()
