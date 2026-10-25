"""
Comprehensive audit and validation logic for UCI Diabetes 130-US Hospitals dataset.
Profiles data quality, patient repeats, target distributions, leakage risks, and grouped splits.
Updated for Milestone 1.1 with:
1. Literal 'None' preservation in lab tests (0% missing).
2. Detailed investigation and policy rationale for the 86 hospice readmission encounters.
3. Explicit prediction-time assumptions and sensitivity analysis plan for discharge_disposition_id.
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from sklearn.model_selection import GroupShuffleSplit


# Fallback clinical discharge disposition codes indicating mortality or hospice
DEFAULT_EXPIRED_HOSPICE_DISPOSITION_IDS = [11, 13, 14, 19, 20, 21]

EXPECTED_TARGET_VALUES = {"<30", ">30", "NO"}

FEATURE_TEMPORAL_TAXONOMY = {
    "identifiers": [
        "encounter_id",
        "patient_nbr",
    ],
    "baseline_demographics": [
        "race",
        "gender",
        "age",
    ],
    "prior_utilization": [
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
    ],
    "admission_context": [
        "admission_type_id",
        "admission_source_id",
    ],
    "inpatient_treatment_and_labs": [
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_diagnoses",
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
    ],
    "discharge_and_coding": [
        "discharge_disposition_id",
        "diag_1",
        "diag_2",
        "diag_3",
    ],
    "administrative_or_heavily_missing": [
        "weight",
        "payer_code",
        "medical_specialty",
    ],
    "target": [
        "readmitted",
    ],
}


def parse_ids_mapping_dispositions(ids_path: str) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Dynamically parse discharge_disposition_id section from IDS_mapping.csv.
    Identifies any description mentioning 'expired' or 'hospice' (case-insensitive).
    """
    disposition_map = {}
    current_section = None
    with open(ids_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "discharge_disposition_id" in line:
                current_section = "disposition"
                continue
            elif "admission_source_id" in line or "admission_type_id" in line:
                current_section = None
                continue
            if current_section == "disposition":
                parts = line.split(",", 1)
                if len(parts) == 2 and parts[0].isdigit():
                    code = int(parts[0])
                    desc = parts[1].strip('" ')
                    disposition_map[code] = desc

    mortality_hospice_codes = {}
    for code, desc in disposition_map.items():
        desc_lower = desc.lower()
        if "expired" in desc_lower or "hospice" in desc_lower:
            mortality_hospice_codes[code] = desc

    return disposition_map, mortality_hospice_codes


def create_eligible_modeling_cohort(
    df: pd.DataFrame,
    ids_path: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Filter the raw UCI dataset into an eligible modeling cohort for 30-day readmission prediction.
    Excludes encounters with discharge disposition codes corresponding to in-hospital death or hospice.
    
    Returns:
        (df_eligible, comparison_metrics)
    """
    # Verify literal 'None' in lab tests is preserved as valid category, not NaN
    assert df["max_glu_serum"].isna().sum() == 0, "max_glu_serum contains NaNs; 'None' was incorrectly converted to missing!"
    assert "None" in df["max_glu_serum"].values, "'None' category missing from max_glu_serum!"
    assert df["A1Cresult"].isna().sum() == 0, "A1Cresult contains NaNs; 'None' was incorrectly converted to missing!"
    assert "None" in df["A1Cresult"].values, "'None' category missing from A1Cresult!"

    # Dynamically verify disposition codes if IDS_mapping.csv is supplied or found
    mortality_hospice_codes = {}
    if ids_path and os.path.isfile(ids_path):
        _, mortality_hospice_codes = parse_ids_mapping_dispositions(ids_path)
    
    if not mortality_hospice_codes:
        mortality_hospice_codes = {
            11: "Expired",
            13: "Hospice / home",
            14: "Hospice / medical facility",
            19: "Expired at home. Medicaid only, hospice.",
            20: "Expired in a medical facility. Medicaid only, hospice.",
            21: "Expired, place unknown. Medicaid only, hospice.",
        }

    excluded_codes = sorted(list(mortality_hospice_codes.keys()))

    # Raw metrics
    n_enc_raw = len(df)
    n_pat_raw = int(df["patient_nbr"].nunique())
    raw_targets = df["readmitted"].value_counts().to_dict()
    raw_pos = int((df["readmitted"] == "<30").sum())
    raw_pos_pct = round(raw_pos / n_enc_raw * 100, 2)

    # Exclusions
    mask_excluded = df["discharge_disposition_id"].isin(excluded_codes)
    df_excluded = df[mask_excluded].copy()
    n_enc_excluded = int(mask_excluded.sum())
    n_pat_excluded = int(df_excluded["patient_nbr"].nunique())
    excluded_targets = df_excluded["readmitted"].value_counts().to_dict()
    excluded_pos = int((df_excluded["readmitted"] == "<30").sum())
    excluded_pos_pct = round(excluded_pos / n_enc_excluded * 100, 2) if n_enc_excluded > 0 else 0.0

    # Detailed audit of the 86 hospice encounters with readmitted != 'NO'
    hospice_readmit_df = df_excluded[df_excluded["readmitted"].isin(["<30", ">30"])]
    hospice_crosstab = pd.crosstab(
        df_excluded["discharge_disposition_id"], df_excluded["readmitted"]
    ).to_dict()

    hospice_86_breakdown = {
        "total_readmitted_fatal_or_hospice": len(hospice_readmit_df),
        "breakdown_by_disposition": {
            "13 (Hospice / home)": {
                "<30": int(((df_excluded["discharge_disposition_id"] == 13) & (df_excluded["readmitted"] == "<30")).sum()),
                ">30": int(((df_excluded["discharge_disposition_id"] == 13) & (df_excluded["readmitted"] == ">30")).sum()),
                "total": int((df_excluded["discharge_disposition_id"] == 13).sum()),
            },
            "14 (Hospice / medical facility)": {
                "<30": int(((df_excluded["discharge_disposition_id"] == 14) & (df_excluded["readmitted"] == "<30")).sum()),
                ">30": int(((df_excluded["discharge_disposition_id"] == 14) & (df_excluded["readmitted"] == ">30")).sum()),
                "total": int((df_excluded["discharge_disposition_id"] == 14).sum()),
            },
            "11, 19, 20, 21 (Expired / In-Hospital Death)": {
                "<30": int((df_excluded["discharge_disposition_id"].isin([11, 19, 20, 21]) & (df_excluded["readmitted"] == "<30")).sum()),
                ">30": int((df_excluded["discharge_disposition_id"].isin([11, 19, 20, 21]) & (df_excluded["readmitted"] == ">30")).sum()),
                "NO": int((df_excluded["discharge_disposition_id"].isin([11, 19, 20, 21]) & (df_excluded["readmitted"] == "NO")).sum()),
                "total": int((df_excluded["discharge_disposition_id"].isin([11, 19, 20, 21])).sum()),
            },
        },
        "miscoded_status_evaluation": (
            "NOT MISCODED. All 86 encounters belong to disposition 13 (Hospice/home: 55 cases) "
            "or 14 (Hospice/facility: 31 cases). Exactly ZERO in-hospital deaths (codes 11, 19, 20, 21) "
            "have readmission labels (all 2,337 true in-hospital deaths have readmitted='NO'). "
            "Patients discharged to hospice are alive at discharge; these 86 cases represent genuine "
            "clinical palliative crises, hospice revocations, or symptom flare-ups leading to acute readmission. "
            "They are excluded under standard health-plan quality metric policies (e.g. CMS HRRP), which "
            "exclude all hospice discharges from readmission penalty benchmarks, not because of database error."
        ),
    }

    # Eligible Cohort
    df_eligible = df[~mask_excluded].copy()
    n_enc_eligible = len(df_eligible)
    n_pat_eligible = int(df_eligible["patient_nbr"].nunique())
    eligible_targets = df_eligible["readmitted"].value_counts().to_dict()
    eligible_pos = int((df_eligible["readmitted"] == "<30").sum())
    eligible_pos_pct = round(eligible_pos / n_enc_eligible * 100, 2)

    # Automated assertions
    assert n_enc_raw == n_enc_excluded + n_enc_eligible, "Encounter count mismatch during cohort filtering!"
    assert (df_eligible["discharge_disposition_id"].isin(excluded_codes)).sum() == 0, "Eligible cohort contains excluded disposition IDs!"

    comparison = {
        "verified_exclusion_codes": mortality_hospice_codes,
        "unfiltered_cohort": {
            "total_encounters": n_enc_raw,
            "unique_patients": n_pat_raw,
            "target_counts": raw_targets,
            "positive_count_lt30": raw_pos,
            "positive_prevalence_pct": raw_pos_pct,
        },
        "excluded_mortality_hospice": {
            "total_encounters": n_enc_excluded,
            "unique_patients": n_pat_excluded,
            "target_counts": excluded_targets,
            "positive_count_lt30": excluded_pos,
            "positive_prevalence_pct": excluded_pos_pct,
            "hospice_86_breakdown": hospice_86_breakdown,
            "policy_rationale": (
                "Patients who expired in-hospital cannot be readmitted, introducing fatal survivorship bias. "
                "Patients transferred to hospice have fundamentally different palliative care goals and are "
                "excluded under CMS and health-plan readmission benchmark standards."
            ),
        },
        "eligible_modeling_cohort": {
            "total_encounters": n_enc_eligible,
            "unique_patients": n_pat_eligible,
            "target_counts": eligible_targets,
            "positive_count_lt30": eligible_pos,
            "positive_prevalence_pct": eligible_pos_pct,
            "imbalance_ratio": round((n_enc_eligible - eligible_pos) / eligible_pos, 2),
        },
    }

    return df_eligible, comparison


def audit_uci_dataset(df: pd.DataFrame, ids_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Run full audit on UCI Diabetes DataFrame.
    Includes both unfiltered baseline metrics and eligible modeling cohort filtering.
    """
    total_encounters = len(df)
    unique_patients = int(df["patient_nbr"].nunique())
    unique_encounters = int(df["encounter_id"].nunique())

    # Check duplicates
    full_duplicates = int(df.duplicated().sum())
    duplicate_encounters = int(df["encounter_id"].duplicated().sum())

    # Patient repeat encounters
    enc_per_patient = df.groupby("patient_nbr")["encounter_id"].count()
    patient_stats = {
        "unique_patients": unique_patients,
        "total_encounters": total_encounters,
        "encounters_per_patient_mean": float(enc_per_patient.mean()),
        "encounters_per_patient_median": float(enc_per_patient.median()),
        "encounters_per_patient_std": float(enc_per_patient.std()),
        "encounters_per_patient_max": int(enc_per_patient.max()),
        "patients_with_single_encounter": int((enc_per_patient == 1).sum()),
        "patients_with_single_encounter_pct": float((enc_per_patient == 1).mean() * 100),
        "patients_with_multiple_encounters": int((enc_per_patient > 1).sum()),
        "patients_with_multiple_encounters_pct": float((enc_per_patient > 1).mean() * 100),
        "patients_with_gt_5_encounters": int((enc_per_patient > 5).sum()),
    }

    # Missing values (only genuine missing value markers such as '?' converted to NaN)
    missing_summary = {}
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        if n_missing > 0:
            missing_summary[col] = {
                "missing_count": n_missing,
                "missing_pct": round(n_missing / total_encounters * 100, 2),
            }

    # Verify lab tests have 0 missingness
    assert df["max_glu_serum"].isna().sum() == 0, "max_glu_serum should have 0 NaNs!"
    assert df["A1Cresult"].isna().sum() == 0, "A1Cresult should have 0 NaNs!"

    lab_distributions = {
        "max_glu_serum": df["max_glu_serum"].value_counts().to_dict(),
        "A1Cresult": df["A1Cresult"].value_counts().to_dict(),
        "clinical_category_note": (
            "The category 'None' in max_glu_serum (96,420 encounters) and A1Cresult (84,748 encounters) "
            "represents 'Test Not Performed/Measured'. It is a valid, clinically informative category, "
            "not a corrupted missing value."
        ),
    }

    # Target audit
    actual_targets = set(df["readmitted"].dropna().unique())
    unexpected_targets = list(actual_targets - EXPECTED_TARGET_VALUES)
    target_counts = df["readmitted"].value_counts().to_dict()
    target_pcts = (df["readmitted"].value_counts(normalize=True) * 100).round(2).to_dict()

    # Proposed binary target: '<30' is positive (1), '>30' and 'NO' are negative (0)
    pos_count = int((df["readmitted"] == "<30").sum())
    neg_count = total_encounters - pos_count
    proposed_binary_target = {
        "positive_class": "<30 (readmitted within 30 days)",
        "negative_class": ">=30 or NO (not readmitted within 30 days in dataset)",
        "positive_count": pos_count,
        "positive_pct": round(pos_count / total_encounters * 100, 2),
        "negative_count": neg_count,
        "negative_pct": round(neg_count / total_encounters * 100, 2),
        "imbalance_ratio": round(neg_count / max(1, pos_count), 2),
        "clinical_definition_note": (
            "Positive class indicates an early unplanned readmission (<30 days). "
            "Negative class represents encounters with either late readmission (>30 days) "
            "or NO recorded readmission. Crucially, 'NO' signifies no recorded readmission "
            "within the 130 participating hospitals during the 1999-2008 study window, "
            "NOT medical proof that the patient was never hospitalized elsewhere."
        ),
    }

    # Eligible cohort filtering
    df_eligible, cohort_comparison = create_eligible_modeling_cohort(df, ids_path=ids_path)

    # Prediction time and feature pipeline documentation
    prediction_time_policy = {
        "intended_prediction_time": "Discharge (Transitional Care Management Planning)",
        "rationale": (
            "The model is intended to run at the time of discharge planning to flag high-risk diabetic "
            "patients who require intensive outpatient care coordination, medication reconciliation, "
            "or post-discharge home health monitoring. At discharge, full inpatient duration "
            "(time_in_hospital), procedure counts, in-hospital lab tests, and medication changes are known."
        ),
        "identifier_policy": (
            "Identifiers ('encounter_id', 'patient_nbr') are STRICTLY EXCLUDED from model features. "
            "They are retained solely for longitudinal linkage, patient-grouped cross-validation, and audit."
        ),
        "discharge_disposition_sensitivity_plan": (
            "In Milestone 2, baseline modeling will execute an explicit sensitivity comparison: "
            "Model A (Clinical Baseline without discharge_disposition_id) versus "
            "Model B (Augmented Model with discharge_disposition_id). "
            "Even after excluding death/hospice codes, disposition codes (e.g. SNF vs home health) "
            "reflect post-acute placement decisions negotiated during stay, which could introduce "
            "circular placement leakage. The sensitivity comparison will quantify this impact."
        ),
        "diagnosis_coding_sensitivity_plan": (
            "Similarly, final billed ICD-9 diagnoses (diag_1, diag_2, diag_3) are typically assigned "
            "post-discharge by professional medical coders. Milestone 2 will evaluate models with and without "
            "these discharge ICD-9 codes to assess lookahead coding bias."
        ),
        "administrative_missing_features": (
            "weight (96.86% missing) and payer_code (39.56% missing) are dropped from clinical modeling "
            "due to non-random administrative missingness."
        ),
    }

    # Data types summary
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    return {
        "dataset_name": "UCI Diabetes 130-US Hospitals (1999-2008)",
        "total_rows": total_encounters,
        "total_columns": int(df.shape[1]),
        "full_duplicate_rows": full_duplicates,
        "duplicate_encounter_ids": duplicate_encounters,
        "patient_statistics": patient_stats,
        "target_audit": {
            "expected_values": sorted(list(EXPECTED_TARGET_VALUES)),
            "actual_values": sorted(list(actual_targets)),
            "unexpected_values": unexpected_targets,
            "distribution_counts": target_counts,
            "distribution_percentages": target_pcts,
            "proposed_binary_target": proposed_binary_target,
        },
        "cohort_policy_and_filtering": cohort_comparison,
        "prediction_time_policy": prediction_time_policy,
        "missing_values": missing_summary,
        "lab_distributions": lab_distributions,
        "column_types": {
            "numeric_count": len(numeric_cols),
            "numeric_columns": numeric_cols,
            "categorical_count": len(categorical_cols),
            "categorical_columns": categorical_cols,
        },
        "feature_temporal_taxonomy": FEATURE_TEMPORAL_TAXONOMY,
    }


def split_uci_patient_grouped(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Partition UCI Diabetes into Train, Validation, and Test sets grouped strictly by patient_nbr.
    Ensures ZERO patient leakage across partitions.
    """
    target_series = (df["readmitted"] == "<30").astype(int)

    # First split off test set
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss_test.split(df, target_series, groups=df["patient_nbr"]))

    df_train_val = df.iloc[train_val_idx]
    df_test = df.iloc[test_idx].copy()

    # Split remaining into train and validation
    rel_val_size = val_size / (1.0 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=rel_val_size, random_state=random_state)
    train_idx_rel, val_idx_rel = next(gss_val.split(
        df_train_val,
        (df_train_val["readmitted"] == "<30").astype(int),
        groups=df_train_val["patient_nbr"]
    ))

    df_train = df_train_val.iloc[train_idx_rel].copy()
    df_val = df_train_val.iloc[val_idx_rel].copy()

    # Automated assertions of zero leakage
    train_patients = set(df_train["patient_nbr"])
    val_patients = set(df_val["patient_nbr"])
    test_patients = set(df_test["patient_nbr"])

    train_val_overlap = len(train_patients.intersection(val_patients))
    train_test_overlap = len(train_patients.intersection(test_patients))
    val_test_overlap = len(val_patients.intersection(test_patients))

    assert train_val_overlap == 0, f"Patient leakage between Train and Val: {train_val_overlap} patients"
    assert train_test_overlap == 0, f"Patient leakage between Train and Test: {train_test_overlap} patients"
    assert val_test_overlap == 0, f"Patient leakage between Val and Test: {val_test_overlap} patients"

    split_metrics = {
        "train": {
            "encounters": len(df_train),
            "patients": len(train_patients),
            "positive_readmissions_30d": int((df_train["readmitted"] == "<30").sum()),
            "positive_rate_pct": round(float((df_train["readmitted"] == "<30").mean() * 100), 2),
        },
        "validation": {
            "encounters": len(df_val),
            "patients": len(val_patients),
            "positive_readmissions_30d": int((df_val["readmitted"] == "<30").sum()),
            "positive_rate_pct": round(float((df_val["readmitted"] == "<30").mean() * 100), 2),
        },
        "test": {
            "encounters": len(df_test),
            "patients": len(test_patients),
            "positive_readmissions_30d": int((df_test["readmitted"] == "<30").sum()),
            "positive_rate_pct": round(float((df_test["readmitted"] == "<30").mean() * 100), 2),
        },
        "overlap_verification": {
            "train_val_overlap": train_val_overlap,
            "train_test_overlap": train_test_overlap,
            "val_test_overlap": val_test_overlap,
            "status": "PASSED - ZERO PATIENT LEAKAGE",
        },
        "protocol_rule": (
            "Do NOT fit encoders, imputers, scalers, or feature selectors on the test set. "
            "All transformations must be fitted solely on the training partition and applied to val/test."
        ),
    }

    return df_train, df_val, df_test, split_metrics
