"""
Track A UI Module: 30-Day Readmission Risk Estimation.
Uses frozen Baseline A Logistic Regression Pipeline (models/baseline_a_logreg.joblib).
Strictly restricts inputs to pre-admission and inpatient features available at discharge planning.
Does NOT expose post-acute disposition or retrospective billing diagnoses.
"""
import streamlit as st
import pandas as pd
import numpy as np

from model_loader import (
    load_track_a_pipeline,
    load_track_a_metrics,
    compute_track_a_plain_statement,
    compute_relative_risk,
    classify_readmission_risk,
    TRACK_A_FROZEN_THRESHOLD,
)

# Load frozen baseline metrics once at startup; fails immediately if file or keys are missing
TRACK_A_METRICS = load_track_a_metrics()



# Exactly 40 features matching Baseline A
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
    "race", "gender", "age", "admission_type_id", "admission_source_id",
    "max_glu_serum", "A1Cresult", "metformin", "repaglinide", "nateglinide",
    "chlorpropamide", "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
    "troglitazone", "tolazamide", "examide", "citoglipton", "insulin",
    "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone", "change", "diabetesMed",
]

PRESETS = {
    "High-Risk Inpatient Profile (Frequent Inpatient, Insulin Titration)": {
        "race": "Caucasian", "gender": "Male", "age": "[70-80)",
        "admission_type_id": "1",  # Emergency
        "admission_source_id": "7",  # Emergency Room
        "time_in_hospital": 8, "num_lab_procedures": 62, "num_procedures": 2, "num_medications": 22,
        "number_outpatient": 1, "number_emergency": 2, "number_inpatient": 3, "number_diagnoses": 9,
        "max_glu_serum": "None", "A1Cresult": ">8", "insulin": "Up", "metformin": "No",
        "change": "Ch", "diabetesMed": "Yes",
    },
    "Moderate-Risk Inpatient Profile (Average Inpatient, Oral Agent)": {
        "race": "AfricanAmerican", "gender": "Female", "age": "[50-60)",
        "admission_type_id": "2",  # Urgent
        "admission_source_id": "1",  # Physician Referral
        "time_in_hospital": 4, "num_lab_procedures": 41, "num_procedures": 1, "num_medications": 12,
        "number_outpatient": 0, "number_emergency": 0, "number_inpatient": 1, "number_diagnoses": 6,
        "max_glu_serum": "None", "A1Cresult": "Norm", "insulin": "No", "metformin": "Steady",
        "change": "No", "diabetesMed": "Yes",
    },
    "Low-Risk Inpatient Profile (Elective, Single Stay, Stable)": {
        "race": "Caucasian", "gender": "Female", "age": "[30-40)",
        "admission_type_id": "3",  # Elective
        "admission_source_id": "1",  # Physician Referral
        "time_in_hospital": 2, "num_lab_procedures": 25, "num_procedures": 0, "num_medications": 6,
        "number_outpatient": 0, "number_emergency": 0, "number_inpatient": 0, "number_diagnoses": 3,
        "max_glu_serum": "Norm", "A1Cresult": "None", "insulin": "No", "metformin": "Steady",
        "change": "No", "diabetesMed": "Yes",
    },
}


def build_default_encounter_dict():
    """Builds a complete baseline dictionary for all 40 features."""
    enc = {
        "time_in_hospital": 3,
        "num_lab_procedures": 40,
        "num_procedures": 1,
        "num_medications": 10,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 0,
        "number_diagnoses": 5,
        "race": "Caucasian",
        "gender": "Female",
        "age": "[50-60)",
        "admission_type_id": "1",
        "admission_source_id": "1",
        "max_glu_serum": "None",
        "A1Cresult": "None",
        "change": "No",
        "diabetesMed": "Yes",
    }
    # Medications default to 'No'
    meds = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
        "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
        "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"
    ]
    for m in meds:
        enc[m] = "No"
    return enc


def render_track_a_page():
    st.header("Track A — Inpatient 30-Day Readmission Risk Estimation")
    st.markdown(
        """
        **Clinical CDS Setting:** Bedside Discharge Planning (Transitional Care Management).  
        **Model Architecture:** Frozen Baseline A Pipeline (`ColumnTransformer` + `LogisticRegression`).  
        **Leakage Safeguard:** Strictly utilizes features available during inpatient stay. Excludes post-acute disposition (`discharge_disposition_id`) and post-discharge ICD-9 billing codes to prevent lookahead bias.
        """
    )

    pipeline = load_track_a_pipeline()

    # Preset selector
    st.subheader("1. Patient Profile Selection")
    preset_col, _ = st.columns([3, 1])
    with preset_col:
        selected_preset = st.selectbox(
            "Load a clinical persona or customize below:",
            ["Custom Manual Entry"] + list(PRESETS.keys())
        )

    encounter_data = build_default_encounter_dict()
    if selected_preset in PRESETS:
        encounter_data.update(PRESETS[selected_preset])

    st.subheader("2. Inpatient Clinical Parameters")
    tab_demo, tab_util, tab_meds = st.tabs([
        "Demographics & Admission",
        "Hospital Stay & Prior Utilization",
        "Glycemic Labs & Pharmacotherapy"
    ])

    with tab_demo:
        c1, c2, c3 = st.columns(3)
        with c1:
            age_options = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
            encounter_data["age"] = st.selectbox("Age Range", age_options, index=age_options.index(encounter_data["age"]))
            gender_options = ["Female", "Male", "Unknown/Invalid"]
            idx_g = gender_options.index(encounter_data["gender"]) if encounter_data["gender"] in gender_options else 0
            encounter_data["gender"] = st.selectbox("Gender", gender_options, index=idx_g)
        with c2:
            race_options = ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "Missing"]
            idx_r = race_options.index(encounter_data["race"]) if encounter_data["race"] in race_options else 0
            encounter_data["race"] = st.selectbox("Race / Ethnicity", race_options, index=idx_r)
            adm_types = {"1": "1 - Emergency", "2": "2 - Urgent", "3": "3 - Elective", "4": "4 - Newborn", "5": "5 - Not Available", "7": "7 - Trauma Center"}
            adm_keys = list(adm_types.keys())
            idx_adm = adm_keys.index(encounter_data["admission_type_id"]) if encounter_data["admission_type_id"] in adm_keys else 0
            encounter_data["admission_type_id"] = st.selectbox("Admission Type", adm_keys, format_func=lambda x: adm_types.get(x, x), index=idx_adm)
        with c3:
            adm_src = {"1": "1 - Physician Referral", "7": "7 - Emergency Room", "2": "2 - Clinic Referral", "4": "4 - Transfer from Hospital", "6": "6 - Transfer from SNF"}
            adm_src_keys = list(adm_src.keys())
            idx_src = adm_src_keys.index(encounter_data["admission_source_id"]) if encounter_data["admission_source_id"] in adm_src_keys else 0
            encounter_data["admission_source_id"] = st.selectbox("Admission Source", adm_src_keys, format_func=lambda x: adm_src.get(x, x), index=idx_src)

    with tab_util:
        u1, u2, u3, u4 = st.columns(4)
        with u1:
            encounter_data["time_in_hospital"] = st.number_input("Inpatient Length of Stay (Days)", min_value=1, max_value=14, value=int(encounter_data["time_in_hospital"]))
            encounter_data["num_lab_procedures"] = st.number_input("Lab Procedures Count", min_value=1, max_value=132, value=int(encounter_data["num_lab_procedures"]))
        with u2:
            encounter_data["num_procedures"] = st.number_input("Non-Lab Procedures Count", min_value=0, max_value=6, value=int(encounter_data["num_procedures"]))
            encounter_data["num_medications"] = st.number_input("Distinct Medications Count", min_value=1, max_value=81, value=int(encounter_data["num_medications"]))
        with u3:
            encounter_data["number_inpatient"] = st.number_input("Prior Inpatient Visits (Past Year)", min_value=0, max_value=21, value=int(encounter_data["number_inpatient"]))
            encounter_data["number_emergency"] = st.number_input("Prior Emergency Visits (Past Year)", min_value=0, max_value=76, value=int(encounter_data["number_emergency"]))
        with u4:
            encounter_data["number_outpatient"] = st.number_input("Prior Outpatient Visits (Past Year)", min_value=0, max_value=42, value=int(encounter_data["number_outpatient"]))
            encounter_data["number_diagnoses"] = st.number_input("Entered Diagnoses Count", min_value=1, max_value=16, value=int(encounter_data["number_diagnoses"]))

    with tab_meds:
        m1, m2 = st.columns(2)
        with m1:
            glu_opts = ["None", "Norm", ">200", ">300"]
            encounter_data["max_glu_serum"] = st.selectbox("Max Glucose Serum Test", glu_opts, index=glu_opts.index(encounter_data["max_glu_serum"]))
            a1c_opts = ["None", "Norm", ">7", ">8"]
            encounter_data["A1Cresult"] = st.selectbox("HbA1c Lab Test Result", a1c_opts, index=a1c_opts.index(encounter_data["A1Cresult"]))
        with m2:
            encounter_data["change"] = st.selectbox("Any Diabetes Med Changed?", ["No", "Ch"], index=0 if encounter_data["change"] == "No" else 1)
            encounter_data["diabetesMed"] = st.selectbox("Any Diabetes Med Prescribed?", ["Yes", "No"], index=0 if encounter_data["diabetesMed"] == "Yes" else 1)

        st.markdown("---")
        with st.expander("All 23 Diabetes Medications (Exposed — Default: 'No')", expanded=True):
            st.caption("All 23 inpatient diabetes medications are configurable below. Each defaults to **No** unless prescribed or loaded via preset.")
            all_23_meds = [
                "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
                "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
                "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
                "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
                "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"
            ]
            med_levels = ["No", "Steady", "Up", "Down"]
            med_cols = st.columns(3)
            for i, med in enumerate(all_23_meds):
                col = med_cols[i % 3]
                cur_val = encounter_data.get(med, "No")
                idx_m = med_levels.index(cur_val) if cur_val in med_levels else 0
                encounter_data[med] = col.selectbox(
                    f"{med} (Default: No)",
                    med_levels,
                    index=idx_m,
                    key=f"med_sel_{med}"
                )

    # Prediction Action
    st.subheader("3. Risk Stratification")
    if st.button("Compute Readmission Risk Estimate", type="primary"):
        # Format into single-row DataFrame
        df_input = pd.DataFrame([encounter_data])
        # Ensure strings for categorical IDs
        df_input["admission_type_id"] = df_input["admission_type_id"].astype(str)
        df_input["admission_source_id"] = df_input["admission_source_id"].astype(str)

        # Predict probability
        prob = float(pipeline.predict_proba(df_input[NUMERICAL_COLS + BASE_CATEGORICAL_COLS])[:, 1][0])
        tau = TRACK_A_FROZEN_THRESHOLD
        risk_class = classify_readmission_risk(prob, tau)
        is_elevated = (risk_class == "ELEVATED")
        base_rate = TRACK_A_METRICS["base_rate"]  # 0.1111 from JSON
        relative_risk = compute_relative_risk(prob, base_rate)
        precision = TRACK_A_METRICS["precision"]  # 0.1806 from JSON
        recall = TRACK_A_METRICS["recall"]        # 0.38 from JSON
        specificity = TRACK_A_METRICS["specificity"]  # 0.7846 from JSON
        auroc = TRACK_A_METRICS["auroc"]          # 0.6324 from JSON

        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        with res_col1:
            st.metric("Predicted 30-Day Risk", f"{prob * 100:.2f}%")
        with res_col2:
            st.metric("Population Base Rate", f"{base_rate * 100:.2f}%")
        with res_col3:
            st.metric("Relative Risk", f"{relative_risk:.2f}x")
        with res_col4:
            if is_elevated:
                st.error("Classification: ELEVATED RISK")
            else:
                st.success("Classification: STANDARD RISK")

        st.progress(
            min(prob, 1.0),
            text=f"Estimated Readmission Probability: {prob * 100:.2f}% (Statistical probability estimate, NOT a patient safety score)"
        )

        plain_statement = compute_track_a_plain_statement(precision=precision, tau=tau)

        st.warning(
            f"**Population Operating Context (Flagged Group Test Performance):**  \n"
            f"{plain_statement}  \n\n"
            "*(This statement describes historical test-set cohort performance for all encounters flagged at this threshold. "
            "It is a population operating characteristic and not an individualized forecast for the specific patient on screen.)*"
        )

        st.info(
            rf"""
            **Decision Threshold vs. Evaluated Test Metrics (Model Operating Range):**
            - **Classification Rule:** Patient encounter flagged as ELEVATED RISK if predicted probability $\ge \tau = {tau:.3f}$; otherwise STANDARD RISK.
            - **Validation-Tuned Decision Threshold:** $\tau = {tau:.3f}$ (calibrated on the validation partition to optimize $F_1$-score; threshold selection is kept strictly separate from test evaluation metrics).
            - **Model Operating Range on Frozen Test Set (at $\tau = {tau:.3f}$):**
              - **Recall:** {recall:.2%}
              - **Precision:** {precision:.2%}
              - **Specificity:** {specificity:.2%}
              - **AUROC:** {auroc:.4f}
            - **Transitional Care Trigger (Hypothetical Example Workflow — NOT Evaluated):** In simulated research workflows, patients above $\tau = {tau:.3f}$ are assigned to post-discharge nurse outreach, early primary care visits, and medication reconciliation. This operational protocol is illustrative and has **not** been prospectively or clinically evaluated.
            """
        )

    st.markdown("---")
    st.caption(
        "**Research Disclaimer:** This risk score is an experimental statistical estimate derived from historical 1999–2008 data. "
        "It is NOT a medical diagnosis, clinical prognosis, or treatment directive. It must never supersede professional clinical judgment."
    )
