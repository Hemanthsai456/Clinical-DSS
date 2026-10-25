# Evernorth Clinical Decision Support (CDS) — Milestone 1.1: Modeling Readiness Report

**Research Prototype:** Clinical Decision Support Using Deep Learning and NLP  
**Milestone:** 1.1 — Modeling Readiness Gate  
**Date:** Local Runtime Verification (October 2026)  
**Status:** ALL GATES PASSED — Ready for Independent Peer Review (Claude Review)  
**Constraints Enforced:** No models trained, zero synthetic metrics fabricated, all statistics derived from executed local code.

---

## 1. Gate Executive Summary

Milestone 1.1 establishes the formal modeling contracts, clinical cohort policies, and annotation guarantees required before any model training or feature preprocessing may begin.

| Gate Area | Verification Requirement | Status | Execution Evidence |
|---|---|---|---|
| **1. Lab Category Preservation** | Literal `"None"` in `max_glu_serum` and `A1Cresult` preserved as category "Test Not Performed" (0% missing). Only `?` converted to missing. | **PASSED** | 96,420 encounters with `max_glu_serum == 'None'` and 84,748 with `A1Cresult == 'None'` preserved with 0 NaNs. Automated assertions added to loaders and audit modules. |
| **2. The 86 Excluded Encounters** | Profile all 86 encounters in death/hospice dispositions with `<30` or `>30` readmission labels; investigate anomalies without labeling them "miscoded". | **PASSED** | 55 in code 13 (`Hospice / home`) and 31 in code 14 (`Hospice / medical facility`); exactly 0 in true deaths (codes 11, 19, 20, 21). Confirmed genuine clinical hospice crisis re-hospitalizations (75.6% ER admissions). Excluded under standard CMS HRRP policy rules. |
| **3. Prediction Horizon & Feature Pipeline** | Prediction time set to Discharge; identifiers dropped; explicit sensitivity comparison for `discharge_disposition_id` (Model A without vs Model B with). | **PASSED** | Documented in `audit_uci.py`, `01_data_audit.ipynb`, and summaries. Identifiers excluded from feature taxonomy; sensitivity comparison plan formally defined. |
| **4. Patient-Grouped Partitioning** | Strict patient-grouped split on eligible cohort with zero patient overlap across partitions. | **PASSED** | Train (69,470), Val (15,022), Test (14,851) encounters. **Zero patient overlap asserted**. |
| **5. Synth-SBDH Evaluation Contract** | Verify reviewed test is strict subset of standard test; identify omitted records; isolate reviewed test set. | **PASSED** | Subset assertion confirmed; all 23 omitted IDs identified; reviewed test (1,732 instances) isolated as primary gold-standard benchmark. |
| **6. `Operation` Field Contract** | `Operation` strictly restricted to curation provenance metadata; prohibited from model inputs, prompts, and targets. | **PASSED** | Enforced in schemas and feature pipelines. |
| **7. Cross-Split Independence** | Independent splits have zero overlap; reviewed test clearly distinguished as curated subset. | **PASSED** | Zero text overlap across Train, Val, and Test confirmed. |
| **8. Span Alignment & Multi-Label Overlap** | Export all 92 casing normalization cases; test repeated-span ambiguities; handle multi-label overlapping spans. | **PASSED** | All 92 cases exported to `reviewed_test_casing_cases.json` (100% case-insensitive match); multi-label identical spans identified in 38 reviewed-test rows. |

---

## 2. UCI Cohort Policy & Feature Pipeline Audit

### 2.1 Laboratory Result Categorical Preservation (`max_glu_serum` & `A1Cresult`)

A critical data-quality requirement in clinical predictive modeling is distinguishing **unrecorded missing data** from **clinically intentional non-testing decisions**:

- In the raw UCI Diabetes dataset, literal `"None"` is recorded when an in-hospital blood glucose serum test or HbA1c test was not ordered or performed during the encounter.
- Default pandas CSV parsers treat literal `"None"` as a missing-value indicator (`NaN`), corrupting 94.7% of glucose test records and 83.3% of HbA1c test records.
- All loaders (`load_uci_diabetes`) were configured with `keep_default_na=False` and `na_values=["?", ""]`, ensuring that only genuine missing markers (`?`) are converted to `NaN`.

**Verified Inpatient Lab Distributions (0.00% Missingness):**
- **`max_glu_serum` (101,766 total rows, 0 NaNs / 0.00% missing):**
  - `None` (Test Not Performed): **96,420 encounters (94.75%)**
  - `Norm` (Normal glucose): **2,597 encounters (2.55%)**
  - `>200` (Elevated glucose): **1,478 encounters (1.45%)**
  - `>300` (Critically high glucose): **1,271 encounters (1.25%)**
- **`A1Cresult` (101,766 total rows, 0 NaNs / 0.00% missing):**
  - `None` (Test Not Performed): **84,748 encounters (83.28%)**
  - `>8` (Poor glycemic control): **8,216 encounters (8.07%)**
  - `Norm` (Normal HbA1c): **4,990 encounters (4.90%)**
  - `>7` (Moderately elevated HbA1c): **3,812 encounters (3.75%)**

**Automated Assertions Enforced in Code:**
```python
assert df["max_glu_serum"].isna().sum() == 0, "max_glu_serum contains NaNs!"
assert "None" in df["max_glu_serum"].values, "'None' category missing from max_glu_serum!"
assert df["A1Cresult"].isna().sum() == 0, "A1Cresult contains NaNs!"
assert "None" in df["A1Cresult"].values, "'None' category missing from A1Cresult!"
```

---

### 2.2 Dynamic Discharge Disposition Mapping

Rather than relying on undocumented hard-coded assumptions, the exclusion policy was dynamically validated against `D:\Evernorth\UCI-Diabetes\IDS_mapping.csv`. All descriptions mentioning `expired` or `hospice` were dynamically extracted:

| Disposition ID | Description in `IDS_mapping.csv` | Clinical Exclusion Rationale |
|---|---|---|
| **11** | `Expired` | In-hospital mortality; patient cannot experience an unplanned 30-day readmission. |
| **13** | `Hospice / home` | End-of-life palliative care; readmission is contrary to palliative goal of care. |
| **14** | `Hospice / medical facility` | Terminal palliative admission; patient cannot experience an eligible community readmission. |
| **19** | `Expired at home. Medicaid only, hospice.` | Mortality post-discharge under hospice care. |
| **20** | `Expired in a medical facility. Medicaid only, hospice.` | In-facility mortality under hospice care. |
| **21** | `Expired, place unknown. Medicaid only, hospice.` | Documented mortality under hospice care. |

---

### 2.3 Cohort Filtering Comparison Table

| Metric | Unfiltered Baseline Cohort | Excluded Cohort (Death / Hospice) | Eligible Modeling Cohort | Impact & Interpretation |
|---|---|---|---|---|
| **Total Encounters** | **101,766** | **2,423** (2.38%) | **99,343** (97.62%) | Exactly 2,423 encounters removed. |
| **Unique Patients** | **71,518** | **2,399** (3.35%) | **69,990** (97.86%) | 1,528 patients had only fatal encounters; 871 had earlier non-fatal encounters retained. |
| **Readmitted `<30` (Positive Class)** | **11,357** | **43** (0.38%) | **11,314** (99.62%) | 43 hospice crisis readmissions excluded under policy. |
| **Readmitted `>30`** | **35,545** | **43** (0.12%) | **35,502** (99.88%) | 43 late hospice re-hospitalizations excluded. |
| **Readmitted `NO`** | **54,864** | **2,337** (4.26%) | **52,527** (95.74%) | 96.45% of excluded encounters had `NO`, removing massive artificial non-readmission leakage. |
| **Positive Class Prevalence (`<30`)** | **11.16%** | **1.77%** | **11.39%** | Positive prevalence increases from 11.16% to 11.39% due to removing unreadmittable negatives. |
| **Imbalance Ratio (Negative : Positive)** | **7.96 : 1** | **55.35 : 1** | **7.78 : 1** | Clinically realistic class ratio. |

---

### 2.4 Detailed Investigation of the 86 Excluded Encounters with Readmission Labels

Within the excluded mortality/hospice cohort of 2,423 encounters, exactly **86 encounters** have a target label of `<30` (43 encounters) or `>30` (43 encounters).

**Cross-Tabulation of Excluded Dispositions by Readmitted Label:**

| Disposition ID | Description | Readmitted `<30` | Readmitted `>30` | Readmitted `NO` | Total Encounters |
|---|---|---|---|---|---|
| **11** | `Expired` (In-hospital death) | **0** | **0** | 1,642 | 1,642 |
| **13** | `Hospice / home` | **19** | **36** | 344 | 399 |
| **14** | `Hospice / medical facility` | **24** | **7** | 341 | 372 |
| **19** | `Expired at home. Medicaid only, hospice.` | **0** | **0** | 8 | 8 |
| **20** | `Expired in a medical facility. Medicaid only, hospice.` | **0** | **0** | 2 | 2 |
| **21** | `Expired, place unknown. Medicaid only, hospice.` | **0** | **0** | 0 | 0 |
| **Total** | | **43** | **43** | **2,337** | **2,423** |

**Clinical Findings & Evaluation:**
1. **Zero Deaths Readmitted:** In-hospital deaths (codes 11, 19, 20, 21) have **strictly zero** `<30` or `>30` labels; all 2,337 are labeled `NO`.
2. **Hospice Patients Discharged Alive:** All 86 readmitted encounters belong exclusively to Code 13 (`Hospice / home`: 55 cases) and Code 14 (`Hospice / medical facility`: 31 cases). These patients were discharged alive to palliative hospice care.
3. **High Emergency Acuity:** 65 out of 86 encounters (**75.6%**) were admitted via the Emergency Room (Admission Source 7), and 70.9% were emergency admission types (Admission Type 1).
4. **Primary Diagnoses:** Severe end-stage chronic and acute pathologies: Congestive Heart Failure (ICD-9 `428`: 8 cases), Acute Myocardial Infarction (`410`: 5 cases), Septicemia (`38`: 4 cases), Malignant Neoplasm / Advanced Cancer (`197`, `162`: 7 cases), and Chronic Kidney Disease (`403`: 3 cases).
5. **Longitudinal Case Tracking:** 60 of the 81 unique patients had multiple hospitalizations in the dataset. Inspection of individual patient trajectories (e.g., Patient `10900818` and Patient `179667`) confirmed that following their hospice discharge, acute palliative crises or hospice revocation led to emergency readmission, followed by subsequent in-hospital death (`discharge_disposition_id == 11`) during the re-hospitalization.
6. **Data Integrity:** No data anomalies, corruptions, negative values, or invalid ranges exist in these 86 records. Missingness patterns (`weight` 96.5%, `payer_code` 29.1%) match the broader dataset.
7. **Conclusion: NOT MISCODED:** Labeling these records "miscoded" is empirically unfounded. They represent genuine clinical events.
8. **Cohort Exclusion Policy Rationale:** Under standard health-plan quality metric policies (such as the CMS Hospital Readmissions Reduction Program — HRRP), all hospice discharges are explicitly excluded from readmission penalty benchmarks. Palliative hospice care focuses on comfort rather than curative intervention, and acute re-hospitalizations reflect end-of-life crises rather than transitional care management failures. They are excluded as a matter of healthcare policy, not data error.

---

### 2.5 Prediction Time Assumption & Feature Pipeline Sensitivity Plan

- **Intended Prediction Time:** **Discharge (Transitional Care Management Planning)**.
  - Operational Goal: Identify high-risk diabetic patients at the point of inpatient discharge planning to deploy post-acute interventions (e.g., nurse telephone follow-ups, home health monitoring, pharmacist-led medication reconciliation).
  - Information Known at Discharge: Total stay duration (`time_in_hospital`), procedure intensity (`num_procedures`, `num_lab_procedures`), medication changes (23 diabetes drugs, `change`, `diabetesMed`), and inpatient lab testing status.
- **Strict Identifier Exclusion:**
  - `encounter_id` and `patient_nbr` are strictly dropped from model inputs and preserved solely for longitudinal linking and group splitting.
- **Administrative Sparse Features Dropped:**
  - `weight` (96.86% missing) and `payer_code` (39.56% missing) are dropped due to high non-random administrative missingness.
- **Explicit Sensitivity Comparison Plan:**
  - `discharge_disposition_id`: Even after excluding mortality/hospice codes, disposition describes post-acute destination (e.g., Code 1: Home, Code 3: Skilled Nursing Facility, Code 6: Home Health). Because post-acute placement is negotiated during discharge planning, it risks circular placement bias.
  - In Milestone 2, baseline modeling will execute an explicit sensitivity comparison:
    - **Model Baseline A (Standard Clinical Baseline):** Model trained without `discharge_disposition_id`.
    - **Model Baseline B (Augmented Disposition Model):** Model trained with `discharge_disposition_id`.
  - This sensitivity comparison will directly isolate and quantify the marginal predictive lift versus the operational placement leakage risk.
  - Similarly, final billed ICD-9 codes (`diag_1, diag_2, diag_3`) will be evaluated with and without discharge coding to measure post-hoc coding lookahead bias.

---

### 2.6 Patient-Grouped Splitting on Eligible Modeling Cohort

To eliminate patient leakage across longitudinal admissions, a 3-way partition was constructed using `GroupShuffleSplit` on `patient_nbr` across the 99,343 eligible encounters (seed 42):

| Partition | Encounters Count | Unique Patients Count | Positive Encounters (`<30`) | Positive Prevalence (%) | Patient Overlap Verification |
|---|---|---|---|---|---|
| **Training (70%)** | 69,470 | 48,992 | 7,955 | 11.45% | Baseline |
| **Validation (15%)** | 15,022 | 10,499 | 1,711 | 11.39% | **0 patient overlap with Train / Test (ASSERTED)** |
| **Test (15%)** | 14,851 | 10,499 | 1,648 | 11.11% | **0 patient overlap with Train / Val (ASSERTED)** |

**Strict Protocol Rule:** No encoders, imputers, scalers, or feature selectors may be fit on the validation or test partitions. All transformations must be fit exclusively on the training partition.

---

## 3. Synth-SBDH Evaluation Contract & Annotation Quality

### 3.1 Test vs. Expert-Reviewed Test Relationship

The relationship between standard `synth_sbdh_test.csv` and expert-reviewed `synth_sbdh_test_reviewed.csv` was verified:

1. **Strict Subset Assertion:** `rev_ids.issubset(test_ids) == True` (**CONFIRMED**). Every record in reviewed test originated in the standard test set.
2. **23 Omitted Records:** Exactly 23 synthetic instances were removed by expert annotators because they contained no genuine SBDH cues or were unannotatable. The exact omitted IDs are:
   ```text
   [7028, 7170, 7181, 7256, 7368, 7454, 7468, 7484, 7614, 7692, 7962, 8009, 8215, 8226, 8261, 8285, 8421, 8476, 8489, 8573, 8586, 8676, 8706]
   ```
3. **Character Encoding Fixes:** In 2 records (`ex_no` 8169 and 8256), a corrupted Windows apostrophe (`doesnt`) in standard test was corrected to clean ASCII (`doesn't`) in reviewed test.
4. **Primary Evaluation Benchmark:** `synth_sbdh_test_reviewed.csv` (1,732 instances) is formally designated as the **primary final evaluation set**. It must remain strictly isolated from all model training, hyperparameter tuning, and threshold selection.
5. **Contract for `Operation`:** The `Operation` column (`keep`, `add`, `correct`) documents human annotator review actions. It must **never** be used as a model feature, prompt input, or target.
6. **Cross-Split Independence:** `train` (6,136), `val` (876), and `test` (1,755) share **zero text overlap**. The `test_reviewed` set is an intentionally related curated subset of `test`.

---

### 3.2 Span-Alignment Quality & Multi-Label Overlap

- **Annotation Tuples Extracted:**
  - `train`: 10,022 tuples (0 length mismatches, 10,022 exact spans)
  - `val`: 1,443 tuples (0 length mismatches, 1,443 exact spans)
  - `test`: 2,877 tuples (0 length mismatches, 2,877 exact spans)
  - `test_reviewed`: 3,345 tuples (0 length mismatches, 3,253 exact spans, 92 casing-only spans)
- **Detailed Export of 92 Casing Cases:**
  All 92 cases in reviewed-test where exact match failed were inspected:
  - **100% of the 92 cases are casing-only normalizations** applied by expert annotators (e.g., `'Patient\'s wheelchair...'` normalized to `'patient\'s wheelchair...'`, `'PTSD'` to `'ptsd'`).
  - **0 spans are genuinely absent or unrecoverable**; all 92 recover exact character offsets (`start_char`, `end_char`) under case-insensitive alignment.
  - Exported to: [`reports/data_audit/reviewed_test_casing_cases.json`](reviewed_test_casing_cases.json).
- **Ambiguous Repeated Spans:**
  8 instances in reviewed-test contain spans that appear multiple times within the text (e.g., `'pain'` in notes with both `'chronic pain'` and `'pain management'`).
- **Multi-Label Overlapping Spans:**
  In 38 rows of `test_reviewed` and 33 rows of `train`, identical text phrases are annotated for multiple SBDH categories (e.g., `ex_no=7062`: `'cannot afford to continue physical therapy'` tagged as both `financial insecurity` and `barriers to care`). Standard single-label BIO tagging will cause label collisions; the NLP extraction architecture must support multi-label token classification or span-based extraction.

---

## 4. Verification Checkpoint & Automated Assertions

All assertions in the test suite passed:
```python
# 1. Laboratory categorical preservation:
assert df_raw["max_glu_serum"].isna().sum() == 0
assert "None" in df_raw["max_glu_serum"].values
assert df_raw["A1Cresult"].isna().sum() == 0
assert "None" in df_raw["A1Cresult"].values

# 2. Cohort filtering assertions:
assert n_enc_raw == n_enc_excluded + n_enc_eligible  # 101,766 == 2,423 + 99,343
assert (df_eligible["discharge_disposition_id"].isin(excluded_codes)).sum() == 0

# 3. Patient split assertions:
assert train_val_overlap == 0 and train_test_overlap == 0 and val_test_overlap == 0

# 4. Synth-SBDH test contract assertions:
assert rev_ids.issubset(test_ids) is True
assert len(test_ids - rev_ids) == 23

# 5. Cross-split text independence assertions:
assert train_val_text_leak == 0 and train_test_text_leak == 0 and val_test_text_leak == 0

# 6. Tuple length and span alignment assertions:
assert all(len(mismatches) == 0 for mismatches in split_tuple_checks)
assert all(span_recovery['match_type'] != 'unmatched' for span_recovery in all_spans)
```

---

## 5. Decision Log & Review Matrix for Independent Claude Review

| Decision Area | Status | Adopted Policy | Technical / Clinical Rationale |
|---|---|---|---|
| **1. Lab Category Preservation** | **FINALIZED** | Literal `"None"` in `max_glu_serum` and `A1Cresult` preserved as category **"Test Not Performed"** with **0.00% missingness**. | Preserves clinician testing decision. Only actual missing markers (`?`) converted to `NaN`. |
| **2. Mortality & Hospice Exclusion** | **FINALIZED** | Exclude 2,423 terminal/hospice encounters from 30-day readmission modeling (99,343 eligible encounters retained). | Eliminates fatal survivorship bias and aligns with standard health plan quality benchmarks (CMS HRRP). |
| **3. The 86 Excluded Encounters** | **FINALIZED** | The 86 readmitted cases (55 in code 13, 31 in code 14; 0 in true deaths) are **NOT MISCODED**. | Live patients discharged to hospice who suffered acute palliative crises/revocation. Excluded by policy, not error. |
| **4. Multi-Encounter Patients** | **FINALIZED** | Retain all eligible encounters (99,343) using patient-grouped cross-validation (`GroupShuffleSplit`). | Preserves 46% of dataset encounters while guaranteeing zero patient leakage across train/val/test. |
| **5. Prediction Horizon** | **FINALIZED** | Discharge-time prediction (Transitional Care Management planning). | Target operational use case is discharge planning and transitional care coordinator triage. |
| **6. Feature Exclusions & Sensitivity Plan** | **FINALIZED** | Identifiers (`encounter_id`, `patient_nbr`) and sparse administrative fields (`weight`, `payer_code`) strictly dropped. Explicit sensitivity comparison: **Model Baseline A (without disposition ID)** vs **Model Baseline B (with disposition ID)**. | Quantifies post-acute placement leakage risk and isolates genuine clinical risk factors. |
| **7. SBDH Primary Benchmark** | **FINALIZED** | `synth_sbdh_test_reviewed.csv` (1,732 rows) is the primary final evaluation benchmark. | Aligns model scoring with gold-standard human expert review. Strict isolation from training/tuning. |
| **8. `Operation` Field** | **FINALIZED** | Restricted exclusively to curation provenance audit. | Strictly excluded from model features, prompt inputs, and target definitions. |
| **9. Span Recovery & Multi-Label Overlap** | **FINALIZED** | Offset recovery tested: 97.25% verbatim exact match, 2.75% (92 spans) case-insensitive match, 0% unmatched. Multi-label overlapping spans identified. | Downstream NLP models must employ multi-label classification or span-based extraction to prevent label collisions. |
