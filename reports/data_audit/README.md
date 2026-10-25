# Evernorth Clinical Decision Support (CDS) — Milestone 1 & 1.1: Data Audit & Modeling Readiness

**Research Prototype:** Clinical Decision Support Using Deep Learning and NLP  
**Milestone:** 1.0 (Audit) & 1.1 (Modeling Readiness Gate)  
**Status:** ALL GATES PASSED — Audited Offline Using Local Datasets  

---

## 1. Executive Summary

Milestone 1 and 1.1 establish a validated, reproducible, leak-free data foundation for the Evernorth Clinical CDS prototype:
1. **UCI Diabetes 130-US Hospitals (1999–2008):** Inpatient diabetic encounters used to formulate a 30-day unplanned readmission prediction task at the point of discharge planning.
2. **Synth-SBDH (bio-nlp-umass):** 10,499 synthetic clinical notes annotated for 15 Social and Behavioral Determinants of Health (SBDH) categories across `train`, `val`, `test`, and `test_reviewed`.

Zero mock or hallucinated values were used. All statistics below reflect executed code on verified local files.

---

## 2. UCI Cohort Policy & Feature Pipeline Audit

### 2.1 Laboratory Result Categorical Preservation (`max_glu_serum` & `A1Cresult`)
- Literal `"None"` in `max_glu_serum` (96,420 encounters, 94.75%) and `A1Cresult` (84,748 encounters, 83.28%) represents **"Test Not Performed/Measured"**.
- It is a valid, clinically informative category with **0.00% missingness**, NOT a corrupted missing value.
- Loaders enforce `keep_default_na=False, na_values=["?", ""]`, converting only actual missing markers (`?`) to `NaN`.

### 2.2 Dynamic Discharge Disposition Verification
Using `D:\Evernorth\UCI-Diabetes\IDS_mapping.csv`, discharge dispositions indicating in-hospital mortality or palliative hospice care were dynamically parsed and verified:
- `11`: **Expired** (in-hospital death)
- `13`: **Hospice / home**
- `14`: **Hospice / medical facility**
- `19`: **Expired at home. Medicaid only, hospice.**
- `20`: **Expired in a medical facility. Medicaid only, hospice.**
- `21`: **Expired, place unknown. Medicaid only, hospice.**

### 2.3 Cohort Filtering Comparison Table

| Metric | Unfiltered Baseline Cohort | Excluded Cohort (Death / Hospice) | Eligible Modeling Cohort | Notes & Impact |
|---|---|---|---|---|
| **Total Encounters** | **101,766** | **2,423** (2.38%) | **99,343** (97.62%) | Excluded from primary readmission analysis. |
| **Unique Patients** | **71,518** | **2,399** (3.35%) | **69,990** (97.86%) | 871 patients had earlier non-fatal encounters retained. |
| **Readmitted `<30` (Positive)** | **11,357** | **43** (0.38%) | **11,314** (99.62%) | 43 hospice crisis readmissions excluded under policy. |
| **Readmitted `>30`** | **35,545** | **43** (0.12%) | **35,502** (99.88%) | Negligible change. |
| **Readmitted `NO`** | **54,864** | **2,337** (4.26%) | **52,527** (95.74%) | 96.45% of excluded cases had `NO`, removing massive leakage. |
| **Positive Class Prevalence (`<30`)** | **11.16%** | **1.77%** | **11.39%** | Positive rate increases slightly (11.16% -> 11.39%). |
| **Imbalance Ratio (Neg : Pos)** | **7.96 : 1** | **55.35 : 1** | **7.78 : 1** | Clinically realistic class ratio. |

### 2.4 Investigation of the 86 Excluded Readmitted Encounters
- **Breakdown by Disposition:**
  - Code 13 (`Hospice / home`): 55 encounters (19 `<30`, 36 `>30`).
  - Code 14 (`Hospice / medical facility`): 31 encounters (24 `<30`, 7 `>30`).
  - Codes 11, 19, 20, 21 (True In-Hospital Deaths): **Exactly 0 readmissions** (all 2,337 are `NO`).
- **Clinical Profile:** 81 unique patients, 75.6% ER admissions, serious end-stage conditions (CHF, AMI, Sepsis, Cancer).
- **Evaluation Conclusion: NOT MISCODED.** Live patients discharged to hospice who suffered acute palliative crises, leading to re-hospitalization.
- **Policy Rationale:** Standard health-plan quality metric policies (e.g. CMS HRRP) exclude all hospice discharges from readmission penalty benchmarks because palliative goals are comfort-oriented rather than curative.

### 2.5 Prediction Time Assumption & Feature Pipeline Sensitivity Plan
- **Prediction Horizon:** Inpatient discharge (Transitional Care Management planning). The model identifies high-risk diabetic patients at discharge to trigger outpatient transitional care interventions.
- **Excluded Features:** Identifiers (`encounter_id`, `patient_nbr`) and sparse administrative fields (`weight`, `payer_code`) are strictly excluded from model feature matrices.
- **Explicit Sensitivity Comparison Plan:**
  - **Model Baseline A (Standard Clinical Baseline):** Model without `discharge_disposition_id`.
  - **Model Baseline B (Augmented Disposition Model):** Model with `discharge_disposition_id`.
  - Directly isolates and measures predictive gain against circular placement leakage risk.
  - Final billing diagnoses (`diag_1`, `diag_2`, `diag_3`) will similarly be evaluated with and without discharge coding to assess post-hoc coding lookahead bias.

### 2.6 Patient-Grouped Splitting on Eligible Modeling Cohort
Using `GroupShuffleSplit` on `patient_nbr` across the 99,343 eligible encounters:
- **Train (70%):** 69,470 encounters | 48,992 unique patients | 11.45% positive rate
- **Validation (15%):** 15,022 encounters | 10,499 unique patients | 11.39% positive rate
- **Test (15%):** 14,851 encounters | 10,499 unique patients | 11.11% positive rate
- **Overlap Verification:** **Zero patient overlap across all three partitions (ASSERTED AND PASSED)**.
- **Protocol Rule:** No encoders, imputers, scalers, or feature selectors may be fit on validation or test partitions.

---

## 3. Synth-SBDH Evaluation Contract & Span Alignment

### 3.1 Test vs. Expert-Reviewed Test Relationship
- **Strict Subset Assertion:** `rev_ids.issubset(test_ids) == True` (**PASSED**). All 1,732 reviewed instances originated in the standard test set.
- **23 Omitted Records:** Identified and verified: `[7028, 7170, 7181, 7256, 7368, 7454, 7468, 7484, 7614, 7692, 7962, 8009, 8215, 8226, 8261, 8285, 8421, 8476, 8489, 8573, 8586, 8676, 8706]`. These represent synthetic items where expert annotators determined that no genuine SBDH cues existed.
- **Encoding Normalization:** In `ex_no` 8169 and 8256, corrupted apostrophes (`doesnt`) in standard test were corrected to clean ASCII (`doesn't`) in reviewed test.
- **Primary Benchmark:** `synth_sbdh_test_reviewed.csv` (1,732 rows) is designated as the primary gold-standard evaluation benchmark. It must remain strictly isolated from all model training, hyperparameter tuning, and threshold selection.
- **Role of `Operation`:** The `Operation` column (`keep`, `add`, `correct`) is strictly curation provenance metadata. It is **never** used as a model feature, prompt input, or target.
- **Cross-Split Overlap Clarification:** Independent splits (`train`, `val`, `test`) have **zero text overlap**. `test_reviewed` is an intentionally related curated subset of `test`.

### 3.2 Span-Alignment Quality & Multi-Label Overlap
- **Tuple Alignment:** 100% of rows exhibit exact alignment across `||`-delimited fields (`Textspan`, `SBDH`, `Presence`, `Period`, `Reasoning`) with **0 length mismatches** across all 10,499 records.
- **Detailed Export of 92 Casing Cases:**
  - In `test_reviewed`, exactly 92 spans differ by letter casing due to expert normalization (e.g., `'Patient\'s wheelchair was stolen'` -> `'patient\'s wheelchair was stolen'`).
  - **100% of these 92 spans match verbatim under case-insensitivity.**
  - **0 spans are genuinely absent.**
  - Exported to: [`reports/data_audit/reviewed_test_casing_cases.json`](reviewed_test_casing_cases.json).
- **Ambiguous Repeated Spans:**
  - Exactly 8 instances in reviewed-test contain spans that appear multiple times within the text (e.g., `'pain'` in notes with both `'chronic pain'` and `'pain management'`).
- **Multi-Label Overlapping Spans:**
  - In 38 rows of `test_reviewed` and 33 rows of `train`, identical text phrases are annotated for multiple SBDH categories (e.g., `cannot afford to continue physical therapy` tagged as both `financial insecurity` and `barriers to care`).
  - Downstream token classification or span extraction models must support multi-label predictions to prevent label collisions.

---

## 4. Deliverables & Artifacts Index

| Artifact | Location | Purpose |
|---|---|---|
| **Modeling Readiness Report** | [`reports/data_audit/modeling_readiness_report.md`](modeling_readiness_report.md) | Formal readiness gate summary for independent Claude review. |
| **Reviewed Casing Cases (JSON)** | [`reports/data_audit/reviewed_test_casing_cases.json`](reviewed_test_casing_cases.json) | Export of all 92 casing-normalization cases with character offsets. |
| **UCI Audit Summary (JSON)** | [`reports/data_audit/uci_diabetes_audit_summary.json`](uci_diabetes_audit_summary.json) | Machine-readable metrics for UCI Diabetes and eligible cohort. |
| **Synth-SBDH Audit Summary (JSON)** | [`reports/data_audit/synth_sbdh_audit_summary.json`](synth_sbdh_audit_summary.json) | Machine-readable metrics for Synth-SBDH contracts and splits. |
| **Executed Audit Notebook** | [`notebooks/01_data_audit.ipynb`](../../notebooks/01_data_audit.ipynb) | Complete Google Colab-compatible notebook with all cell outputs populated. |
| **Audit Runner Tool** | [`src/run_data_audit.py`](../../src/run_data_audit.py) | Automated CLI audit script with automated assertions. |
| **Readiness Audit Script** | [`src/cohort_and_span_audit.py`](../../src/cohort_and_span_audit.py) | Standalone verification script for Gate 1.1 checks. |
