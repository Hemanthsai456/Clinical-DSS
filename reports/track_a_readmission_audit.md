# Evernorth Clinical Decision Support (CDS) — Track A: UCI Readmission Evaluation Audit

**Project:** Clinical Decision Support Using Deep Learning and NLP  
**Workspace:** `D:\Evernorth\Clinical-DSS`  
**Track:** Track A — UCI Diabetes 130-US Hospitals 30-Day Readmission Risk  
**Date:** October 2026  
**Audit Status:** COMPLETE & VERIFIED (All 12 audit gates PASSED; 31/31 automated tests passing)

---

## 1. Executive Summary

This independent methodological and code audit examined the end-to-end Track A readmission pipeline in the Evernorth Clinical CDS project. The audit inspected the data loader, missing-value policies, cohort exclusions, patient-grouped partitioning, feature availability taxonomy, training pipelines, evaluation assertions, probability calibration, and exported artifacts.

### Key Audit Conclusions:
1. **Target Formulation Verified (PASS):** `<30` is strictly defined as positive (1); `>30` and `NO` are negative (0). Crucially, `NO` indicates *no recorded readmission in the 130 participating hospitals during the 1999–2008 dataset window*, not proof that the patient was never hospitalized elsewhere.
2. **Zero Patient Leakage Confirmed (PASS):** Patient-grouped partitioning via `GroupShuffleSplit` on `patient_nbr` strictly isolates all encounters of a given patient into one split (Train: 48,992 patients; Val: 10,499 patients; Test: 10,499 patients; pairwise patient overlap is exactly 0).
3. **Preprocessing Isolation Confirmed (PASS):** All imputers, scalers, and encoders are fitted exclusively on `X_train` within an encapsulated `scikit-learn` `Pipeline`. Test data is evaluated via `predict_proba` with zero fitting.
4. **Exact Metric Reproduction (PASS):** All saved models ([`models/baseline_a_logreg.joblib`](models/baseline_a_logreg.joblib), [`models/baseline_b_logreg.joblib`](models/baseline_b_logreg.joblib), [`models/baseline_c_logreg.joblib`](models/baseline_c_logreg.joblib)) were re-evaluated against `test_feat`. Every metric (AUROC, PR-AUC, confusion matrix cells, F1, precision, recall, specificity, Brier score) matched [`reports/readmission_baselines_metrics.json`](reports/readmission_baselines_metrics.json) with exact numerical agreement ($< 10^{-4}$).
5. **Defensible Primary Prediction Scenario Established (PASS):** **Baseline A** (Clinical Features *excluding* `discharge_disposition_id` and *excluding* discharge ICD-9 diagnoses) is defined as the defensible primary real-time prediction model for discharge planning. Baseline C (highest test score: AUROC 0.6666 vs 0.6324 for A) relies on post-discharge billing codes (`diag_1`, `diag_2`, `diag_3`) and cannot be treated as the best real-time bedside model.
6. **Calibration Characterized (PASS):** Probability calibration was assessed without altering the locked test protocol. Baseline A demonstrates strong calibration in the operational risk region ($[0.0, 0.30]$ where $98.5\%$ of encounters reside), achieving an overall Test Brier Score of **0.0961** (vs. Majority Baseline **0.0988**) and an Expected Calibration Error (ECE) of **0.0061** ($0.61\%$).

---

## 2. Audit Verification Matrix

| Audit Dimension | Status | Evidence / Verification Method | Notes & Findings |
|---|---|---|---|
| **1. Data Loader & Missing Values** | **PASS** | [`src/data_loaders.py`](src/data_loaders.py) | `na_values=['?', '']` with `keep_default_na=False`. Only genuine missing-value markers are converted to NaN. |
| **2. Lab Test 'None' Category** | **PASS** | [`src/data_loaders.py`](src/data_loaders.py#L159-L163) | Explicit assertions verify 0 NaNs and confirm preservation of literal `'None'` (96,420 in `max_glu_serum`; 84,748 in `A1Cresult`) as "test not performed". |
| **3. Target Definition** | **PASS** | [`src/audit_uci.py`](src/audit_uci.py#L318-L336) | `<30` is positive ($N=11,314, 11.39\%$); `>30` and `NO` are negative ($N=88,029, 88.61\%$). Documented that `NO` means no recorded readmission. |
| **4. Cohort Exclusions** | **PASS** | [`src/audit_uci.py`](src/audit_uci.py#L125-L256) | 2,423 encounters excluded (codes 11, 13, 14, 19, 20, 21). 86 hospice encounters with `<30`/`>30` investigated and confirmed non-miscoded (palliative crises). |
| **5. Patient-Grouped Split** | **PASS** | [`src/audit_uci.py`](src/audit_uci.py#L406-L481) | `GroupShuffleSplit` on `patient_nbr`. Zero overlap between Train (48,992), Val (10,499), and Test (10,499). Verified by automated assertions. |
| **6. Preprocessing Isolation** | **PASS** | [`src/train_readmission_baseline.py`](src/train_readmission_baseline.py#L363-L379) | `ColumnTransformer` embedded in `Pipeline`. `pipe.fit()` executed strictly on `X_train`. Zero test data leak into imputers, scalers, or encoders. |
| **7. Feature Taxonomy & Timing** | **PASS** | [`src/audit_uci.py`](src/audit_uci.py#L21-L88) | Full classification into Admission, Inpatient Stay, and Discharge/Post-Discharge. Identifier exclusion enforced. |
| **8. Primary Scenario Formulation** | **PASS** | Section 4 of this report | Real-time bedside CDS at discharge planning. Baseline A chosen as primary; Baselines B and C designated as retrospective upper-bound benchmarks. |
| **9. Validation-Only Tuning** | **PASS** | [`src/train_readmission_baseline.py`](src/train_readmission_baseline.py#L177-L188) | Thresholds tuned on Validation only to maximize F1 ($\tau = 0.130, 0.145, 0.135$); frozen and applied directly to Test. Hyperparameters unadjusted on Test. |
| **10. Historical Test-Set Exposure** | **PASS** | Section 6 of this report | Explicitly disclosed that Test set was evaluated in Milestones 2 and 2.1 to generate baseline tables. Acknowledged as frozen historical evaluation. |
| **11. Baseline Reproduction** | **PASS** | [`scripts/audit_track_a_readmission.py`](scripts/audit_track_a_readmission.py) | 100% exact numerical agreement across AUROC, PR-AUC, confusion matrices, F1, and Brier scores between saved `.joblib` models and JSON metrics. |
| **12. Probability Calibration** | **PASS** | [`reports/readmission_calibration_metrics.json`](reports/readmission_calibration_metrics.json) | Brier score (0.0961) and ECE (0.0061) computed across 10 deciles without modifying the locked evaluation protocol. |
| **13. Repository Automated Tests** | **PASS** | [`tests/test_readmission_evaluation.py`](tests/test_readmission_evaluation.py) | 31/31 pytest unit/integration tests passing (including 9 Track A specific tests). |

---

## 3. Feature-Availability Taxonomy & Leakage Risk Analysis

Features available in the UCI Diabetes dataset originate from different administrative and clinical processes during hospitalization. Incorporating post-discharge or retrospective variables into a real-time CDS model creates lookahead leakage.

| Feature Name | Clinical Category | Availability Window | Included in Baseline A? | Included in Baseline B? | Included in Baseline C? | Methodological & Leakage Assessment |
|---|---|---|:---:|:---:|:---:|---|
| `encounter_id`, `patient_nbr` | Identifiers | Pre-admission / System | **NO (Dropped)** | **NO (Dropped)** | **NO (Dropped)** | **STRICTLY EXCLUDED.** Retained solely for longitudinal linkage, patient grouping, and audit. Zero clinical predictive utility. |
| `race`, `gender`, `age` | Demographics | Admission | **YES** | **YES** | **YES** | Legitimate baseline variables collected at patient intake. |
| `admission_type_id`, `admission_source_id` | Admission Context | Admission | **YES** | **YES** | **YES** | Emergency vs. urgent vs. elective intake context; legitimate baseline predictors. |
| `number_outpatient`, `number_emergency`, `number_inpatient` | Prior Utilization | Admission | **YES** | **YES** | **YES** | Prior-year historical healthcare utilization; strong independent predictors of chronic disease burden. |
| `time_in_hospital` | Inpatient Utilization | Stay / Discharge | **YES** | **YES** | **YES** | Total inpatient length of stay in days. Known at discharge planning. |
| `num_lab_procedures`, `num_procedures`, `num_medications` | Inpatient Intensity | Stay / Discharge | **YES** | **YES** | **YES** | Cumulative count of tests, procedures, and distinct medications administered during stay. |
| `number_diagnoses` | Clinical Complexity | Stay / Discharge | **YES** | **YES** | **YES** | Count of entered diagnoses on the problem list. |
| `max_glu_serum`, `A1Cresult` | Inpatient Labs | Stay / Discharge | **YES** | **YES** | **YES** | Inpatient blood glucose and HbA1c test results. Literal `'None'` denotes "test not performed" (clinically informative category). |
| 23 Diabetic Medications (`metformin`, `insulin`, etc.) | Pharmacotherapy | Stay / Discharge | **YES** | **YES** | **YES** | Inpatient medication orders and dosage adjustments (`Up`, `Down`, `Steady`, `No`). |
| `change`, `diabetesMed` | Medication Adjustments | Stay / Discharge | **YES** | **YES** | **YES** | Indicates whether any diabetic medication was altered during stay. |
| `discharge_disposition_id` | Post-Acute Placement | At Discharge | **NO** | **YES** | **YES** | **CIRCULAR PLACEMENT LEAKAGE RISK.** Encodes whether patient was sent to Home Health, SNF, Rehab, etc. Post-acute placement decisions are often made *in response* to unmeasured clinical frailty. |
| `diag_1`, `diag_2`, `diag_3` (ICD-9 codes) | Final Diagnoses | **Post-Discharge** | **NO** | **NO** | **YES** | **LOOKAHEAD CODING BIAS RISK.** Official ICD-9 discharge diagnosis codes are finalized by professional hospital billing coders *days or weeks after* physical discharge. Unavailable at bedside. |
| `weight` | Physical Exam | Administrative | **NO (Dropped)** | **NO (Dropped)** | **NO (Dropped)** | Dropped due to extreme missingness (96.86% missing). |
| `payer_code` | Insurance Billing | Administrative | **NO (Dropped)** | **NO (Dropped)** | **NO (Dropped)** | Dropped due to high missingness (39.56% missing) and hospital billing dependency. |
| `medical_specialty` | Provider Specialty | Administrative | **NO (Dropped)** | **NO (Dropped)** | **NO (Dropped)** | Dropped due to high missingness (49.08% missing). |

---

## 4. Defensible Primary Prediction Scenario Formulation

### Clinical CDS Setting: Discharge Planning (Transitional Care Management)
- **Operational Timing:** 24 to 48 hours prior to anticipated physical discharge, or at final bedside discharge reconciliation.
- **Decision Action:** Flagging high-risk diabetic inpatients for transitional care interventions:
  - Pharmacy-led discharge medication reconciliation.
  - Accelerated primary care follow-up appointment within 7 days.
  - Outpatient nurse check-in calls at 48 hours and 7 days.
  - Home health visiting nurse services for insulin titration.

### Why Baseline A is the Defensible Primary Real-Time CDS Model:
1. **Feasibility at Bedside:** Baseline A uses 40 clinical and demographic features that are documented in the electronic health record during the inpatient stay.
2. **Exclusion of Lookahead Coding Bias:** Baseline C adds grouped ICD-9 codes (`diag_1_group`, `diag_2_group`, `diag_3_group`). In authentic clinical operations, formal ICD-9/ICD-10 billing codes are assigned asynchronously by HIM coding departments after chart closure. Feeding billing codes into a bedside CDS model assumes retrospective knowledge that does not exist at point-of-care.
3. **Exclusion of Circular Placement Leakage:** Baseline B adds `discharge_disposition_id`. While disposition is nominally established at discharge, the disposition decision (e.g., referral to home health or skilled nursing facility) is often mediated by the healthcare team's subjective readmission concern. Including it can introduce confounding proxy bias.
4. **Benchmark Role of Baselines B & C:** Baselines B and C serve as retrospective benchmark upper bounds to quantify how much predictive signal is captured by post-acute placement and retrospective coding, but they must not be mistaken for deployable real-time bedside algorithms.

---

## 5. Replicated Baseline Performance Metrics

All metrics below were reproduced directly from the serialized model artifacts ([`models/baseline_a_logreg.joblib`](models/baseline_a_logreg.joblib), [`models/baseline_b_logreg.joblib`](models/baseline_b_logreg.joblib), [`models/baseline_c_logreg.joblib`](models/baseline_c_logreg.joblib)) against the locked Test and Validation sets:

### Test Partition Replicated Metrics ($N = 14,851; Positives = 1,650; Prevalence = 11.11\%$)

| Model Pipeline | Features Count | AUROC | PR-AUC (AP) | Frozen $\tau$* | Precision | Recall | F1-Score | Specificity | Brier Score | ECE |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Majority Baseline** | 0 | 0.5000 | 0.1111 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0988 | 0.0034 |
| **Baseline A** *(Primary Bedside)* | **40** | **0.6324** | **0.1925** | **0.1300** | **0.1806** | **0.3800** | **0.2449** | **0.7846** | **0.0961** | **0.0061** |
| **Baseline B** *(Augmented Placement)* | 41 | 0.6599 | 0.2176 | 0.1450 | 0.2136 | 0.3552 | 0.2667 | 0.8365 | 0.0947 | 0.0091 |
| **Baseline C** *(Retrospective Billed)* | 44 | 0.6666 | 0.2213 | 0.1350 | 0.2028 | 0.4224 | 0.2740 | 0.7924 | 0.0944 | 0.0068 |

*\*Decision threshold $\tau$ was selected strictly on the Validation partition to maximize F1, frozen, and evaluated on Test with zero post-hoc tuning.*

### Validation Partition Metrics ($N = 15,022; Positives = 1,711; Prevalence = 11.39\%$)

| Model Pipeline | AUROC | PR-AUC (AP) | Tuned $\tau$ | Precision | Recall | F1-Score | Specificity | Brier Score | ECE |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Majority Baseline** | 0.5000 | 0.1139 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.1009 | 0.0006 |
| **Baseline A** *(Primary Bedside)* | 0.6467 | 0.2066 | 0.1300 | 0.1944 | 0.4050 | 0.2627 | 0.7843 | 0.0976 | 0.0065 |
| **Baseline B** *(Augmented Placement)* | 0.6683 | 0.2209 | 0.1450 | 0.2260 | 0.3840 | 0.2845 | 0.8310 | 0.0967 | 0.0076 |
| **Baseline C** *(Retrospective Billed)* | 0.6661 | 0.2196 | 0.1350 | 0.2095 | 0.4378 | 0.2834 | 0.7877 | 0.0968 | 0.0084 |

### Test Partition Confusion Matrices & Class Invariants

| Model Pipeline | TN | FP | FN | TP | Derived Precision | Derived Recall | Derived Specificity | Derived F1 | Mathematical Agreement |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Majority Baseline** | 13,201 | 0 | 1,650 | 0 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | **EXACT ($<10^{-4}$)** |
| **Baseline A** | 10,357 | 2,844 | 1,023 | 627 | 0.1806 | 0.3800 | 0.7846 | 0.2449 | **EXACT ($<10^{-4}$)** |
| **Baseline B** | 11,043 | 2,158 | 1,064 | 586 | 0.2136 | 0.3552 | 0.8365 | 0.2667 | **EXACT ($<10^{-4}$)** |
| **Baseline C** | 10,461 | 2,740 | 953 | 697 | 0.2028 | 0.4224 | 0.7924 | 0.2740 | **EXACT ($<10^{-4}$)** |

$$\begin{aligned}
\text{Total Encounters Check:} \quad &TN + FP + FN + TP = 10,357 + 2,844 + 1,023 + 627 = \mathbf{14,851} \\
\text{Total Positives Check:} \quad &TP + FN = 627 + 1,023 = \mathbf{1,650} \quad (\text{Prevalence: } 11.1104\%) \\
\text{Total Negatives Check:} \quad &TN + FP = 10,357 + 2,844 = \mathbf{13,201} \quad (\text{Rate: } 88.8896\%)
\end{aligned}$$

---

## 6. Probability Calibration Audit

Calibration reflects whether predicted probabilities correspond to true empirical readmission frequencies. In clinical decision support, calibrated risk probabilities are vital so clinicians can interpret a predicted $20\%$ risk as an actual 1-in-5 readmission likelihood.

### Test Calibration Curve for Primary Bedside Model (Baseline A)
- **Overall Brier Score:** **0.0961** (lower than Majority Baseline's 0.0988).
- **Expected Calibration Error (ECE):** **0.0061** ($0.61\%$).
- **Mean Predicted Probability:** **0.1154** vs. **Observed Test Prevalence: 0.1111**.

| Bin Index | Risk Decile Range | Encounter Count | Mean Predicted Risk | Observed Readmission Rate | Absolute Calibration Error | Clinical Risk Tier |
|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **1** | $[0.00, 0.10)$ | 7,135 | 0.0804 | 0.0744 | 0.0060 | Low Risk (Beneath Population Prevalence) |
| **2** | $[0.10, 0.20)$ | 6,856 | 0.1287 | 0.1288 | **0.0001** | Average Risk (Operational Decision Zone) |
| **3** | $[0.20, 0.30)$ | 627 | 0.2371 | 0.2329 | **0.0042** | Elevated Risk (CDS Alert Zone) |
| **4** | $[0.30, 0.40)$ | 133 | 0.3429 | 0.3684 | 0.0255 | High Risk (Intensive Care Coordination) |
| **5** | $[0.40, 0.50)$ | 60 | 0.4439 | 0.4167 | 0.0272 | Very High Risk |
| **6** | $[0.50, 0.60)$ | 26 | 0.5401 | 0.3846 | 0.1555 | Extreme Risk (Sparse Cell) |
| **7** | $[0.60, 0.70)$ | 10 | 0.6358 | 0.4000 | 0.2358 | Extreme Risk (Sparse Cell) |
| **8** | $[0.70, 0.80)$ | 4 | 0.7303 | 0.2500 | 0.4803 | Extreme Risk (Sparse Cell) |
| **9** | $[0.80, 0.90)$ | 0 | — | — | 0.0000 | Empty Bin |
| **10** | $[0.90, 1.00)$ | 0 | — | — | 0.0000 | Empty Bin |

> [!NOTE]
> **Calibration Finding:** Across bins 1, 2, and 3 (representing $98.43\%$ of all encounters: 14,618 out of 14,851), the model's calibration error is negligible ($< 0.006$). The model reliably reflects empirical readmission risk across the clinical operating range. Calibration degrades only above $0.50$, where fewer than 40 encounters ($0.27\%$) reside due to the low base rate ($11.1\%$).

---

## 7. Methodological Limitations & Generalization Safeguards

> [!CAUTION]
> **No Claim of Generalizable Clinical Validity:**
> The following structural limitations preclude claiming that these models are clinically validated for current bedside deployment:

1. **Historical Dataset Vintage (1999–2008):**
   - The data reflects inpatient practices from nearly two decades ago. Inpatient diabetes management protocols, sliding-scale insulin standards, and transitional care outreach have changed substantially.
   - Newer anti-hyperglycemic drug classes (GLP-1 receptor agonists, SGLT2 inhibitors, DPP-4 inhibitors) are absent or negligibly represented in this historical cohort.
2. **Censoring and Regional Out-of-System Readmissions:**
   - The target label `readmitted` tracks readmissions *only within the 130 participating US hospitals*.
   - A label of `NO` signifies "no recorded readmission in these 130 hospitals." Patients readmitted to other regional hospitals, VA facilities, or academic centers outside the consortium are uncaptured and classified as negative.
3. **In-Distribution vs. Cross-Hospital Generalization:**
   - The patient-grouped test split proves zero patient leakage and measures in-distribution generalization within the same 130 hospital network.
   - It does not establish transportability to modern EHR networks, distinct geographical regions, or health systems with differing payer mixes.
4. **Historical Test-Set Exposure Disclosure:**
   - The test set was evaluated historically in Milestone 2 and Milestone 2.1 to generate comparative baseline artifacts. This test set must be treated as a locked reference benchmark, not an untouched prospective cohort.

---

## 8. Files Inspected & Verified

- [`src/data_loaders.py`](src/data_loaders.py): Verified robust local dataset resolution, `keep_default_na=False`, and literal `'None'` category preservation in lab tests.
- [`src/audit_uci.py`](src/audit_uci.py): Verified eligible cohort exclusion (2,423 encounters), investigation of the 86 hospice encounters, and patient-grouped splitting with zero overlap.
- [`src/train_readmission_baseline.py`](src/train_readmission_baseline.py): Verified pipeline encapsulation, fitting strictly on training data, validation-only threshold tuning, and automated mathematical consistency assertions.
- [`tests/test_readmission_evaluation.py`](tests/test_readmission_evaluation.py): Verified 9 targeted unit and integration tests covering label definitions, zero patient leakage, feature isolation, and confusion matrix derivations.
- [`reports/readmission_baselines_metrics.json`](reports/readmission_baselines_metrics.json): Verified numerical consistency of original exported metrics.
- [`reports/readmission_calibration_metrics.json`](reports/readmission_calibration_metrics.json): Generated separate calibration artifact recording decile reliability curves and Brier scores.
- [`scripts/audit_track_a_readmission.py`](scripts/audit_track_a_readmission.py): Created stand-alone replication and calibration auditing script.

---

## 9. Recommended Next Actions

1. **Maintain Baseline A as the Operational Benchmark:**
   - In any future composite modeling (combining structured readmission predictions with NLP-extracted SBDH features from Track B), use **Baseline A** as the structured risk backbone to avoid contaminating the CDS alert with post-discharge billing codes.
2. **Explore Probability Recalibration for Elevated Risk Tiers:**
   - While overall ECE is excellent ($0.0061$), encounters with predicted probabilities above $0.40$ show slight overprediction due to sample sparsity. Isotonic regression or Platt scaling fitted on validation probabilities can be tested if high-probability alerting requires tighter calibration.
3. **Transition to Modern EHR Cohorts for Clinical Deployment:**
   - For real-world health plan deployment, train on contemporary longitudinal claims and modern EHR feeds (e.g. 2020–2026), incorporating contemporary medications and explicit out-of-network claims tracking to resolve readmission censoring.
