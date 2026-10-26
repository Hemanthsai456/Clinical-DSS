# Evernorth Clinical Decision Support (CDS) — Final Technical Report

**Project:** Clinical Decision Support Using Deep Learning and Natural Language Processing  
**Workspace:** `D:\Evernorth\Clinical-DSS`  
**Date:** October 2026  
**Document Type:** Final Technical Engineering & Research Report  
**Git Baseline Commit:** `7dd92b5bda82042f5009af52ab7013223df567b4`  
**Status:** Verified Educational & Research Prototype  

---

## 1. Executive Summary

This report documents the design, rigorous audit, empirical evaluation, and baseline freeze of the **Evernorth Clinical Decision Support (CDS)** research prototype. The system investigates machine learning and natural language processing approaches to two fundamental challenges in health plan care coordination and transitional care management:

1. **Track A — Inpatient 30-Day Readmission Risk Estimation:**  
   Formulated as a structured binary classification task on diabetic inpatient admissions from the **UCI Diabetes 130-US Hospitals (1999–2008)** dataset. The eligible modeling cohort consists of **99,343 encounters** (69,990 unique patients) after excluding 2,423 in-hospital mortality and hospice transfers. Using strict patient-grouped partitioning (`GroupShuffleSplit`, random state 42; 0 patient leakage), we evaluated three logistic regression baselines.  
   - **Baseline A (Primary Bedside Model):** Incorporates 40 pre-admission and inpatient clinical features available during discharge planning, excluding post-acute placement and post-discharge billing codes. It achieves an **AUROC of 0.6324**, **PR-AUC of 0.1925** (vs. 0.1111 population prevalence), **F1-score of 0.2449** (recall 0.3800, precision 0.1806, specificity 0.7846) at a validation-calibrated threshold of $\tau = 0.130$, with strong probability calibration (**Brier score 0.0961**, **ECE 0.0061**).  
   - **Baselines B & C:** Add discharge disposition (AUROC 0.6599) and retrospective ICD-9 billing codes (AUROC 0.6666). While scoring higher retrospectively, they introduce lookahead coding bias and circular placement leakage and are designated strictly as retrospective benchmark upper bounds.

2. **Track B — Social and Behavioral Determinants of Health (SBDH) Extraction:**  
   Formulated as a multi-label token-level sequence tagging and category classification task on the **Synth-SBDH** benchmark (bio-nlp-umass/Synth-SBDH; $N = 1,732$ reviewed test notes). The architecture employs a pretrained `distilroberta-base` backbone with **45 independent binary sigmoid logits** ($15 \text{ SBDH categories} \times \{B, I, O\}$), enabling multiple overlapping categories on identical spans (e.g., `financial insecurity` and `barriers to care`).  
   - Evaluated under strict dual-bound accounting and Kuhn's **Maximum Bipartite 1-to-1 Matching** (order-invariant), the 3-epoch GPU Colab-trained model (`best_model.pt`) achieves a **Relaxed Span F1 of 0.8330** (precision 0.8816, recall 0.7895) and **Exact Span F1 of 0.5105** (precision 0.5403, recall 0.4838) at threshold $\theta = 0.30$. This represents a substantial improvement over the deterministic rules baseline (Relaxed F1 0.5163, Exact F1 0.2642).  
   - All 5 unresolved ambiguous test annotations are conservatively penalized as False Negatives, establishing rigorous worst-case lower bounds (Relaxed F1 0.8324, Exact F1 0.5101).

---

## 2. Problem Statement & Project Objectives

Unplanned 30-day hospital readmissions represent a major driver of preventable morbidity, mortality, and healthcare expenditures in diabetes care. While structured electronic health record (EHR) data captures physiological parameters, inpatient utilization, and laboratory results, adverse social and behavioral determinants of health (SBDH)—such as housing instability, food insecurity, financial barriers to therapy, and social isolation—frequently go unrecorded in structured diagnostic fields and reside exclusively in unstructured clinical narratives.

### Engineering Objectives:
1. **Develop Real-Time CDS Readmission Models:** Build a reproducible structured predictive model to stratify 30-day readmission risk using only variables legitimately accessible prior to physical hospital departure.
2. **Eliminate Data Leakage:** Enforce strict patient-level isolation between training, validation, and test partitions, and isolate real-time bedside clinical features from retrospective billing codes and post-acute disposition proxies.
3. **Automate SBDH Extraction from Unstructured Text:** Implement a deep learning architecture capable of identifying clinical text spans indicating social vulnerability across 15 standard SBDH domains, supporting co-occurring categories on identical text spans.
4. **Establish Audit Integrity & Reproducibility:** Replace unconstrained greedy matching with globally optimal, order-invariant bipartite matching, track every raw gold annotation through auditable accounting invariants, and provide fully documented, test-verified code.

---

## 3. System Scope & Non-Clinical Use Disclaimer

> [!CAUTION]
> **MANDATORY REGULATORY AND CLINICAL DISCLAIMER:**  
> This software system, associated machine learning models, and interactive prototype are developed strictly as an **educational and research prototype**.
> 
> - **NOT A MEDICAL DEVICE:** This system has **not** been evaluated, cleared, or approved by the U.S. Food and Drug Administration (FDA) or any regulatory body under Section 510(k), De Novo classification, or Pre-Market Approval (PMA).
> - **NO CLINICAL DECISION-MAKING:** The risk estimates, classification outputs, and extracted text spans must **never** be used for clinical diagnosis, patient management, triage, treatment selection, discharge authorization, or medication modification.
> - **NO REAL-WORLD PROSPECTIVE VALIDATION:** The models were evaluated on historical retrospective data (1999–2008) and synthetic text. Passing automated test suites confirms software verification and numerical consistency; it does **not** establish clinical safety, transportability, or prospective efficacy.
> - **PRIVACY & PHI SAFEGUARD:** Real protected health information (PHI) must never be entered into the prototype or committed to version control. All interactive inference executes locally without network transmission.

---

## 4. Dataset Descriptions, Provenance & Limitations

The project investigates two completely independent datasets, each designed for a specific sub-problem:

| Dimension | Track A: UCI Diabetes 130-US Hospitals | Track B: Synth-SBDH Benchmark |
|---|---|---|
| **Data Modality** | Tabular, structured electronic health records | Unstructured narrative text notes |
| **Study Period** | Historical (1999–2008, 10-year collection window) | Contemporary synthetic dataset (published 2024) |
| **Source / Provenance** | 130 participating non-profit hospitals in the US; hosted by UCI Machine Learning Repository (Dataset ID 296; Strack et al., 2014) | BioNLP Lab, University of Massachusetts Amherst (Hugging Face: `bio-nlp-umass/Synth-SBDH`; Soni et al., 2024) |
| **Patient Population** | Real historical diabetic inpatient encounters | **100% Synthetic LLM-generated clinical notes** simulating discharge summaries and social work notes |
| **Volume (Raw)** | 101,766 encounters across 71,518 unique patients | 8,744 total synthetic notes (Train: 6,136; Val: 876; Test: 1,732) |
| **Eligible / Evaluated** | 99,343 encounters (69,990 patients) after mortality/hospice exclusions | 1,732 expert-reviewed test notes (`synth_sbdh_test_reviewed.csv`; 3,345 raw annotations) |
| **Licensing** | Creative Commons Attribution 4.0 International (CC BY 4.0) | Open access research license (bio-nlp-umass) |
| **Primary Limitation** | Historical vintage (absent SGLT2i/GLP-1RA); regional readmission censoring (unparticipating facilities unrecorded) | Synthetic text may under-represent authentic clinical documentation noise, abbreviations, and real EHR narrative complexity |

---

## 5. Track A Methodology: UCI Readmission

### 5.1 Cohort Filtering & Target Formulation
The raw dataset contains 101,766 inpatient encounters. Standard health plan quality measures (e.g., CMS Hospital Readmissions Reduction Program) exclude in-hospital deaths and transfers to hospice from 30-day readmission performance benchmarks.

1. **Verified Exclusion Codes (`IDS_mapping.csv`):**
   - Dispositions 11, 19, 20, 21: In-hospital mortality / expired ($N = 1,652$ encounters; all have `readmitted = 'NO'`).
   - Dispositions 13, 14: Hospice transfers (Home hospice: 399 encounters; Medical facility hospice: 372 encounters).
   - Total Exclusions: **2,423 encounters** (2,399 unique patients).
2. **Analysis of the 86 Hospice Encounters with Readmission Labels:**
   - 86 encounters discharged to hospice had readmission labels (`<30`: 43; `>30`: 43). Detailed audit confirmed these records are **not miscoded**: all 86 were discharged alive to hospice (dispositions 13 and 14); zero in-hospital deaths had readmission labels. Patients discharged to hospice occasionally experience symptom flare-ups or hospice revocation leading to acute rehospitalization. They are excluded solely for policy alignment, not data corruption.
3. **Eligible Modeling Cohort:**
   - **99,343 encounters** (69,990 unique patients).
   - Target Label Distribution: `NO` = 52,527 ($52.87\%$), `>30` = 35,502 ($35.74\%$), `<30` = 11,314 ($11.39\%$).
   - Binary Target Definition: Class 1 = `<30` (early readmission); Class 0 = `>30` or `NO` ($88.61\%$).
   - **Critical Clinical Nuance:** `NO` denotes absence of readmission *within the 130 participating hospitals in the 1999–2008 window*; it does not guarantee the patient was never hospitalized at an unparticipating facility.

### 5.2 Patient-Grouped Partitioning
To prevent optimistic data leakage from repeat encounters of chronic diabetic patients, partitions were generated using `GroupShuffleSplit` on `patient_nbr` (random state 42, test size 0.15, validation size 0.15):
- **Train Split (70%):** 69,470 encounters | 48,992 patients | 7,953 positive ($11.45\%$) | 61,517 negative
- **Validation Split (15%):** 15,022 encounters | 10,499 patients | 1,711 positive ($11.39\%$) | 13,311 negative
- **Test Split (15%):** 14,851 encounters | 10,499 patients | 1,650 positive ($11.11\%$) | 13,201 negative
- **Leakage Verification:** Pairwise intersection of patient sets is **identically zero** ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$).

### 5.3 Feature Availability Taxonomy & Sensitivity Baselines

| Feature Group | Included Features | In Baseline A? | In Baseline B? | In Baseline C? | Operational Justification |
|---|---|:---:|:---:|:---:|---|
| **Identifiers** | `encounter_id`, `patient_nbr` | **Dropped** | **Dropped** | **Dropped** | Excluded to prevent spurious correlation; retained only for grouped splitting. |
| **Demographics** | `race`, `gender`, `age` | **Yes** | **Yes** | **Yes** | Recorded at admission intake. |
| **Admission Context** | `admission_type_id`, `admission_source_id` | **Yes** | **Yes** | **Yes** | Emergency, urgent, or elective status. |
| **Prior Utilization** | `number_outpatient`, `number_emergency`, `number_inpatient` | **Yes** | **Yes** | **Yes** | Utilization in the 12 months preceding admission. |
| **Inpatient Intensity** | `time_in_hospital`, `num_lab_procedures`, `num_procedures`, `num_medications`, `number_diagnoses` | **Yes** | **Yes** | **Yes** | Cumulative metrics known at discharge planning. |
| **Glycemic Labs** | `max_glu_serum`, `A1Cresult` | **Yes** | **Yes** | **Yes** | Inpatient glucose and HbA1c tests. Literal `'None'` is preserved as a clinically informative category ("test not performed"). |
| **Pharmacotherapy** | 23 Diabetic Medications, `change`, `diabetesMed` | **Yes** | **Yes** | **Yes** | Inpatient medications and dosage titrations. |
| **Post-Acute Placement** | `discharge_disposition_id` | **No** | **Yes** | **Yes** | **Circular Placement Leakage:** Placement to SNF or home health is often triggered by provider readmission concern. |
| **Discharge Diagnoses** | `diag_1_group`, `diag_2_group`, `diag_3_group` | **No** | **No** | **Yes** | **Lookahead Coding Bias:** Final ICD-9 billing codes are assigned days/weeks *after* discharge by professional HIM coders. |
| **Administrative Sparse** | `weight` (96.86% miss), `payer_code` (39.56% miss), `medical_specialty` (49.08% miss) | **Dropped** | **Dropped** | **Dropped** | Excluded due to severe administrative missingness. |

### 5.4 Preprocessing & Threshold Selection
All preprocessing is encapsulated in a `scikit-learn` `Pipeline`:
- **Numerical Features (8):** Median imputation (`SimpleImputer`) followed by standard scaling (`StandardScaler`).
- **Categorical Features (32 in Baseline A):** Constant imputation (`fill_value="Missing"`) followed by one-hot encoding (`OneHotEncoder(handle_unknown="ignore", sparse_output=False)`).
- **Isolation Guarantee:** The entire pipeline (`ColumnTransformer` + `LogisticRegression(C=1.0, max_iter=1000)`) is fitted strictly on `X_train`. Zero test data is used for fitting.
- **Threshold Policy:** Decision thresholds ($\tau_A = 0.130, \tau_B = 0.145, \tau_C = 0.135$) were calibrated via grid search on the Validation partition to maximize F1-score and frozen for test evaluation.

---

## 6. Track A Results, Calibration & Primary Model Selection

### 6.1 Performance Comparison on Test Partition ($N = 14,851$)

| Model Pipeline | Features Count | AUROC | PR-AUC (AP) | Frozen $\tau$ | Precision | Recall | F1-Score | Specificity | Brier Score | ECE |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Majority Baseline** | 0 | 0.5000 | 0.1111 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0988 | 0.0034 |
| **Baseline A (Primary Bedside)** | **40** | **0.6324** | **0.1925** | **0.1300** | **0.1806** | **0.3800** | **0.2449** | **0.7846** | **0.0961** | **0.0061** |
| **Baseline B (Placement Benchmark)** | 41 | 0.6599 | 0.2176 | 0.1450 | 0.2136 | 0.3552 | 0.2667 | 0.8365 | 0.0947 | 0.0091 |
| **Baseline C (Retrospective Billed)** | 44 | 0.6666 | 0.2213 | 0.1350 | 0.2028 | 0.4224 | 0.2740 | 0.7924 | 0.0944 | 0.0068 |

### 6.2 Confusion Matrix Derivations (Baseline A on Test Partition)
$$\begin{aligned}
TN &= 10,357, \quad FP = 2,844, \quad FN = 1,023, \quad TP = 627 \\
\text{Total Encounters Check:} \quad &TN + FP + FN + TP = 10,357 + 2,844 + 1,023 + 627 = \mathbf{14,851} \\
\text{Total Positives Check:} \quad &TP + FN = 627 + 1,023 = \mathbf{1,650} \quad (\text{Prevalence: } 11.1104\%) \\
\text{Precision} &= \frac{627}{627 + 2,844} = \mathbf{0.180639...} \implies \mathbf{0.1806} \\
\text{Recall} &= \frac{627}{627 + 1,023} = \frac{627}{1,650} = \mathbf{0.380000...} \implies \mathbf{0.3800} \\
\text{Specificity} &= \frac{10,357}{10,357 + 2,844} = \frac{10,357}{13,201} = \mathbf{0.784561...} \implies \mathbf{0.7846} \\
F_1 &= \frac{2 \times 0.180639 \times 0.3800}{0.180639 + 0.3800} = \mathbf{0.244917...} \implies \mathbf{0.2449}
\end{aligned}$$

### 6.3 Probability Calibration (Baseline A)
- **Overall Brier Score:** **0.0961** (lower than Majority Baseline's 0.0988).
- **Expected Calibration Error (ECE):** **0.0061** ($0.61\%$).
- **Mean Predicted Risk:** 0.1154 vs. Observed Test Prevalence: 0.1111.
- In the primary operating decile ($[0.10, 0.20)$, containing 6,856 encounters), mean predicted risk is **0.1287** versus an observed readmission rate of **0.1288** (absolute calibration error of only **0.0001**).

### 6.4 Selected Primary Model Justification
**Baseline A is designated as the primary real-time bedside model.**  
Although Baseline C achieves higher retrospective discrimination (AUROC 0.6666 vs. 0.6324), it uses official ICD-9 diagnosis codes (`diag_1`, `diag_2`, `diag_3`). In hospital operations, formal ICD billing codes are assigned asynchronously by medical records coders days after patient discharge. Feeding retrospective billing codes into a discharge-planning CDS tool assumes lookahead knowledge that does not exist at the bedside. Baseline B includes discharge disposition, which acts as a circular proxy for provider concern. Therefore, **Baseline A represents the only realistic, defensible real-time bedside clinical model**.

---

## 7. Track B Methodology: SBDH Extraction

### 7.1 Architecture: 45-Class Independent Sigmoid Tagging
Traditional named entity recognition taggers use a single mutually exclusive BIO softmax over all classes ($2C + 1$ outputs). In clinical text, social determinants frequently co-occur within the exact same phrase (e.g., `"cannot afford to continue physical therapy"` is annotated with both `financial insecurity` and `barriers to care`). A single-label softmax forces the model to drop one valid category.

Our architecture ([src/transformer_model.py](file:///d:/Evernorth/Clinical-DSS/src/transformer_model.py)) employs:
- **Encoder:** `distilbert/distilroberta-base` (82M parameters, 6 layers, hidden dimension 768).
- **Classification Head:** Linear projection layer ($768 \to 45$ outputs: 15 SBDH categories $\times \{B, I, O\}$).
- **Activation:** Element-wise independent sigmoid $\sigma(z_{t, c, tag})$.
- **Loss Function:** Masked Binary Cross-Entropy with Logits:
  $$\mathcal{L} = \frac{1}{\sum_{t, k} M_{t, k}} \sum_{t=1}^{T} \sum_{k=1}^{45} M_{t, k} \cdot \text{BCE}(z_{t, k}, y_{t, k})$$

### 7.2 Loss Masking & Training Ambiguity (`ex_no 1192`)
In training note `ex_no 1192`, the gold annotation specifies `span="pain"`, but `"pain"` appears twice in the text (`[10:14]` and `[61:65]`). Arbitrarily picking one candidate would penalize the model for recognizing an equally valid mention.  
**Safeguard:** For `ex_no 1192`, candidate tokens across both intervals receive $M_{t, k} = 0.0$ for category `pain`. The candidate tokens are masked out of the gradient calculation entirely.

### 7.3 Evaluator Safeguard: Maximum Bipartite Matching
In relaxed span evaluation, greedy sequential matching is sensitive to prediction ordering when predicted spans overlap multiple gold intervals. We implemented **Kuhn's augmenting path algorithm for Maximum Bipartite Matching** ([src/sbdh_rules_baseline.py](file:///d:/Evernorth/Clinical-DSS/src/sbdh_rules_baseline.py)):
- **Order Invariance:** Permuting predictions yields identical True Positive counts.
- **Global Optimality:** Maximizes the cardinality of valid 1-to-1 matches under category agreement.

---

## 8. Track B Results: Rules vs. Transformer

Evaluated against `synth_sbdh_test_reviewed.csv` ($N = 1,732$ notes, 3,340 resolvable gold spans, 5 ambiguous cases):

| Evaluation Metric | Rules Baseline (Milestone 2) | Full 3-Epoch Colab Transformer Checkpoint | Absolute Delta |
|---|:---:|:---:|:---:|
| **Training Setup** | Deterministic Regexes | DistilRoBERTa (6,136 notes, 3 epochs GPU) | Full Supervised Learning |
| **Decision Threshold** | Deterministic | Calibrated on Validation at **$\theta = 0.30$** | Frozen for Test |
| **Total Predicted Spans** | 1,444 | **2,991** | +1,547 |
| **Exact Match True Positives (TP)** | 632 | **1,616** | **+984** |
| **Exact Match False Positives (FP)** | 812 | **1,375** | +563 |
| **Exact Match False Negatives (FN)** | 2,708 | **1,724** | **-984** |
| **Exact Precision (Resolvable)** | 0.4377 | **0.5403** | **+0.1026** |
| **Exact Recall (Resolvable)** | 0.1892 | **0.4838** | **+0.2946** |
| **Exact F1 (Resolvable Subset, $N=3,340$)** | **0.2642** | **0.5105** | **+0.2463** |
| **Exact F1 (Worst-Case Bound, $N=3,345$)** | **0.2639** | **0.5101** | **+0.2462** |
| **Relaxed Match True Positives (TP)** | 1,235 | **2,637** | **+1,402** |
| **Relaxed Match False Positives (FP)** | 209 | **354** | +145 |
| **Relaxed Match False Negatives (FN)** | 2,105 | **703** | **-1,402** |
| **Relaxed Precision (Resolvable)** | 0.8553 | **0.8816** | **+0.0263** |
| **Relaxed Recall (Resolvable)** | 0.3698 | **0.7895** | **+0.4197** |
| **Relaxed F1 (Resolvable Subset, $N=3,340$)** | **0.5163** | **0.8330** | **+0.3167** |
| **Relaxed F1 (Worst-Case Bound, $N=3,345$)** | **0.5158** | **0.8324** | **+0.3166** |
| **Boundary Mismatches (Overlap but != Exact)** | 603 | **1,021** (38.72% of relaxed TPs) | Character boundary variance |
| **Note-Level Micro Precision** | 0.9331 | **0.9516** | **+0.0185** |
| **Note-Level Micro Recall** | 0.4245 | **0.8337** | **+0.4092** |
| **Note-Level Micro F1** | **0.5835** | **0.8888** | **+0.3053** |
| **Note-Level Macro F1** | **0.5278** | **0.8684** | **+0.3406** |

### Accounting for the 5 Unresolved Test Annotations
Exactly 5 test annotations cannot be mapped to a single text offset without guessing:
- `ex_no 7556`: `"pain"`, `ex_no 7589`: `"home"`, `ex_no 7843`: `"wheelchair"`, `ex_no 8581`: `"pain"`, `ex_no 8654`: `"pain"`.
- In our conservative worst-case bounds, all 5 are penalized as False Negatives:
  $$\text{Recall}_{\text{worst-case}} = \frac{\text{TP}}{3,340 + 5} = \frac{2,637}{3,345} = 0.7883 \implies \text{F1} = 0.8324$$

---

## 9. Architecture & Implementation

```text
D:\Evernorth\Clinical-DSS\
├── app/                               # Interactive Streamlit Prototype
│   ├── app.py                         # Main multi-page application
│   ├── track_a_ui.py                  # Readmission risk estimation UI
│   └── track_b_ui.py                  # SBDH text extraction UI
├── models/                            # Frozen Model Artifacts
│   ├── baseline_a_logreg.joblib       # Primary bedside readmission pipeline
│   ├── baseline_b_logreg.joblib       # Placement sensitivity pipeline
│   ├── baseline_c_logreg.joblib       # Retrospective diagnostic pipeline
│   └── sbdh_transformer/             # Local pilot checkpoint (superseded)
├── evernorth_m3_results/              # Colab GPU Checkpoint & Outputs
│   └── models/sbdh_transformer/
│       └── best_model.pt              # Authoritative 3-epoch transformer weights
├── notebooks/                         # Self-contained Jupyter/Colab Workflows
│   ├── 01_data_audit.ipynb            # Milestone 1 data audit notebook
│   └── 02_sbdh_transformer.ipynb      # Milestone 3 Colab GPU training notebook
├── reports/                           # Machine-readable Metrics & Audits
│   ├── baseline_freeze_manifest.md    # Complete artifact manifest & hashes
│   ├── reproduction_guide.md          # Step-by-step reproduction commands
│   ├── final_evidence_consistency_audit.md # Audit verification report
│   ├── track_a_readmission_audit.md   # Track A methodological audit
│   ├── readmission_baselines_metrics.json # Track A metrics artifact
│   ├── readmission_calibration_metrics.json # Track A calibration deciles
│   └── sbdh_colab_transformer_metrics_corrected.json # Track B metrics artifact
├── scripts/                           # Reproducible CLI Runners
│   ├── audit_track_a_readmission.py   # Replicates Track A metrics
│   ├── evaluate_colab_checkpoint.py   # Evaluates Colab transformer on test
│   ├── validate_colab_notebook.py     # Colab notebook invariant validator
│   └── run_full_alignment_audit.py    # Split-wide annotation auditor
├── src/                               # Reusable Core Python Modules
│   ├── data_loaders.py                # Dataset loaders with None preservation
│   ├── audit_uci.py                   # UCI cohort filtering & grouped split
│   ├── train_readmission_baseline.py  # Readmission training pipeline
│   ├── span_alignment.py              # Auditable span alignment engine
│   ├── transformer_dataset.py         # Multi-label BIO token encoding/decoding
│   ├── transformer_model.py           # MultiLabelSBDHTransformer class
│   ├── sbdh_rules_baseline.py         # Regex baseline & bipartite matching
│   └── evaluate_sbdh_transformer.py   # Transformer evaluation engine
└── tests/                             # Automated Test Suite (32 tests)
```

---

## 10. Reproducibility & Testing

All results are verifiable using the existing frozen environment and automated tests:
1. **Full Test Suite (`pytest -v`):** 32 passed in ~27 seconds.
   - `test_readmission_evaluation.py` (9 tests): Mathematical consistency between confusion matrix cells and metrics, zero patient leakage, literal 'None' preservation, cohort exclusion counts.
   - `test_transformer_alignment.py` (8 tests): Multi-label identical span encoding (`ex_no 7062`), ambiguous span masking (`ex_no 1192`), token truncation detection, whitespace trimming, Colab checkpoint provenance and artifact consistency.
   - `test_span_alignment.py` (9 tests): Repeated mention context resolution (`ex_no 7411`, `7611`), accounting invariants.
   - `test_span_matching.py` (6 tests): Maximum bipartite matching order-invariance, 1-to-1 match constraints.
2. **Notebook Invariant Validator (`python scripts/validate_colab_notebook.py`):** 8/8 checks passed.
3. **Standalone Reproduction:**
   - Track A: `python scripts/audit_track_a_readmission.py`
   - Track B: `python scripts/evaluate_colab_checkpoint.py`

---

## 11. Limitations, Potential Biases & Ethical Considerations

1. **Temporal & Demographic Shift in Historical Data:**  
   The UCI diabetes cohort reflects inpatient care from **1999 to 2008**. Modern diabetes management guidelines, continuous glucose monitors (CGMs), outpatient care coordination, and newer medications (GLP-1 receptor agonists, SGLT2 inhibitors) are absent. Deploying this model in modern clinical practice without retraining on contemporary EHR data would lead to severe distribution shift.
2. **Regional Readmission Censoring:**  
   The label `NO` denotes absence of readmission within the 130 consortium hospitals. Patients readmitted to unparticipating regional hospitals, VA facilities, or out-of-state centers are recorded as negative, underestimating true readmission rates.
3. **Synthetic Narrative Artifacts in SBDH Text:**  
   Synth-SBDH notes are generated by large language models. While realistic, synthetic narratives often exhibit higher syntactic regularity and cleaner sentence structures than authentic EHR clinical documentation, which contains misspellings, fragmented shorthand, and dense copy-forward text.
4. **Disparity Risks in Algorithm Alerting:**  
   Care coordination resources triggered by readmission risk scores must be allocated equitably. Models with low precision (18% for Baseline A) generate false positives that could overwhelm care coordination teams if thresholds are improperly calibrated.

---

## 12. Future Performance-Improvement Experiments

The following experimental directions are documented as prospective research opportunities, preserving the frozen baseline intact:
1. **Clinical Composite Modeling (Multimodal Integration):**  
   Concatenate extracted SBDH risk indicators (from narrative discharge notes) with Baseline A structured features to evaluate whether social risk confers additive prognostic discrimination over clinical markers alone.
2. **Boundary Refinement & Post-Processing:**  
   Address the 1,021 boundary mismatches by implementing lexical boundary expansion (snapping subword token spans to word boundary delimiters like whitespace and punctuation).
3. **Clinical Domain Encoder Backbones:**  
   Evaluate domain-specific clinical encoders (e.g., `Clinical-Longformer` or `Bio_ClinicalBERT`) capable of processing full 4,096-token hospital notes without truncation.
4. **Isotonic Probability Recalibration:**  
   Apply isotonic regression on validation probabilities to improve probability calibration in higher risk deciles ($> 0.40$).

---

## 13. Conclusion

The Evernorth Clinical-DSS project successfully established two methodologically rigorous, fully audited machine learning baselines for clinical decision support research:
1. An auditable, leak-free 30-day readmission prediction baseline (**Baseline A: AUROC 0.6324, PR-AUC 0.1925, ECE 0.0061**) that eliminates lookahead coding bias and circular disposition leakage.
2. A true multi-label neural transformer for social determinants extraction (**Colab Transformer: Relaxed F1 0.8330, Exact F1 0.5105, Note Micro F1 0.8888**) that surpasses rule-based methods while enforcing bipartite matching order-invariance and dual-bound accounting.

Both modeling tracks are frozen, documented, and packaged with a local interactive Streamlit demonstration for research and educational exploration.

---

## 14. References & Artifact Inventory

### Primary References:
1. **UCI Diabetes 130-US Hospitals (Dataset ID 296):**  
   Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Balasubramanian, S., Higgins, K. A., & Cios, K. J. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Records*. BioMed Research International, 2014, 781670. DOI: `10.1155/2014/781670`.
2. **Synth-SBDH Benchmark:**  
   Soni, S., Roberts, K., & UMass BioNLP. (2024). *Synth-SBDH: A Synthetic Benchmark for Extracting Social and Behavioral Determinants of Health from Clinical Text*. Hugging Face Dataset: `bio-nlp-umass/Synth-SBDH`.
3. **DistilRoBERTa Encoder:**  
   Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*. arXiv preprint arXiv:1910.01108.
4. **Maximum Bipartite Matching:**  
   Kuhn, H. W. (1955). *The Hungarian Method for the Assignment Problem*. Naval Research Logistics Quarterly, 2(1‐2), 83-97.

### Artifact Inventory:
- **Baseline Freeze Manifest:** [`reports/baseline_freeze_manifest.md`](file:///d:/Evernorth/Clinical-DSS/reports/baseline_freeze_manifest.md)
- **Reproduction Guide:** [`reports/reproduction_guide.md`](file:///d:/Evernorth/Clinical-DSS/reports/reproduction_guide.md)
- **Track A Model Pipeline:** [`models/baseline_a_logreg.joblib`](file:///d:/Evernorth/Clinical-DSS/models/baseline_a_logreg.joblib) (SHA256: `794cff2d...`)
- **Track B Neural Weights:** [`evernorth_m3_results/models/sbdh_transformer/best_model.pt`](file:///d:/Evernorth/Clinical-DSS/evernorth_m3_results/models/sbdh_transformer/best_model.pt) (SHA256: `0cb00010...`)
- **Track A Metrics JSON:** [`reports/readmission_baselines_metrics.json`](file:///d:/Evernorth/Clinical-DSS/reports/readmission_baselines_metrics.json) (SHA256: `168a73c1...`)
- **Track B Metrics JSON:** [`reports/sbdh_colab_transformer_metrics_corrected.json`](file:///d:/Evernorth/Clinical-DSS/reports/sbdh_colab_transformer_metrics_corrected.json) (SHA256: `f20c59f4...`)
- **Portable Baseline Archive:** [`reports/baseline_freeze_archive_20261026.zip`](file:///d:/Evernorth/Clinical-DSS/reports/baseline_freeze_archive_20261026.zip) (SHA256: `34b5ab8b...`)
