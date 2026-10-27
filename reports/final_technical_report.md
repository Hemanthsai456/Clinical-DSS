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
| **Source / Provenance** | 130 participating non-profit hospitals in the US; hosted by UCI Machine Learning Repository (Dataset ID 296; Strack et al., 2014) | BioNLP Lab, University of Massachusetts Amherst (Hugging Face: `bio-nlp-umass/Synth-SBDH`; Mitra et al., 2025) |
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
│   ├── run_full_alignment_audit.py    # Split-wide annotation auditor
│   ├── run_span_boundary_ablation.py  # Validation boundary ablation runner (Exp 2.2)
│   ├── evaluate_candidate_test_set.py # Candidate Arm 6A test set evaluator
│   └── verify_candidate_test_metrics.py # Direct mathematical JSON metric auditor
├── src/                               # Reusable Core Python Modules
│   ├── data_loaders.py                # Dataset loaders with None preservation
│   ├── audit_uci.py                   # UCI cohort filtering & grouped split
│   ├── train_readmission_baseline.py  # Readmission training pipeline
│   ├── span_alignment.py              # Auditable span alignment engine
│   ├── transformer_dataset.py         # Multi-label BIO token encoding/decoding (Production)
│   ├── transformer_model.py           # MultiLabelSBDHTransformer class
│   ├── sbdh_rules_baseline.py         # Regex baseline & bipartite matching
│   ├── evaluate_sbdh_transformer.py   # Transformer evaluation engine
│   ├── experimental_decoders.py       # Candidate active-threshold decoders (Experimental)
│   └── experimental_boundary_snapping.py # Candidate snapping & fusion engine (Experimental)
└── tests/                             # Automated Test Suite (61 tests)
```

---

## 10. Reproducibility & Testing

All results are verifiable using the existing frozen environment and automated tests:
1. **Full Test Suite (`pytest -v`):** **61 passed** in ~76 seconds.
   - `test_readmission_evaluation.py` (9 tests): Mathematical consistency between confusion matrix cells and metrics, zero patient leakage, literal 'None' preservation, cohort exclusion counts.
   - `test_transformer_alignment.py` (8 tests): Multi-label identical span encoding (`ex_no 7062`), ambiguous span masking (`ex_no 1192`), token truncation detection, whitespace trimming, Colab checkpoint provenance and artifact consistency.
   - `test_span_alignment.py` (9 tests): Repeated mention context resolution (`ex_no 7411`, `7611`), accounting invariants.
   - `test_span_matching.py` (6 tests): Maximum bipartite matching order-invariance, 1-to-1 match constraints.
   - `test_boundary_ablation.py` (16 tests): Candidate regression tests covering balanced delimiter protection (e.g. `(PTSD)`), orphan bracket stripping, mid-word boundary repair, punctuation trimming, dangling prepositions, hyphenated terms, distinct adjacent gold mention isolation under $\text{gap} \le 2$, and sentence boundary blocking.
   - `test_streamlit_app.py` (13 tests): Production Streamlit app verification, preset testing, schema isolation, and artifact hash integrity.
2. **Notebook Invariant Validator (`python scripts/validate_colab_notebook.py`):** 8/8 checks passed.
3. **Standalone Reproduction:**
   - Track A Readmission: `python scripts/audit_track_a_readmission.py`
   - Track B Colab Transformer: `python scripts/evaluate_colab_checkpoint.py`
   - Arm 6A Candidate Evaluation: `python scripts/evaluate_candidate_test_set.py`
   - Candidate Metrics Verification: `python scripts/verify_candidate_test_metrics.py`

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

## 12. Experimental Post-Processing Optimization: Arm 6A Candidate Evaluation

> [!NOTE]
> **EXPERIMENTAL DESIGNATION & PRODUCTION BASELINE INVARIANCE:**  
> The post-processing optimization described in this section is designated strictly as an **experimental candidate (Arm 6A)**, **NOT** the production model.  
> 
> - The production MVP (`app/app.py`, `src/transformer_dataset.py`, `models/baseline_a_logreg.joblib`) and the authoritative baseline metrics artifact ([`reports/sbdh_colab_transformer_metrics_corrected.json`](file:///d:/Evernorth/Clinical-DSS/reports/sbdh_colab_transformer_metrics_corrected.json)) remain **100% untouched and unchanged**.
> - Model checkpoint weights (`evernorth_m3_results/models/sbdh_transformer/best_model.pt`) remain bit-identical before and after evaluation (`SHA-256: 0cb00010631b089746fdd97ff3f9019e1bcc29012a7c093b33501a9c30186ba4`).
> - Zero retraining was performed. All gains stem exclusively from post-processing inference logic on frozen model logits.

### 12.1 Candidate Architecture & Validation Selection (Arm 6A)

In validation ablation experiments across 876 validation notes ([`reports/experiments/span_boundary_ablation/span_boundary_ablation_report.md`](file:///d:/Evernorth/Clinical-DSS/reports/experiments/span_boundary_ablation/span_boundary_ablation_report.md)), the original decoder's Outside-dominance rule was found to prematurely terminate or suppress SBDH spans when the background Outside logit exceeded SBDH category logits, even if SBDH logits were elevated ($\ge 0.30$). Furthermore, subword tokenization frequently clipped leading/trailing letters or fragmented phrases.

To address these failure modes without retraining, the **Arm 6A Candidate Pipeline** was constructed and validated:
1. **Strategy B Active-Threshold Decoder ($\theta = 0.30$):** Decouples Begin and Inside activations, allowing SBDH token spans whenever category logits exceed 0.30 regardless of Outside dominance.
2. **Lexical Boundary Snapping with Balanced Delimiter Protection:** Expands mid-word character slices to full whitespace word boundaries, strips sentence-ending punctuation (`.`, `;`, `:`), preserves balanced parenthetical acronyms (e.g., `post-traumatic stress disorder (PTSD)`), strips orphan unanchored delimiters, and trims dangling functional prepositions (`"to"`, `"for"`, `"with"`).
3. **Conservative Same-Category Span Fusion ($\text{gap} \le 2$):** Fuses subword fragments separated by $\le 2$ characters within the exact same SBDH category and sentence boundary. Crucially, fusion across conjunctions (`" and "`, `" or "`) was **rejected** because it damaged 27 coordinate clinical gold mentions in the validation set.

### 12.2 Single Final Test Set Evaluation Results

The candidate pipeline was evaluated **once** on the reviewed test set (`synth_sbdh_test_reviewed.csv`, $N = 1,732$ notes, 3,345 raw annotations, 3,340 resolvable annotations, 3,303 unique intervals). All 10 mathematical metrics were audited and verified directly from the saved JSON artifact ([`reports/experiments/candidate_test_evaluation/candidate_test_evaluation_metrics.json`](file:///d:/Evernorth/Clinical-DSS/reports/experiments/candidate_test_evaluation/candidate_test_evaluation_metrics.json)).

#### Side-by-Side Test Benchmark Comparison

| Metric | Frozen Baseline (Control) | Arm 6A Candidate (Experimental) | Absolute Difference | Relative Change |
| :--- | :---: | :---: | :---: | :---: |
| **Total Predicted Spans** | 2,991 | **3,225** | +234 | +7.82% |
| **Notes with $\ge 1$ Prediction** | 1,713 | **1,726** | +13 | +0.76% |
| **Boundary Snaps Applied** | 0 | **91** | +91 | — |
| **Span Merges Applied** | 0 | **16** | +16 | — |
| **Invalid Spans ($s \ge e$ or OOB)** | 0 | **0** | 0 | Clean |
| **Overlapping Intra-Category Spans** | 0 | **0** | 0 | Clean |
| **Exact True Positives (TP)** | 1,616 | **1,698** | **+82** | **+5.07%** |
| **Exact False Positives (FP)** | 1,375 | 1,527 | +152 | +11.05% |
| **Exact False Negatives (FN, Resolvable)** | 1,724 | **1,642** | **-82** | **-4.76%** |
| **Exact Precision** | **0.5403** | 0.5265 | -0.0138 | -2.55% |
| **Exact Recall (Resolvable, $N=3,340$)** | 0.4838 | **0.5084** | **+0.0246** | **+5.08%** |
| **Exact Recall (Worst-Case, $N=3,345$)** | 0.4831 | **0.5076** | **+0.0245** | **+5.07%** |
| **Exact F1-Score (Resolvable)** | 0.5105 | **0.5173** | **+0.0068** | **+1.33%** |
| **Exact F1-Score (Worst-Case Bound)** | 0.5101 | **0.5169** | **+0.0068** | **+1.33%** |
| **Relaxed True Positives (TP)** | 2,637 | **2,762** | **+125** | **+4.74%** |
| **Relaxed False Positives (FP)** | 354 | 463 | +109 | +30.79% |
| **Relaxed False Negatives (FN, Resolvable)** | 703 | **578** | **-125** | **-17.78%** |
| **Relaxed Precision** | **0.8816** | 0.8564 | -0.0252 | -2.86% |
| **Relaxed Recall (Resolvable, $N=3,340$)** | 0.7895 | **0.8269** | **+0.0374** | **+4.74%** |
| **Relaxed Recall (Worst-Case, $N=3,345$)** | 0.7883 | **0.8257** | **+0.0374** | **+4.74%** |
| **Relaxed F1-Score (Resolvable)** | 0.8330 | **0.8414** | **+0.0084** | **+1.01%** |
| **Relaxed F1-Score (Worst-Case Bound)** | 0.8324 | **0.8408** | **+0.0084** | **+1.01%** |
| **Note-Level Micro Precision** | **0.9516** | 0.9272 | -0.0244 | -2.56% |
| **Note-Level Micro Recall** | 0.8337 | **0.8649** | **+0.0312** | **+3.74%** |
| **Note-Level Micro F1-Score** | 0.8888 | **0.8949** | **+0.0061** | **+0.69%** |
| **Note-Level Macro F1-Score** | 0.8684 | **0.8773** | **+0.0089** | **+1.02%** |
| **Boundary Errors (Relaxed TP - Exact TP)** | 1,021 | 1,064 | +43 | +4.21% |
| **Boundary Error Rate (% of Relaxed TPs)** | 38.72% | **38.52%** | **-0.20%** | **-0.52%** |

### 12.3 Per-Category Breakdown (Reviewed Test Set)

To maintain absolute methodological clarity, per-category evaluation metrics are partitioned into three distinct, unambiguous side-by-side comparisons using identical evaluation procedures for both the frozen baseline and the Arm 6A candidate.

#### Table 12.3A: Note-Level Category Classification (Baseline vs Arm 6A Candidate)
*Procedure: Binary multi-label classification per note-category pair (1 = category present in note, 0 = absent). Evaluated across all 1,732 reviewed test notes (2,827 total note-category occurrences).*

| SBDH Category | Note Support | Base TP | Cand TP | TP Delta | Base P | Cand P | Base R | Cand R | Base F1 | Cand F1 | F1 Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **barriers to care** | 107 | 77 | 82 | +5 | 0.8556 | 0.8039 | 0.7196 | 0.7664 | 0.7817 | 0.7847 | +0.0030 |
| **financial insecurity** | 243 | 186 | 201 | +15 | 0.9347 | 0.9013 | 0.7654 | 0.8272 | 0.8416 | 0.8627 | +0.0211 |
| **food insecurity** | 125 | 121 | 123 | +2 | 0.9758 | 0.9609 | 0.9680 | 0.9840 | 0.9719 | 0.9723 | +0.0004 |
| **housing insecurity** | 209 | 149 | 159 | +10 | 0.9868 | 0.9578 | 0.7129 | 0.7608 | 0.8278 | 0.8480 | +0.0202 |
| **job insecurity** | 304 | 281 | 283 | +2 | 0.9723 | 0.9725 | 0.9243 | 0.9309 | 0.9477 | 0.9513 | +0.0036 |
| **legal problems** | 107 | 94 | 98 | +4 | 0.9792 | 0.9800 | 0.8785 | 0.9159 | 0.9261 | 0.9469 | +0.0208 |
| **loss of relationship** | 243 | 184 | 188 | +4 | 1.0000 | 1.0000 | 0.7572 | 0.7737 | 0.8618 | 0.8724 | +0.0106 |
| **pain** | 202 | 198 | 200 | +2 | 1.0000 | 1.0000 | 0.9802 | 0.9901 | 0.9900 | 0.9950 | +0.0050 |
| **patient disability** | 139 | 128 | 135 | +7 | 0.8312 | 0.7941 | 0.9209 | 0.9712 | 0.8737 | 0.8738 | +0.0001 |
| **physical isolation** | 71 | 26 | 32 | +6 | 0.7879 | 0.6667 | 0.3662 | 0.4507 | 0.5000 | 0.5378 | +0.0378 |
| **psychiatric symptoms or disorders** | 289 | 232 | 242 | +10 | 0.9098 | 0.8643 | 0.8028 | 0.8374 | 0.8529 | 0.8506 | -0.0023 |
| **social isolation** | 303 | 240 | 250 | +10 | 0.9877 | 0.9843 | 0.7921 | 0.8251 | 0.8791 | 0.8977 | +0.0186 |
| **substance abuse** | 203 | 200 | 201 | +1 | 0.9709 | 0.9526 | 0.9852 | 0.9901 | 0.9780 | 0.9710 | -0.0070 |
| **transitions of care** | 128 | 110 | 114 | +4 | 0.9091 | 0.8571 | 0.8594 | 0.8906 | 0.8835 | 0.8736 | -0.0099 |
| **violence** | 154 | 131 | 137 | +6 | 0.9776 | 0.9580 | 0.8506 | 0.8896 | 0.9097 | 0.9226 | +0.0129 |
| **NOTE MICRO-SUM** | **2,827** | **2,357** | **2,445** | **+88** | **0.9516** | **0.9272** | **0.8337** | **0.8649** | **0.8888** | **0.8949** | **+0.0061** |
| **NOTE MACRO-AVG** | — | — | — | — | — | — | — | — | **0.8684** | **0.8773** | **+0.0089** |

#### Table 12.3B: Span-Level Exact Matching (Baseline vs Arm 6A Candidate)
*Procedure: Exact character offset and category agreement: `(start_char, end_char, category)`. Evaluated across all 3,340 resolvable gold spans.*

| SBDH Category | Gold Support | Base Pred | Cand Pred | Base TP | Cand TP | TP Delta | Base P | Cand P | Base R | Cand R | Base F1 | Cand F1 | F1 Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **barriers to care** | 118 | 113 | 121 | 42 | 45 | +3 | 0.3717 | 0.3719 | 0.3559 | 0.3814 | 0.3636 | 0.3766 | +0.0130 |
| **financial insecurity** | 267 | 220 | 247 | 143 | 153 | +10 | 0.6500 | 0.6194 | 0.5356 | 0.5730 | 0.5873 | 0.5953 | +0.0080 |
| **food insecurity** | 141 | 159 | 159 | 71 | 77 | +6 | 0.4465 | 0.4843 | 0.5035 | 0.5461 | 0.4733 | 0.5133 | +0.0400 |
| **housing insecurity** | 274 | 195 | 234 | 100 | 116 | +16 | 0.5128 | 0.4957 | 0.3650 | 0.4234 | 0.4264 | 0.4567 | +0.0303 |
| **job insecurity** | 346 | 328 | 341 | 205 | 221 | +16 | 0.6250 | 0.6481 | 0.5925 | 0.6387 | 0.6083 | 0.6434 | +0.0351 |
| **legal problems** | 127 | 117 | 117 | 43 | 48 | +5 | 0.3675 | 0.4103 | 0.3386 | 0.3780 | 0.3525 | 0.3934 | +0.0409 |
| **loss of relationship** | 255 | 194 | 196 | 119 | 117 | -2 | 0.6134 | 0.5969 | 0.4667 | 0.4588 | 0.5301 | 0.5188 | -0.0113 |
| **pain** | 226 | 210 | 214 | 141 | 137 | -4 | 0.6714 | 0.6402 | 0.6239 | 0.6062 | 0.6468 | 0.6227 | -0.0241 |
| **patient disability** | 160 | 199 | 231 | 80 | 82 | +2 | 0.4020 | 0.3550 | 0.5000 | 0.5125 | 0.4457 | 0.4194 | -0.0263 |
| **physical isolation** | 74 | 33 | 50 | 15 | 17 | +2 | 0.4545 | 0.3400 | 0.2027 | 0.2297 | 0.2804 | 0.2742 | -0.0062 |
| **psychiatric symptoms or disorders** | 338 | 303 | 336 | 140 | 146 | +6 | 0.4620 | 0.4345 | 0.4142 | 0.4320 | 0.4368 | 0.4332 | -0.0036 |
| **social isolation** | 403 | 325 | 355 | 199 | 215 | +16 | 0.6123 | 0.6056 | 0.4938 | 0.5335 | 0.5467 | 0.5673 | +0.0206 |
| **substance abuse** | 270 | 291 | 289 | 166 | 166 | +0 | 0.5704 | 0.5744 | 0.6148 | 0.6148 | 0.5918 | 0.5939 | +0.0021 |
| **transitions of care** | 144 | 139 | 153 | 75 | 75 | +0 | 0.5396 | 0.4902 | 0.5208 | 0.5208 | 0.5300 | 0.5051 | -0.0249 |
| **violence** | 197 | 165 | 182 | 77 | 83 | +6 | 0.4667 | 0.4560 | 0.3909 | 0.4213 | 0.4254 | 0.4380 | +0.0126 |
| **EXACT MICRO-SUM** | **3,340** | **2,991** | **3,225** | **1616** | **1698** | **+82** | **0.5403** | **0.5265** | **0.4838** | **0.5084** | **0.5105** | **0.5173** | **+0.0068** |
| **EXACT MACRO-AVG** | — | — | — | — | — | — | — | — | — | — | **0.4830** | **0.4901** | **+0.0071** |

#### Table 12.3C: Span-Level Relaxed Matching (Baseline vs Arm 6A Candidate)
*Procedure: Kuhn's Maximum Bipartite 1-to-1 Matching under category agreement & character overlap $\ge 1$. Evaluated across all 3,340 resolvable gold spans.*

| SBDH Category | Gold Support | Base Pred | Cand Pred | Base TP | Cand TP | TP Delta | Base P | Cand P | Base R | Cand R | Base F1 | Cand F1 | F1 Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **barriers to care** | 118 | 113 | 121 | 78 | 85 | +7 | 0.6903 | 0.7025 | 0.6610 | 0.7203 | 0.6753 | 0.7113 | +0.0360 |
| **financial insecurity** | 267 | 220 | 247 | 195 | 210 | +15 | 0.8864 | 0.8502 | 0.7303 | 0.7865 | 0.8008 | 0.8171 | +0.0163 |
| **food insecurity** | 141 | 159 | 159 | 133 | 136 | +3 | 0.8365 | 0.8553 | 0.9433 | 0.9645 | 0.8867 | 0.9067 | +0.0200 |
| **housing insecurity** | 274 | 195 | 234 | 175 | 195 | +20 | 0.8974 | 0.8333 | 0.6387 | 0.7117 | 0.7463 | 0.7677 | +0.0214 |
| **job insecurity** | 346 | 328 | 341 | 309 | 315 | +6 | 0.9421 | 0.9238 | 0.8931 | 0.9104 | 0.9169 | 0.9170 | +0.0001 |
| **legal problems** | 127 | 117 | 117 | 106 | 110 | +4 | 0.9060 | 0.9402 | 0.8346 | 0.8661 | 0.8689 | 0.9016 | +0.0327 |
| **loss of relationship** | 255 | 194 | 196 | 188 | 193 | +5 | 0.9691 | 0.9847 | 0.7373 | 0.7569 | 0.8374 | 0.8559 | +0.0185 |
| **pain** | 226 | 210 | 214 | 203 | 206 | +3 | 0.9667 | 0.9626 | 0.8982 | 0.9115 | 0.9312 | 0.9364 | +0.0052 |
| **patient disability** | 160 | 199 | 231 | 142 | 149 | +7 | 0.7136 | 0.6450 | 0.8875 | 0.9313 | 0.7911 | 0.7621 | -0.0290 |
| **physical isolation** | 74 | 33 | 50 | 26 | 31 | +5 | 0.7879 | 0.6200 | 0.3514 | 0.4189 | 0.4860 | 0.5000 | +0.0140 |
| **psychiatric symptoms or disorders** | 338 | 303 | 336 | 255 | 269 | +14 | 0.8416 | 0.8006 | 0.7544 | 0.7959 | 0.7956 | 0.7982 | +0.0026 |
| **social isolation** | 403 | 325 | 355 | 299 | 322 | +23 | 0.9200 | 0.9070 | 0.7419 | 0.7990 | 0.8214 | 0.8496 | +0.0282 |
| **substance abuse** | 270 | 291 | 289 | 255 | 256 | +1 | 0.8763 | 0.8858 | 0.9444 | 0.9481 | 0.9091 | 0.9159 | +0.0068 |
| **transitions of care** | 144 | 139 | 153 | 123 | 128 | +5 | 0.8849 | 0.8366 | 0.8542 | 0.8889 | 0.8693 | 0.8620 | -0.0073 |
| **violence** | 197 | 165 | 182 | 150 | 157 | +7 | 0.9091 | 0.8626 | 0.7614 | 0.7970 | 0.8287 | 0.8285 | -0.0002 |
| **RELAXED MICRO-SUM** | **3,340** | **2,991** | **3,225** | **2637** | **2762** | **+125** | **0.8816** | **0.8564** | **0.7895** | **0.8269** | **0.8330** | **0.8414** | **+0.0084** |
| **RELAXED MACRO-AVG** | — | — | — | — | — | — | — | — | — | — | **0.8110** | **0.8220** | **+0.0110** |

### 12.4 Precision–Recall Trade-Off Analysis

A fundamental trade-off governs the transition from the control decoder to Strategy B:
- **Recall Expansion:** By removing the Outside-dominance constraint, the candidate model extracts valid clinical mentions where the background token logit hovers between 0.40 and 0.55 while the SBDH category logit is strong ($\ge 0.30$). This captures **+82 additional exact true positives** and **+125 relaxed true positives**, boosting exact recall by +2.46% (0.4838 $\to$ 0.5084) and relaxed recall by +3.74% (0.7895 $\to$ 0.8269).
- **Precision Cost:** Enabling active SBDH signals below the Outside threshold introduces +234 additional predictions (2,991 $\to$ 3,225), which increases exact false positives from 1,375 to 1,527 (+152) and relaxed false positives from 354 to 463 (+109). Consequently, precision drops by 1.38% on exact matching (0.5403 $\to$ 0.5265) and 2.52% on relaxed matching (0.8816 $\to$ 0.8564).
- **Net F1 Gain:** Because the relative gain in recall (+5.08% exact, +4.74% relaxed) substantially exceeds the precision reduction (-2.55% exact, -2.86% relaxed), the harmonic mean (F1-score) improves consistently across both evaluation regimes:
  - Exact F1: **0.5105 $\to$ 0.5173** (+0.0068)
  - Relaxed F1: **0.8330 $\to$ 0.8414** (+0.0084)
  - Note Micro F1: **0.8888 $\to$ 0.8949** (+0.0061)

### 12.5 Unresolved Benchmark Annotation Caveats

The evaluation strictly enforces dual-bound accounting to prevent bias from ambiguous benchmark annotations:
- **Resolvable Subset ($N = 3,340$):** Covers all annotations where text offsets are mapped unambiguously.
- **Worst-Case Lower Bound ($N = 3,345$):** Directly incorporates the **5 unresolved ambiguous annotations** identified during the audit:
  1. `ex_no 7556`: `"pain"` (category: `pain`, 2 text occurrences, 1 annotation).
  2. `ex_no 7589`: `"home"` (category: `housing insecurity`, 2 text occurrences, 1 annotation).
  3. `ex_no 7843`: `"wheelchair"` (category: `patient disability`, 2 text occurrences, 1 annotation).
  4. `ex_no 8581`: `"pain"` (category: `pain`, 2 text occurrences, 1 annotation).
  5. `ex_no 8654`: `"pain"` (category: `pain`, 2 text occurrences, 1 annotation).
- **Conservative Penalization:** Because these 5 annotations lack contextual anchors in the gold benchmark, resolving them to arbitrary occurrences would inflate performance. In the worst-case bound, all 5 cases are treated as False Negatives, establishing a rock-solid lower bound:
  - Exact Worst-Case F1: **0.5169** (vs 0.5101 baseline)
  - Relaxed Worst-Case F1: **0.8408** (vs 0.8324 baseline)

---

## 13. Future Performance-Improvement Experiments

The following experimental directions are documented as prospective research opportunities, preserving the frozen baseline intact:
1. **Production Promotion of Arm 6A Post-Processing:**  
   Pending clinical advisory sign-off, evaluate transitioning the production decoder in `app/app.py` from Strategy A to Arm 6A, leveraging the 16 regression unit tests in `tests/test_boundary_ablation.py`.
2. **Clinical Composite Modeling (Multimodal Integration):**  
   Concatenate extracted SBDH risk indicators (from narrative discharge notes) with Baseline A structured features to evaluate whether social risk confers additive prognostic discrimination over clinical markers alone.
3. **Clinical Domain Encoder Backbones:**  
   Evaluate domain-specific clinical encoders (e.g., `Clinical-Longformer` or `Bio_ClinicalBERT`) capable of processing full 4,096-token hospital notes without truncation.
4. **Isotonic Probability Recalibration:**  
   Apply isotonic regression on validation probabilities to improve probability calibration in higher risk deciles ($> 0.40$).

---

## 14. Conclusion

The Evernorth Clinical-DSS project successfully established two methodologically rigorous, fully audited machine learning baselines for clinical decision support research:
1. An auditable, leak-free 30-day readmission prediction baseline (**Baseline A: AUROC 0.6324, PR-AUC 0.1925, ECE 0.0061**) that eliminates lookahead coding bias and circular disposition leakage.
2. A true multi-label neural transformer for social determinants extraction (**Colab Transformer: Relaxed F1 0.8330, Exact F1 0.5105, Note Micro F1 0.8888**) that surpasses rule-based methods while enforcing bipartite matching order-invariance and dual-bound accounting.
3. An empirically validated post-processing candidate (**Arm 6A: Relaxed F1 0.8414, Exact F1 0.5173**) that demonstrates statistically positive recall and F1 improvements (+82 exact TPs, +125 relaxed TPs) without model retraining.

Both modeling tracks are frozen, documented, and packaged with a local interactive Streamlit demonstration for research and educational exploration.

---

## 15. References & Artifact Inventory

### Primary References:
1. **UCI Diabetes 130-US Hospitals (Dataset ID 296):**  
   Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Balasubramanian, S., Higgins, K. A., & Cios, K. J. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Records*. BioMed Research International, 2014, 781670. DOI: `10.1155/2014/781670`.
2. **Synth-SBDH Benchmark:**  
   Mitra, A., Yang, Z., Druhl, E., Goodwin, R., & Yu, H. (2025). *Synth-SBDH: A Synthetic Dataset of Social and Behavioral Determinants of Health for Clinical Text*. arXiv:2406.06056 (EMNLP 2025). Dataset: `huggingface.co/datasets/bio-nlp-umass/Synth-SBDH`.
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
- **Track B Frozen Baseline Metrics JSON:** [`reports/sbdh_colab_transformer_metrics_corrected.json`](file:///d:/Evernorth/Clinical-DSS/reports/sbdh_colab_transformer_metrics_corrected.json) (SHA256: `f20c59f4...`)
- **Track B Arm 6A Candidate Metrics JSON:** [`reports/experiments/candidate_test_evaluation/candidate_test_evaluation_metrics.json`](file:///d:/Evernorth/Clinical-DSS/reports/experiments/candidate_test_evaluation/candidate_test_evaluation_metrics.json)
- **Track B Arm 6A Candidate Technical Report:** [`reports/experiments/candidate_test_evaluation/candidate_test_evaluation_report.md`](file:///d:/Evernorth/Clinical-DSS/reports/experiments/candidate_test_evaluation/candidate_test_evaluation_report.md)
- **Portable Baseline Archive:** [`reports/baseline_freeze_archive_20261026.zip`](file:///d:/Evernorth/Clinical-DSS/reports/baseline_freeze_archive_20261026.zip) (SHA256: `34b5ab8b...`)

