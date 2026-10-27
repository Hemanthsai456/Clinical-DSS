# Evernorth Clinical Decision Support (CDS) — Milestone 2: Span Validation and First Baselines

**Research Prototype:** Clinical Decision Support Using Deep Learning and NLP  
**Milestone:** 2.0 — Span Validation and First Baselines (Track A & Track B)  
**Date:** Local Runtime Verification (October 2026)  
**Status:** ALL PHASES COMPLETED & VERIFIED  

---

## 1. Executive Summary

Milestone 2 establishes the primary empirical baselines for both clinical CDS tasks:
1. **Phase 1 (Span Alignment Gate):** Resolved multi-label identical span preservation, contextual disambiguation of repeated mentions (`ex_no 7411`, `7062`, `7611`), and automated testing. All 6 remaining ambiguous cases across the 10,499 notes were explicitly documented and exported to an audit artifact.
2. **Phase 2 (Track A — 30-Day Readmission Baselines):** Established a majority-class baseline and three logistic regression models under patient-grouped splitting with zero patient leakage, evaluating retrospective feature sensitivity across **Baseline A** (no disposition ID, no discharge diagnoses), **Baseline B** (adds disposition ID), and **Baseline C** (adds disposition ID + grouped ICD-9 diagnoses).
3. **Phase 3 (Track B — Rules-Based SBDH Baseline):** Implemented a transparent, frozen domain-knowledge rules baseline supporting multi-label span extraction. Evaluated against `synth_sbdh_test_reviewed.csv` (1,732 notes) with note-level and span-level exact and relaxed matching.

Zero mock values were used. All statistics below represent verified execution on local datasets.

---

## 2. Phase 1 — SBDH Span-Alignment Gate Resolution

### 2.1 Named Test Case Adjudication & Automated Unit Tests

Automated unit tests were established in [`tests/test_span_alignment.py`](../tests/test_span_alignment.py) (5 passed in 0.82s):

| Test Case | Scenario / Text | Target Challenge | Resolution & Execution Verification |
|---|---|---|---|
| **`ex_no 7411`** | `"Patient recently moved from a nursing home back to her own home."` | Annotated span `'home'` appears twice in text. Blind string matching selects `'nursing home'` (facility). | The aligner does NOT assume the first substring is correct. Contextual audit verifies that the second occurrence (`'her own home'`, char `59..63`) represents the non-insecurity anchor (`housing insecurity == no`), avoiding the nursing home facility (`38..42`). (**PASSED**) |
| **`ex_no 7062`** | `"Patient suffers from chronic pain and cannot afford to continue physical therapy."` | Phrase `'cannot afford to continue physical therapy'` is annotated with both `financial insecurity` AND `barriers to care`. | Both categories are preserved on the exact same character interval `[38, 80]`. Downstream token conversion retains both labels across all tokens with zero truncation. (**PASSED**) |
| **`ex_no 7611`** | `"Patient experiencing abdominal pain and has a documented history of opioid usage for pain management."` | Annotated with two `'pain'` mentions: one `current`, one `history`. Occurrence order must not be arbitrarily assumed. | Contextual symptom matching aligns `current` pain to `'abdominal pain'` (`31..35`) and `history` pain to `'pain management'` (`85..89`) based on local clinical modifiers without guessing. (**PASSED**) |
| **Casing & Ungrounded** | Case-only variations (e.g. `PTSD` vs `ptsd`) and missing spans. | Ensure casing differences recover offsets while truly absent spans are flagged. | 100% of casing differences recovered; missing spans flagged as `unmatched`. (**PASSED**) |
| **Ambiguity Flagging** | Notes with multiple plausible alignments without conclusive evidence. | Flag for human review rather than silently picking an arbitrary occurrence. | Correctly raises `AMBIGUOUS_REPEATED_SPAN` and outputs all candidate character offsets. (**PASSED**) |

### 2.2 Corpus-Wide Alignment Audit Across All Splits

The full audit script [`src/run_span_audit.py`](../src/run_span_audit.py) audited all 10,499 notes:

| Split | Total Notes | Extracted Annotations | Resolved Unique Intervals | Multi-Label Identical Intervals | Casing-Only Normalization | Contextually Adjudicated | Ambiguous Unresolved Spans | Unmatched / Missing Spans |
|---|---|---|---|---|---|---|---|---|
| **`train`** | 6,136 | 10,022 | 9,988 | 33 | 0 | 0 | **1** | 0 |
| **`val`** | 876 | 1,443 | 1,437 | 6 | 0 | 0 | **0** | 0 |
| **`test`** | 1,755 | 2,877 | 2,866 | 11 | 0 | 0 | **0** | 0 |
| **`test_reviewed`** | 1,732 | 3,345 | 3,303 | 37 | 92 | 3 | **5** | 0 |

### 2.3 Unresolved Cases Audit Artifact

In accordance with gate criteria, rather than silently guessing, every remaining unresolved ambiguous case was explicitly exported to [`reports/sbdh_alignment/unresolved_alignment_cases.json`](../reports/sbdh_alignment/unresolved_alignment_cases.json) (exactly 6 cases across the entire corpus):
1. `train` (`ex_no 1192`): Span `'pain'` (2 candidates in `"Patient's pain is severe and has been prescribed opioids for pain management"`).
2. `test_reviewed` (`ex_no 7556`): Span `'pain'` (2 candidates in text, one nested inside `'chronic pain due to arthritis'`).
3. `test_reviewed` (`ex_no 7589`): Span `'home'` (2 candidates in text, one nested as substring of `'homeless'`).
4. `test_reviewed` (`ex_no 7843`): Span `'wheelchair'` (2 candidates in text: `'uses a wheelchair'` vs `'lack of wheelchair ramps'`).
5. `test_reviewed` (`ex_no 8581`): Span `'pain'` (2 candidates in text, one nested inside `'chronic pain'`).
6. `test_reviewed` (`ex_no 8654`): Span `'pain'` (2 candidates in text: `'severe abdominal pain'` vs `'when pain is managed'`).

Neither `Reasoning` nor `Operation` metadata are used as model inputs or targets; they are restricted exclusively to audit provenance.

---

## 3. Phase 2 — Track A: Readmission Prediction Baselines

### 3.1 Cohort & Patient-Grouped Partitioning
- **Dataset:** UCI Diabetes 130-US Hospitals (1999–2008), eligible modeling cohort excluding in-hospital mortality and hospice dispositions (`11, 13, 14, 19, 20, 21`).
- **Eligible Encounters:** 99,343 encounters (69,990 unique patients).
- **Patient-Grouped Split:** `GroupShuffleSplit` on `patient_nbr` (seed 42):
  - **Train (70%):** 69,470 encounters | 48,992 patients | Positive rate: **11.45%** (7,953 positive)
  - **Val (15%):** 15,022 encounters | 10,499 patients | Positive rate: **11.39%** (1,711 positive)
  - **Test (15%):** 14,851 encounters | 10,499 patients | Positive rate: **11.11%** (1,650 positive)
  - **Zero Patient Overlap Asserted Across All Partitions**.
- **Preprocessing:** Imputers, scalers, and one-hot encoders fitted strictly on the **Train** partition.
- **Threshold Selection:** Optimal decision threshold tuned to maximize F1 score on the **Validation** set, frozen, and evaluated directly on **Test**. Final test set results were never used to tune features, thresholds, or hyperparameters.

### 3.2 Models & Feature Sensitivity Definitions
1. **Majority-Class Baseline:** Predicts constant non-readmission (`0`) with probability equal to empirical training prevalence (11.45%).
2. **Baseline A (Clinical Baseline):** Standard clinical features excluding post-acute placement (`discharge_disposition_id`) and discharge diagnoses (`diag_1, diag_2, diag_3`).
3. **Baseline B (Placement Model):** Adds `discharge_disposition_id` to quantify marginal lift from post-acute placement decisions.
4. **Baseline C (Full Retrospective Model):** Adds `discharge_disposition_id` + grouped ICD-9 discharge diagnoses (using Strack et al. 2014 clinical disease categories: Circulatory, Respiratory, Digestive, Diabetes, Genitourinary, Neoplasms, Musculoskeletal, Injury/Poisoning, Other).

*Note: These are retrospective feature sensitivity experiments to quantify potential circular placement and coding lookahead bias, not proof that all features would be available at operational discharge triage.*

### 3.3 Test Partition Results

| Model | AUROC | PR-AUC (AP)* | Frozen Threshold | Precision | Recall | F1-Score | Specificity | Brier Score |
|---|---|---|---|---|---|---|---|---|
| **Majority Class Baseline** | 0.5000 | 0.1111 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0988 |
| **Baseline A** *(No Disposition, No Diag)* | **0.6324** | **0.1925** | 0.1300 | **0.1806** | **0.3800** | **0.2449** | **0.7846** | 0.0961 |
| **Baseline B** *(With Disposition, No Diag)* | **0.6599** | **0.2176** | 0.1450 | **0.2136** | **0.3552** | **0.2667** | **0.8365** | 0.0947 |
| **Baseline C** *(With Disposition, With Diag)* | **0.6666** | **0.2213** | 0.1350 | **0.2028** | **0.4224** | **0.2740** | **0.7924** | 0.0944 |

*\*PR-AUC Definition: Average Precision (`sklearn.metrics.average_precision_score`), calculated as the weighted mean of precisions achieved at each recall threshold: $\text{AP} = \sum_n (R_n - R_{n-1}) P_n$.*

### 3.4 Validation Partition Results (Threshold Selection)

| Model | AUROC | PR-AUC (AP) | Selected Threshold | Precision | Recall | F1-Score | Specificity | Brier Score |
|---|---|---|---|---|---|---|---|---|
| **Majority Class Baseline** | 0.5000 | 0.1139 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.1009 |
| **Baseline A** *(No Disposition, No Diag)* | 0.6467 | 0.2066 | 0.1300 | **0.1944** | **0.4050** | **0.2627** | **0.7843** | 0.0976 |
| **Baseline B** *(With Disposition, No Diag)* | 0.6683 | 0.2209 | 0.1450 | **0.2260** | **0.3840** | **0.2845** | **0.8310** | 0.0967 |
| **Baseline C** *(With Disposition, With Diag)* | 0.6661 | 0.2196 | 0.1350 | **0.2095** | **0.4378** | **0.2834** | **0.7877** | 0.0968 |

### 3.5 Test Confusion Matrices & Mathematical Verification

All classification metrics agree mathematically with the confusion matrix cells within $10^{-4}$ tolerance:
$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad \text{Specificity} = \frac{TN}{TN + FP}, \quad F_1 = \frac{2 \cdot P \cdot R}{P + R}$$

- **Majority Class Baseline:**  
  $\begin{pmatrix} 13201 & 0 \\ 1650 & 0 \end{pmatrix}$ (TN=13,201, FP=0, FN=1,650, TP=0; Total=14,851)
- **Baseline A:**  
  $\begin{pmatrix} 10357 & 2844 \\ 1023 & 627 \end{pmatrix}$ (TN=10,357, FP=2,844, FN=1,023, TP=627; Positives=1,650, Negatives=13,201)
  - Precision: $627 / (627 + 2844) = 627 / 3471 = \mathbf{0.1806}$
  - Recall: $627 / (627 + 1023) = 627 / 1650 = \mathbf{0.3800}$
  - Specificity: $10357 / (10357 + 2844) = 10357 / 13201 = \mathbf{0.7846}$
  - F1-Score: $2 \times 0.180639 \times 0.3800 / (0.180639 + 0.3800) = \mathbf{0.2449}$
- **Baseline B:**  
  $\begin{pmatrix} 11043 & 2158 \\ 1064 & 586 \end{pmatrix}$ (TN=11,043, FP=2,158, FN=1,064, TP=586; Positives=1,650, Negatives=13,201)
  - Precision: $586 / (586 + 2158) = 586 / 2744 = \mathbf{0.2136}$
  - Recall: $586 / (586 + 1064) = 586 / 1650 = \mathbf{0.3552}$
  - Specificity: $11043 / (11043 + 2158) = 11043 / 13201 = \mathbf{0.8365}$
  - F1-Score: $\mathbf{0.2667}$
- **Baseline C:**  
  $\begin{pmatrix} 10461 & 2740 \\ 953 & 697 \end{pmatrix}$ (TN=10,461, FP=2,740, FN=953, TP=697; Positives=1,650, Negatives=13,201)
  - Precision: $697 / (697 + 2740) = 697 / 3437 = \mathbf{0.2028}$
  - Recall: $697 / (697 + 953) = 697 / 1650 = \mathbf{0.4224}$
  - Specificity: $10461 / (10461 + 2740) = 10461 / 13201 = \mathbf{0.7924}$
  - F1-Score: $\mathbf{0.2740}$

### 3.6 Sensitivity Analysis Findings
1. **Disposition Impact:** Incorporating `discharge_disposition_id` improves AUROC from **0.6324 to 0.6599** (+0.0275 lift) and PR-AUC from **0.1925 to 0.2176** (+0.0251 lift). This demonstrates that post-acute placement (e.g. SNF, Home Health, Rehab) carries significant predictive correlation, but risks circular placement confounding if used at bedside before placement is settled.
2. **Diagnosis Impact:** Adding discharge ICD-9 diagnosis codes yields a minor additional lift of **+0.0067 AUROC** (0.6599 to 0.6666) and **+0.0037 PR-AUC**. This confirms that in-hospital medication titrations, procedure intensity, and prior utilization provide the primary clinical signal, while post-discharge billed diagnoses contribute only marginal incremental lift.
3. **Saved Models:** Exported to [`models/baseline_a_logreg.joblib`](../models/baseline_a_logreg.joblib), [`models/baseline_b_logreg.joblib`](../models/baseline_b_logreg.joblib), [`models/baseline_c_logreg.joblib`](../models/baseline_c_logreg.joblib).
4. **Machine-Readable Metrics:** Exported to [`reports/readmission_baselines_metrics.json`](../reports/readmission_baselines_metrics.json).

---

## 4. Phase 3 — Track B: Rules-Based SBDH Baseline

### 4.1 Specification & Evaluation Setup
- **Baseline Design:** Deterministic regular expression and keyword extraction rules covering all 15 SBDH categories, supporting multi-label assignment on single spans (e.g. `'cannot afford to continue physical therapy'` -> `financial insecurity` + `barriers to care`).
- **Rule Independence:** Rules were derived strictly from domain clinical taxonomies (PRAPARE, AHC-HRSN, ICD-10 Z-codes) and training patterns, and **frozen prior to evaluation**. The expert-reviewed test set (`test_reviewed`, 1,732 notes) was **never** used to develop, optimize, or tune extraction rules.
- **Evaluation Dataset:** `synth_sbdh_test_reviewed.csv` (1,732 notes, 3,345 raw gold annotations).
- **Ambiguous Spans Handling:** All 5 ambiguous repeated-mention annotations (`ex_no 7556, 7589, 7843, 8581, 8654`) are explicitly accounted for (0.15% of total annotations). Evaluation reports both the **Resolvable Subset** ($N = 3,340$ annotations across 3,303 unique intervals) and a **Conservative Worst-Case Sensitivity Bound** treating all 5 unresolved annotations as False Negatives ($FN_{\text{worst-case}} = FN + 5$). Zero annotations are silently dropped or arbitrarily guessed.
- **Task Scope Boundaries:**
  - **Evaluated Tasks:** Task 1 (Span Location: character offsets, exact and relaxed matching) and Task 2 (SBDH Category: 15 social determinants).
  - **Unevaluated Tasks:** Task 3 (Presence: yes/no polarity) and Task 4 (Period: current/history). These assertion and temporal tasks are unevaluated in this keyword baseline and reserved for downstream transformer models.

### 4.2 Aggregate Classification & Span-Level Extraction Metrics

| Metric Level | Evaluation Mode | Precision | Recall | F1-Score | Details / True Positives |
|---|---|---|---|---|---|
| **Note-Level** | **Micro-Average** | **0.9331** | **0.4245** | **0.5835** | Aggregates all category indicators across notes. |
| **Note-Level** | **Macro-Average** | **0.9316** | **0.3860** | **0.5278** | Unweighted average across all 15 SBDH categories. |
| **Note-Level** | **Weighted-Average** | — | — | **0.5751** | Weighted by category support in reviewed test. |
| **Span-Level** | **Exact Boundary (Resolvable)** | **0.4377** | **0.1892** | **0.2642** | TP=632, FP=812, FN=2,708 ($N=3,340$). Exact character start/end + category match. |
| **Span-Level** | **Exact Boundary (Worst-Case)** | **0.4377** | **0.1889** | **0.2639** | TP=632, FP=812, FN=2,713 ($N=3,345$). Bounds unresolved ambiguous cases as FN. |
| **Span-Level** | **Relaxed / Overlap (Resolvable)** | **0.8553** | **0.3698** | **0.5163** | TP=1,235, FP=209, FN=2,105 ($N=3,340$). 1-to-1 greedy overlap match + category match. |
| **Span-Level** | **Relaxed / Overlap (Worst-Case)** | **0.8553** | **0.3692** | **0.5158** | TP=1,235, FP=209, FN=2,110 ($N=3,345$). Bounds unresolved ambiguous cases as FN. |

*Span matching enforcement: Enforces greedy 1-to-1 matching to prevent redundant overlapping predictions from double-counting a single gold span as multiple true positives.*

### 4.3 Per-Category Breakdown (All 15 SBDH Categories)

| Category | Support | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **pain** | 202 | **1.0000** | **0.6238** | **0.7683** |
| **job insecurity** | 304 | **1.0000** | **0.6118** | **0.7592** |
| **psychiatric symptoms or disorders** | 289 | **1.0000** | **0.5467** | **0.7069** |
| **social isolation** | 303 | **1.0000** | **0.5017** | **0.6681** |
| **housing insecurity** | 209 | **1.0000** | **0.4880** | **0.6559** |
| **patient disability** | 139 | **0.9853** | **0.4820** | **0.6473** |
| **loss of relationship** | 243 | **1.0000** | **0.4486** | **0.6193** |
| **legal problems** | 107 | **1.0000** | **0.3458** | **0.5139** |
| **transitions of care** | 128 | **1.0000** | **0.3359** | **0.5029** |
| **substance abuse** | 203 | **0.9706** | **0.3251** | **0.4871** |
| **financial insecurity** | 243 | **0.9545** | **0.2593** | **0.4078** |
| **barriers to care** | 107 | **1.0000** | **0.2243** | **0.3664** |
| **food insecurity** | 125 | **0.9310** | **0.2160** | **0.3506** |
| **violence** | 154 | **0.9600** | **0.1558** | **0.2682** |
| **physical isolation** | 71 | **0.1720** | **0.2254** | **0.1951** |

### 4.4 Regulatory & Methodological Limitations
- **Rules Baseline Limitation:** This rules-based baseline is a deterministic heuristic keyword/regex system, **not** a machine learning or deep-learning model.
- **Synthetic Data Disclaimer:** Synth-SBDH notes are synthetically generated via LLM prompt templates and exhibit uniform grammatical phrasing. High synthetic-data benchmark performance does **not** establish real-world clinical validity or generalizability to genuine, noisy electronic health records containing dictation typos, acronyms, and fragmented shorthand.
- **Machine-Readable Metrics:** Exported to [`reports/sbdh_rules_metrics.json`](../reports/sbdh_rules_metrics.json).

---

## 5. Artifacts and Verification Index

| Artifact | Location | Description |
|---|---|---|
| **Span Alignment Engine** | [`src/span_alignment.py`](../src/span_alignment.py) | Module implementing multi-label identical span recovery, contextual adjudication, and ambiguity review flagging. |
| **Span Unit Tests** | [`tests/test_span_alignment.py`](../tests/test_span_alignment.py) | Automated test suite for ex_no 7411, 7062, 7611, multi-label tokens, and casing differences (100% passed). |
| **Readmission Evaluation Tests** | [`tests/test_readmission_evaluation.py`](../tests/test_readmission_evaluation.py) | Automated mathematical consistency assertions between confusion matrices and reported metrics (100% passed). |
| **Span Matching & Bounds Tests** | [`tests/test_span_matching.py`](../tests/test_span_matching.py) | Hand-constructed unit tests for exact/relaxed 1-to-1 span matching, multi-label retention, and worst-case bounds (100% passed). |
| **Unresolved Cases Export** | [`reports/sbdh_alignment/unresolved_alignment_cases.json`](../reports/sbdh_alignment/unresolved_alignment_cases.json) | Exhaustive JSON export of all 6 ambiguous repeated mentions across the corpus. |
| **Span Alignment Audit Report** | [`reports/sbdh_alignment/alignment_audit_report.json`](../reports/sbdh_alignment/alignment_audit_report.json) | Machine-readable breakdown of resolved intervals across all splits. |
| **Readmission Baseline Script** | [`src/train_readmission_baseline.py`](../src/train_readmission_baseline.py) | Script training Majority Baseline and Baselines A, B, and C with frozen validation thresholds and mathematical assertions. |
| **Fitted Model Artifacts** | [`models/baseline_a_logreg.joblib`](../models/baseline_a_logreg.joblib) <br> [`models/baseline_b_logreg.joblib`](../models/baseline_b_logreg.joblib) <br> [`models/baseline_c_logreg.joblib`](../models/baseline_c_logreg.joblib) | Serialized scikit-learn preprocessing and logistic regression pipelines. |
| **Readmission Metrics JSON** | [`reports/readmission_baselines_metrics.json`](../reports/readmission_baselines_metrics.json) | Complete validation and test metrics with mathematical consistency verification. |
| **SBDH Rules Baseline Script** | [`src/sbdh_rules_baseline.py`](../src/sbdh_rules_baseline.py) | Script running frozen multi-label rule extraction, 1-to-1 span matching, and worst-case bounds. |
| **SBDH Rules Metrics JSON** | [`reports/sbdh_rules_metrics.json`](../reports/sbdh_rules_metrics.json) | Machine-readable note-level and span-level exact/relaxed metrics and ambiguous case audit. |
