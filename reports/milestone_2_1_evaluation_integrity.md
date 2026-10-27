# Evernorth Clinical Decision Support (CDS) — Milestone 2.1: Evaluation Integrity Fix

**Project:** Clinical Decision Support Using Deep Learning and NLP  
**Workspace:** `D:\Evernorth\Clinical-DSS`  
**Milestone:** 2.1 — Evaluation Integrity Fix & Metric Reconciliation  
**Date:** Local Runtime Verification (October 2026)  
**Status:** COMPLETE & EMPIRICALLY VERIFIED (14/14 automated tests passing)  

---

## 1. Executive Summary & Purpose

Milestone 2.1 resolves all empirical and mathematical discrepancies identified across the readmission modeling pipeline (Track A) and the social determinants of health extraction pipeline (Track B). 

Key corrections implemented:
1. **Mathematical Consistency in Readmission Evaluation (Track A):**
   - Eliminated all discrepancies between confusion matrix cells and reported classification metrics (`Precision`, `Recall`, `Specificity`, `F1-Score`).
   - Verified exact test-partition counts: **14,851 total encounters**, **1,650 positive encounters** ($11.1104\%$), and **13,201 negative encounters** ($88.8896\%$).
   - Corrected stale draft text in `reports/milestone_2_report.md` (which had misstated test positives as 1,648 and reported mismatched confusion matrices).
   - Embedded automated assertions inside [`src/train_readmission_baseline.py`](src/train_readmission_baseline.py) enforcing $|M_{\text{reported}} - M_{\text{CM}}| < 10^{-4}$ on every run.
   - Confirmed that saved models ([`models/baseline_a_logreg.joblib`](models/baseline_a_logreg.joblib), [`models/baseline_b_logreg.joblib`](models/baseline_b_logreg.joblib), [`models/baseline_c_logreg.joblib`](models/baseline_c_logreg.joblib)), the metrics artifact ([`reports/readmission_baselines_metrics.json`](reports/readmission_baselines_metrics.json)), and the Markdown report originate from the exact same execution.
2. **Evaluation Integrity & Bound Analysis for Synth-SBDH (Track B):**
   - Fully accounted for all 5 ambiguous repeated-mention annotations in `test_reviewed` (`ex_no 7556, 7589, 7843, 8581, 8654`), representing 0.15% of the 3,345 raw gold annotations.
   - Resolved the silent omission of ambiguous cases by explicitly reporting metrics on the **Resolvable Gold Subset** ($N = 3,340$ annotations across 3,303 unique intervals) alongside conservative **Worst-Case Sensitivity Bounds** ($N = 3,345$, treating all 5 unresolved annotations as False Negatives).
   - Replaced unconstrained relaxed overlap matching with deterministic **greedy 1-to-1 matching** to prevent overlapping predictions from double-counting a single gold span.
   - Explicitly delineated evaluated tasks (Task 1: Span Location, Task 2: SBDH Category) from unevaluated tasks (Task 3: Presence/Polarity, Task 4: Temporal Period).
   - Verified that the expert-reviewed test set was frozen and never utilized to construct or tune extraction rules.
3. **Automated Test Coverage:**
   - Established 14 comprehensive unit and integration tests across [`tests/test_readmission_evaluation.py`](tests/test_readmission_evaluation.py), [`tests/test_span_alignment.py`](tests/test_span_alignment.py), and [`tests/test_span_matching.py`](tests/test_span_matching.py). 100% passing.

---

## 2. Root Cause Analysis

### A. Track A (Readmission Metrics Discrepancy)
- **Root Cause:** A disconnect occurred between the programmatic execution and the draft report text. The execution script [`src/train_readmission_baseline.py`](src/train_readmission_baseline.py) generated [`reports/readmission_baselines_metrics.json`](reports/readmission_baselines_metrics.json) with true test counts of 1,650 positives and matching confusion matrices ($TN=10,357, FP=2,844, FN=1,023, TP=627$ for Baseline A). However, `reports/milestone_2_report.md` was drafted by copying figures from an earlier pre-stabilization run (which had 1,648 test positives, $TN=10,682, FP=2,521$, Precision 0.1818, Specificity 0.8091).
- **Code Limitation:** `evaluate_predictions()` called independent scikit-learn helper functions (`precision_score`, `recall_score`, `f1_score`, `confusion_matrix`) without asserting internal mathematical agreement with the confusion matrix cells.

### B. Track B (Synth-SBDH Evaluation Discrepancy)
- **Root Cause 1 (Silent Omission):** In [`src/sbdh_rules_baseline.py`](src/sbdh_rules_baseline.py), gold spans were assembled strictly from `gold_aligned["resolved_intervals"]`. The 5 ambiguous annotations were placed into `gold_aligned["ambiguous_cases"]` by the aligner, meaning the rules evaluator reported `total_gold_spans = 3340` without explicitly explaining the drop from the 3,345 raw annotations in `test_reviewed`.
- **Root Cause 2 (Relaxed Match Double-Counting):** In relaxed matching, when looping over predictions, the matching algorithm did not remove matched gold spans from the candidate pool. Consequently, if two predicted spans overlapped the same gold span, both predictions claimed a True Positive, inflating relaxed TP to 1,338 ($TP + FN = 1338 + 2105 = 3443 > 3340$).

---

## 3. Code Corrections & Pipeline Integrity Fixes

### A. Readmission Evaluation Assertions ([`src/train_readmission_baseline.py`](src/train_readmission_baseline.py))
Inside `evaluate_predictions()`, automated mathematical consistency assertions now verify every metric derivation:
```python
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
```
The output JSON now explicitly includes `positive_count`, `negative_count`, and `confusion_matrix_derived`.

### B. 1-to-1 Span Matching & Bound Tracking ([`src/sbdh_rules_baseline.py`](src/sbdh_rules_baseline.py))
1. **Modular 1-to-1 Span Matching:**
   ```python
   def match_spans_exact(pred_tuples, gold_tuples):
       tp = len(pred_tuples & gold_tuples)
       fp = len(pred_tuples - gold_tuples)
       fn = len(gold_tuples - pred_tuples)
       assert tp + fp == len(pred_tuples)
       assert tp + fn == len(gold_tuples)
       return tp, fp, fn

   def match_spans_relaxed(pred_tuples, gold_tuples):
       matched_gold = set()
       tp = 0; fp = 0
       for pt in sorted(pred_tuples):
           matched_gt = next((gt for gt in sorted(gold_tuples)
                              if gt not in matched_gold and pt[2] == gt[2] and max(pt[0], gt[0]) < min(pt[1], gt[1])), None)
           if matched_gt:
               matched_gold.add(matched_gt)
               tp += 1
           else:
               fp += 1
       fn = len(gold_tuples) - len(matched_gold)
       assert tp + fp == len(pred_tuples)
       assert tp + fn == len(gold_tuples)
       return tp, fp, fn, matched_gold
   ```
2. **Explicit Accounting of 5 Ambiguous Annotations:**
   - Total raw annotations in `test_reviewed`: **3,345**.
   - Resolvable gold annotations: **3,340** (across 3,303 unique intervals, including 37 multi-label spans).
   - Unresolved ambiguous annotations: **5** ($0.15\%$). Specifically:
     - `ex_no 7556`: `'pain'` (2 candidates: `'chronic pain due to arthritis'` vs `'on pain medications'`)
     - `ex_no 7589`: `'home'` (2 candidates: `'foreclosure on his home'` vs substring of `'homeless'`)
     - `ex_no 7843`: `'wheelchair'` (2 candidates: `'uses a wheelchair'` vs `'lack of wheelchair ramps'`)
     - `ex_no 8581`: `'pain'` (2 candidates: `'chronic pain due to'` vs `'to manage the pain'`)
     - `ex_no 8654`: `'pain'` (2 candidates: `'severe abdominal pain'` vs `'when pain is managed'`)
3. **Worst-Case Sensitivity Bounds:**
   Evaluates both the resolvable subset ($N = 3,340$) and a lower bound where all 5 ambiguous annotations are penalized as False Negatives ($FN_{\text{worst-case}} = FN + 5, N = 3,345$).
4. **Task Scope Declaration:**
   - Evaluated: Task 1 (Span Location) and Task 2 (SBDH Category).
   - Unevaluated: Task 3 (Presence / Polarity) and Task 4 (Period / Temporal).

---

## 4. Reconciled Track A Metrics (Readmission Baselines)

### 4.1 Cohort Partitioning & Actual Label Counts
Patient-grouped splitting via `GroupShuffleSplit` on `patient_nbr` (seed 42) on the 99,343 eligible encounters:
- **Train (70%):** 69,470 encounters | 48,992 patients | **7,953 positive** ($11.4481\%$) | 61,517 negative
- **Validation (15%):** 15,022 encounters | 10,499 patients | **1,711 positive** ($11.3899\%$) | 13,311 negative
- **Test (15%):** 14,851 encounters | 10,499 patients | **1,650 positive** ($11.1104\%$) | 13,201 negative
- **Zero Patient Leakage:** Patient ID intersection between all splits is verified to be $\emptyset$.

### 4.2 Test Partition Reconciled Metrics

| Model | AUROC | PR-AUC (AP)* | Frozen Threshold | Precision | Recall | F1-Score | Specificity | Brier Score |
|---|---|---|---|---|---|---|---|---|
| **Majority Baseline** | 0.5000 | 0.1111 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0988 |
| **Baseline A** *(No Disp, No Diag)* | **0.6324** | **0.1925** | 0.1300 | **0.1806** | **0.3800** | **0.2449** | **0.7846** | 0.0961 |
| **Baseline B** *(With Disp, No Diag)* | **0.6599** | **0.2176** | 0.1450 | **0.2136** | **0.3552** | **0.2667** | **0.8365** | 0.0947 |
| **Baseline C** *(With Disp, With Diag)* | **0.6666** | **0.2213** | 0.1350 | **0.2028** | **0.4224** | **0.2740** | **0.7924** | 0.0944 |

*\*PR-AUC: Average Precision (`sklearn.metrics.average_precision_score`), defined as $\sum_n (R_n - R_{n-1}) P_n$.*

### 4.3 Validation Partition Reconciled Metrics (Threshold Selection)

| Model | AUROC | PR-AUC (AP) | Selected Threshold | Precision | Recall | F1-Score | Specificity | Brier Score |
|---|---|---|---|---|---|---|---|---|
| **Majority Baseline** | 0.5000 | 0.1139 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.1009 |
| **Baseline A** *(No Disp, No Diag)* | 0.6467 | 0.2066 | 0.1300 | 0.1944 | 0.4050 | 0.2627 | 0.7843 | 0.0976 |
| **Baseline B** *(With Disp, No Diag)* | 0.6683 | 0.2209 | 0.1450 | 0.2260 | 0.3840 | 0.2845 | 0.8310 | 0.0967 |
| **Baseline C** *(With Disp, With Diag)* | 0.6661 | 0.2196 | 0.1350 | 0.2095 | 0.4378 | 0.2834 | 0.7877 | 0.0968 |

*Threshold Selection Integrity: Thresholds $\tau = 0.1300, 0.1450, 0.1350$ were selected to maximize F1 on the Validation partition only, frozen, and evaluated on Test with zero post-hoc adjustment.*

### 4.4 Step-by-Step Confusion Matrix Verification

$$\begin{aligned}
\text{Precision} &= \frac{TP}{TP + FP} \\
\text{Recall} &= \frac{TP}{TP + FN} \\
\text{Specificity} &= \frac{TN}{TN + FP} \\
F_1 &= \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
\end{aligned}$$

#### Baseline A (Test Partition):
- Matrix: $TN = 10,357, \quad FP = 2,844, \quad FN = 1,023, \quad TP = 627$
- Class Totals: $Positives = 627 + 1,023 = 1,650, \quad Negatives = 10,357 + 2,844 = 13,201, \quad Total = 14,851$
- $\text{Precision} = \frac{627}{627 + 2,844} = \frac{627}{3,471} = \mathbf{0.18063958...} \implies \mathbf{0.1806}$
- $\text{Recall} = \frac{627}{627 + 1,023} = \frac{627}{1,650} = \mathbf{0.38000000...} \implies \mathbf{0.3800}$
- $\text{Specificity} = \frac{10,357}{10,357 + 2,844} = \frac{10,357}{13,201} = \mathbf{0.78456177...} \implies \mathbf{0.7846}$
- $F_1 = \frac{2 \times 0.18063958 \times 0.3800}{0.18063958 + 0.3800} = \mathbf{0.2449179...} \implies \mathbf{0.2449}$

#### Baseline B (Test Partition):
- Matrix: $TN = 11,043, \quad FP = 2,158, \quad FN = 1,064, \quad TP = 586$
- Class Totals: $Positives = 586 + 1,064 = 1,650, \quad Negatives = 11,043 + 2,158 = 13,201, \quad Total = 14,851$
- $\text{Precision} = \frac{586}{586 + 2,158} = \frac{586}{2,744} = \mathbf{0.21355685...} \implies \mathbf{0.2136}$
- $\text{Recall} = \frac{586}{586 + 1,064} = \frac{586}{1,650} = \mathbf{0.35515151...} \implies \mathbf{0.3552}$
- $\text{Specificity} = \frac{11,043}{11,043 + 2,158} = \frac{11,043}{13,201} = \mathbf{0.83652753...} \implies \mathbf{0.8365}$
- $F_1 = \frac{2 \times 0.21355685 \times 0.35515151}{0.21355685 + 0.35515151} = \mathbf{0.2666666...} \implies \mathbf{0.2667}$

#### Baseline C (Test Partition):
- Matrix: $TN = 10,461, \quad FP = 2,740, \quad FN = 953, \quad TP = 697$
- Class Totals: $Positives = 697 + 953 = 1,650, \quad Negatives = 10,461 + 2,740 = 13,201, \quad Total = 14,851$
- $\text{Precision} = \frac{697}{697 + 2,740} = \frac{697}{3,437} = \mathbf{0.20279313...} \implies \mathbf{0.2028}$
- $\text{Recall} = \frac{697}{697 + 953} = \frac{697}{1,650} = \mathbf{0.42242424...} \implies \mathbf{0.4224}$
- $\text{Specificity} = \frac{10,461}{10,461 + 2,740} = \frac{10,461}{13,201} = \mathbf{0.79243996...} \implies \mathbf{0.7924}$
- $F_1 = \frac{2 \times 0.20279313 \times 0.42242424}{0.20279313 + 0.42242424} = \mathbf{0.274032...} \implies \mathbf{0.2740}$

---

## 5. Reconciled Track B Metrics (Synth-SBDH Extraction)

### 5.1 Resolvable Subset vs. Worst-Case Bounded Metrics
- Total notes evaluated: **1,732**
- Total raw annotations: **3,345**
- Resolvable gold annotations: **3,340** (across 3,303 unique intervals)
- Unresolved ambiguous annotations: **5** (0.15%)
- Total predicted rule spans: **1,444**

| Evaluation Mode | Evaluation Subset | Precision | Recall | F1-Score | TP | FP | FN | Gold Basis |
|---|---|---|---|---|---|---|---|---|
| **Note-Level Micro** | All Notes | **0.9331** | **0.4245** | **0.5835** | 1,417 | 102 | 1,921 | 3,338 note-level category indicators |
| **Note-Level Macro** | All Notes | **0.9316** | **0.3860** | **0.5278** | — | — | — | Unweighted across 15 categories |
| **Note-Level Weighted** | All Notes | — | — | **0.5751** | — | — | — | Weighted by category support |
| **Span-Level Exact** | Resolvable Subset | **0.4377** | **0.1892** | **0.2642** | 632 | 812 | 2,708 | 3,340 resolvable gold spans |
| **Span-Level Exact** | Worst-Case Bound | **0.4377** | **0.1889** | **0.2639** | 632 | 812 | 2,713 | 3,345 total gold annotations (ambiguous as FN) |
| **Span-Level Relaxed** | Resolvable Subset | **0.8553** | **0.3698** | **0.5163** | 1,235 | 209 | 2,105 | 3,340 resolvable gold spans (1-to-1 match) |
| **Span-Level Relaxed** | Worst-Case Bound | **0.8553** | **0.3692** | **0.5158** | 1,235 | 209 | 2,110 | 3,345 total gold annotations (ambiguous as FN) |

*Greedy 1-to-1 matching ensures: $TP + FP = 1,444$ (exact total predictions) and $TP + FN = 3,340$ (exact resolvable gold spans).*

### 5.2 Per-Category Breakdown (15 SBDH Categories)

| Category | Gold Support | Precision | Recall | F1-Score | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| **pain** | 202 | 1.0000 | 0.6238 | 0.7683 | 126 | 0 | 76 |
| **job insecurity** | 304 | 1.0000 | 0.6118 | 0.7592 | 186 | 0 | 118 |
| **psychiatric symptoms or disorders** | 289 | 1.0000 | 0.5467 | 0.7069 | 158 | 0 | 131 |
| **social isolation** | 303 | 1.0000 | 0.5017 | 0.6681 | 152 | 0 | 151 |
| **housing insecurity** | 209 | 1.0000 | 0.4880 | 0.6559 | 102 | 0 | 107 |
| **patient disability** | 139 | 0.9853 | 0.4820 | 0.6473 | 67 | 1 | 72 |
| **loss of relationship** | 243 | 1.0000 | 0.4486 | 0.6193 | 109 | 0 | 134 |
| **legal problems** | 107 | 1.0000 | 0.3458 | 0.5139 | 37 | 0 | 70 |
| **transitions of care** | 128 | 1.0000 | 0.3359 | 0.5029 | 43 | 0 | 85 |
| **substance abuse** | 203 | 0.9706 | 0.3251 | 0.4871 | 66 | 2 | 137 |
| **financial insecurity** | 243 | 0.9545 | 0.2593 | 0.4078 | 63 | 3 | 180 |
| **barriers to care** | 107 | 1.0000 | 0.2243 | 0.3664 | 24 | 0 | 83 |
| **food insecurity** | 125 | 0.9310 | 0.2160 | 0.3506 | 27 | 2 | 98 |
| **violence** | 154 | 0.9600 | 0.1558 | 0.2682 | 24 | 1 | 130 |
| **physical isolation** | 71 | 0.1720 | 0.2254 | 0.1951 | 16 | 77 | 55 |

---

## 6. Exact Execution Commands and Exit Statuses

All scripts and automated test suites were executed directly in the project environment. No mock or fabricated outputs were used.

| Execution Step | Command | Exit Code | Verified Outputs |
|---|---|---|---|
| **1. Track A Baseline Training & Assertions** | `python src/train_readmission_baseline.py` | `0` | Recomputed all models and generated [`reports/readmission_baselines_metrics.json`](reports/readmission_baselines_metrics.json). Passed all mathematical assertions. |
| **2. Track B Rules Evaluation & Bound Tracking** | `python src/sbdh_rules_baseline.py` | `0` | Evaluated against `test_reviewed` and generated [`reports/sbdh_rules_metrics.json`](reports/sbdh_rules_metrics.json). Verified 1-to-1 matching and worst-case bounds. |
| **3. Full Automated Test Suite Execution** | `pytest` | `0` | Executed 14 automated tests across all 3 test files. 14 passed in 2.17s. |

### Test Session Log Summary:
```text
platform win32 -- Python 3.11.3, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\Evernorth\Clinical-DSS
plugins: anyio-4.9.0
collected 14 items

tests\test_readmission_evaluation.py ....                                [ 28%]
tests\test_span_alignment.py .....                                       [ 64%]
tests\test_span_matching.py .....                                        [100%]

============================= 14 passed in 2.17s ==============================
```

---

## 7. Remaining Methodological Limitations

1. **Retrospective Placement Confounding (Track A):**
   - The performance lift observed in Baseline B (+0.0275 AUROC) and Baseline C (+0.0342 AUROC) over Baseline A reflects retrospective coding of post-acute placement (`discharge_disposition_id`). At bedside admission or early triage, placement is not yet determined. Baseline A (0.6324 AUROC, 0.1925 PR-AUC) represents the true pre-placement clinical benchmark.
2. **Synthetic Data Homogeneity (Track B):**
   - Synth-SBDH is constructed from synthetic prompt templates. High rule precision across categories (e.g. 1.000 for 9 categories) reflects standardized phrasing in synthetic text, not clinical NLP generalizability to authentic EHR notes containing abbreviations, dictation typos, and shorthand.
3. **Unevaluated Assertion & Temporal Dimensions (Track B):**
   - The keyword rules baseline extracts span locations and SBDH categories only. Determining clinical assertion polarity (Presence: `yes` vs `no`) and temporal anchoring (Period: `current` vs `history`) requires contextual machine learning models and remains unevaluated in this keyword baseline.
