# Evernorth Clinical-DSS — Reproduction Guide

**Project:** Clinical Decision Support Using Deep Learning and NLP  
**Workspace:** `D:\Evernorth\Clinical-DSS`  
**Date:** October 2026  
**Git Commit:** `7dd92b5bda82042f5009af52ab7013223df567b4`  
**Status:** Tested & Empirically Verified

---

## 1. Overview & Reproduction Principles

This guide details the exact, tested command lines required to reproduce every verified metric, test suite, and audit artifact in the Evernorth Clinical-DSS project.

### Core Distinctions
- **Evaluation of Existing Frozen Checkpoints:** Rapid local evaluation of saved `.joblib` pipelines and the Colab `.pt` transformer checkpoint using CPU inference.
- **Training New Models from Scratch:** Local execution of the training scripts (`src/train_readmission_baseline.py` for Track A; `src/train_sbdh_transformer.py` for Track B).
- **Recreating the Colab GPU Environment:** Executing `notebooks/02_sbdh_transformer.ipynb` on an NVIDIA T4 GPU in Google Colab to reproduce the 3-epoch deep learning training run.

---

## 2. Environment Setup

### 2.1 Python Environment & Dependencies
The project requires **Python 3.10+** (developed and tested on Python 3.11.3 Windows 64-bit).

```powershell
# 1. Create a virtual environment
python -m venv venv

# 2. Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# (Alternative for Linux/macOS bash)
# source venv/bin/activate

# 3. Upgrade pip and install repository dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Install testing and linting tools
pip install pytest pytest-cov
```

### 2.2 Verify Installed Package Versions
```powershell
python -c "import pandas, numpy, sklearn, torch, transformers; print(f'pandas {pandas.__version__}, numpy {numpy.__version__}, sklearn {sklearn.__version__}, torch {torch.__version__}, transformers {transformers.__version__}')"
```
*Expected Output:*
```text
pandas 2.2.x, numpy 1.26.x, sklearn 1.4.x, torch 2.x, transformers 4.x / 5.x
```

---

## 3. Automated Test Suite Execution

All tests are implemented in `tests/` and run without network access using local fixture data and verified datasets.

### 3.1 Run Full Test Suite
```powershell
pytest -v
```
*Expected Result:* **66 passed in ~20 seconds.**

### 3.2 Targeted Test Execution
```powershell
# 1. Track A readmission evaluation integrity (CM derivations, zero leakage, label mapping)
pytest -v tests/test_readmission_evaluation.py

# 2. Track B transformer multi-label encoding, loss masking, and Colab provenance
pytest -v tests/test_transformer_alignment.py

# 3. Span alignment, repeated mention disambiguation (ex_no 7411, 7611), and accounting invariants
pytest -v tests/test_span_alignment.py

# 4. Maximum bipartite 1-to-1 matching and adversarial order-invariance
pytest -v tests/test_span_matching.py

# 5. Candidate boundary snapping, balanced delimiter preservation (PTSD), and fusion regression tests
pytest -v tests/test_boundary_ablation.py

# 6. Streamlit interactive prototype and preset regression tests
pytest -v tests/test_streamlit_app.py
```

---

## 4. Track A: UCI Readmission Reproduction

### 4.1 Reproduce Test Metrics from Saved Models (No Retraining)
This command loads the frozen `.joblib` pipelines, evaluates them against the exact 14,851 test encounters, verifies $<10^{-4}$ agreement with reported metrics, and exports probability calibration curves:

```powershell
python scripts/audit_track_a_readmission.py
```

- **Models Evaluated:**
  - `models/baseline_a_logreg.joblib` (Primary Bedside Model)
  - `models/baseline_b_logreg.joblib` (Augmented Placement Benchmark)
  - `models/baseline_c_logreg.joblib` (Retrospective Billed Benchmark)
  - `Majority_Class_Baseline` (Statistical baseline)
- **Output Artifacts Generated:**
  - [`reports/readmission_calibration_metrics.json`](file:///d:/Evernorth/Clinical-DSS/reports/readmission_calibration_metrics.json)
- **Console Output:**
  ```text
  [MATCH CONFIRMED] Model Baseline_A_No_Disposition_No_Diag reproduced with exact numerical agreement!
  Calibration - Test Brier: 0.0961, Test ECE: 0.0061
  [MATCH CONFIRMED] Model Baseline_B_With_Disposition_No_Diag reproduced with exact numerical agreement!
  Calibration - Test Brier: 0.0947, Test ECE: 0.0091
  [MATCH CONFIRMED] Model Baseline_C_With_Disposition_With_Diag reproduced with exact numerical agreement!
  Calibration - Test Brier: 0.0944, Test ECE: 0.0068
  ```

### 4.2 Retrain Baseline Models from Scratch (Optional Verification)
To re-fit all three logistic regression models on `X_train` ($N = 69,470$), recalibrate thresholds on `X_val` ($N = 15,022$), and overwrite the baseline artifacts:

```powershell
python src/train_readmission_baseline.py
```
- **Execution Time:** ~45 seconds on standard CPU.
- **Outputs Overwritten:**
  - `models/baseline_a_logreg.joblib`
  - `models/baseline_b_logreg.joblib`
  - `models/baseline_c_logreg.joblib`
  - `reports/readmission_baselines_metrics.json`

---

## 5. Track B: SBDH Extraction Reproduction

### 5.1 Reproduce Rules Baseline Evaluation
Evaluates deterministic regular expressions on `synth_sbdh_test_reviewed.csv` ($N = 1,732$) using Kuhn's maximum bipartite matching algorithm:

```powershell
python src/sbdh_rules_baseline.py --output_file reports/sbdh_rules_metrics_corrected.json
```
- **Execution Time:** Instantaneous (< 2 seconds).
- **Target Metrics:**
  - Exact F1: **0.2642** (Resolvable) / **0.2639** (Worst-case)
  - Relaxed F1: **0.5163** (Resolvable) / **0.5158** (Worst-case)
  - Note Micro F1: **0.5835**

### 5.2 Reproduce Colab 3-Epoch Transformer Evaluation
Loads the 3-epoch trained weights (`best_model.pt`) from `evernorth_m3_results/models/sbdh_transformer/` and evaluates on `synth_sbdh_test_reviewed.csv` using CPU inference and the corrected alignment engine:

```powershell
python scripts/evaluate_colab_checkpoint.py
```
- **Execution Time:** ~5 to 7 minutes on multi-core CPU.
- **Output Artifact:**
  - [`reports/sbdh_colab_transformer_metrics_corrected.json`](file:///d:/Evernorth/Clinical-DSS/reports/sbdh_colab_transformer_metrics_corrected.json)
- **Target Metrics:**
  - Exact F1: **0.5105** (Resolvable) / **0.5101** (Worst-case bound)
  - Relaxed F1: **0.8330** (Resolvable) / **0.8324** (Worst-case bound)
  - Note Micro F1: **0.8888**

### 5.3 Audit Full Dataset Alignment & Invariants
Verifies that 100% of raw gold annotations across `train` (10,022), `val` (1,443), and `test_reviewed` (3,345) satisfy the accounting identity:

```powershell
python scripts/run_full_alignment_audit.py
```
- **Execution Time:** ~10 seconds.
- **Output:** Logs exact match counts, 0 token truncations at 256 tokens, and accounting equality assertions.

---

## 6. Colab Notebook Validation & GPU Re-Execution

### 6.1 Validate Notebook Invariants
Verifies that the standalone Colab notebook adheres to all 8 architectural and methodological gates without running training:

```powershell
python scripts/validate_colab_notebook.py
```
*Expected Output:*
```text
[AUDIT] Loaded notebook: 40 total cells (19 code, 21 markdown).
[PASS] Check 1: All 19 executable code cells passed Python AST syntax validation.
[PASS] Check 2: Hugging Face download and dataset split assertions (6136/876/1732) verified.
[PASS] Check 3: Exclusion of 'Reasoning' and 'Operation' metadata confirmed.
[PASS] Check 4: All 10 pre-training verification unit tests present and properly constructed.
[PASS] Check 5: Maximum bipartite matching algorithm satisfies all mathematical invariants.
[PASS] Check 6: Dual-bound reporting policy (Resolvable subset vs Worst-Case bounds) strictly implemented.
[PASS] Check 7: Validation-exclusive threshold calibration and frozen checkpoint reload verified.
[PASS] Check 8: All 8 promised artifacts verified in ZIP packaging cell.
ALL NOTEBOOK PRE-RUN AUDIT CHECKS PASSED (8/8 INVARIANTS VERIFIED)
```

### 6.2 Full GPU Re-Training in Google Colab (Optional)
To retrain the 3-epoch transformer from scratch on an NVIDIA GPU:
1. Open [`notebooks/02_sbdh_transformer.ipynb`](file:///d:/Evernorth/Clinical-DSS/notebooks/02_sbdh_transformer.ipynb) in Google Colab.
2. Select **Runtime > Change runtime type > T4 GPU**.
3. Run all cells sequentially (**Runtime > Run all**).
4. Training completes in ~12 minutes across 6,136 notes.
5. The final cell downloads `evernorth_m3_results.zip`, containing `best_model.pt`, `training_curves.png`, and metrics.

---

## 7. Track B: Experimental Candidate Evaluation & Metric Tracing (Arm 6A)

> [!NOTE]
> **EXPERIMENTAL DESIGNATION:**  
> The Arm 6A pipeline is documented as an **experimental candidate**, not the production model. The production MVP in `app/app.py` and the authoritative frozen baseline (`reports/sbdh_colab_transformer_metrics_corrected.json`) remain completely untouched.

### 7.1 Instant Metric Verification from Saved Artifacts (< 2 seconds)
To mathematically verify every reported candidate metric, dual bounds, per-category breakdown, and baseline deltas directly from the saved JSON artifact:

```powershell
python scripts/verify_candidate_test_metrics.py
```

- **Execution Time:** ~1 to 2 seconds.
- **Verification Assertions (10/10 passed):**
  - Checkpoint SHA-256 bit-identical match (`0cb00010631b089746fdd97ff3f9019e1bcc29012a7c093b33501a9c30186ba4`).
  - Annotation accounting invariant: 3,345 raw = 3,340 resolvable + 5 ambiguous cases (`ex_no 7556`, `7589`, `7843`, `8581`, `8654`).
  - Resolvable exact metrics: TP=1,698, FP=1,527, FN=1,642 $\to$ P=0.5265, R=0.5084, F1=0.5173.
  - Worst-case exact metrics: TP=1,698, FP=1,527, FN=1,647 $\to$ P=0.5265, R=0.5076, F1=0.5169.
  - Resolvable relaxed metrics: TP=2,762, FP=463, FN=578 $\to$ P=0.8564, R=0.8269, F1=0.8414.
  - Worst-case relaxed metrics: TP=2,762, FP=463, FN=583 $\to$ P=0.8564, R=0.8257, F1=0.8408.
  - Boundary error accounting: 1,064 errors = 2,762 relaxed TP - 1,698 exact TP (38.52%).
  - Aggregate note classification: Micro F1=0.8949, Macro F1=0.8773.
  - Sum of category support = 3,340; sum of category preds = 3,225; sum of category exact TP = 1,698; sum of category relaxed TP = 2,762.
  - All 15 individual categories satisfy exact arithmetic equality ($P = \frac{TP}{TP+FP}$, $R = \frac{TP}{TP+FN}$, $F_1 = \frac{2PR}{P+R}$).

### 7.2 Full Test Set Re-Evaluation (Inference on 1,732 Notes)
To rerun the full forward-pass evaluation of the frozen candidate pipeline across all 1,732 reviewed test notes from scratch:

```powershell
python scripts/evaluate_candidate_test_set.py
```

- **Execution Time:** ~8 minutes on multi-core CPU (using PyTorch `torch.inference_mode()`).
- **Input Artifacts:**
  - Test Set: `D:\Evernorth\Synth-SBDH\synth_sbdh_test_reviewed.csv` ($N = 1,732$ notes)
  - Model Checkpoint: `evernorth_m3_results/models/sbdh_transformer/best_model.pt`
- **Output Artifacts Generated:**
  - [`reports/experiments/candidate_test_evaluation/candidate_test_evaluation_metrics.json`](file:///d:/Evernorth/Clinical-DSS/reports/experiments/candidate_test_evaluation/candidate_test_evaluation_metrics.json)
- **Console Output Confirms:**
  - Checkpoint SHA-256 before & after: `0cb00010...ba4` (Bit-identical).
  - Exact Kuhn maximum bipartite matching order-invariance.
  - Complete comparative benchmark table with absolute deltas.

### 7.3 Reproduce Validation Ablation (Experiment 2.2)
To re-evaluate all 6 experimental arms across 876 validation notes (`synth_sbdh_val.csv`):

```powershell
python scripts/run_span_boundary_ablation.py
```

- **Execution Time:** ~35 seconds (using cached validation logits in `reports/cached_val_notes.pt`).
- **Output Artifact:**
  - [`reports/experiments/span_boundary_ablation/span_boundary_ablation_metrics.json`](file:///d:/Evernorth/Clinical-DSS/reports/experiments/span_boundary_ablation/span_boundary_ablation_metrics.json)

### 7.4 Metric Traceability Table

Every number reported in Section 12 of the Technical Report traces directly to a specific JSON key:

| Reported Table Metric | JSON Key in `candidate_test_evaluation_metrics.json` | Formula / Derivation |
| :--- | :--- | :--- |
| **Total Predicted Spans (3,225)** | `total_pred_spans` | Count of unique predicted spans across all 1,732 notes |
| **Notes with $\ge 1$ Pred (1,726)** | `notes_with_predictions` | Count of notes having non-empty prediction set |
| **Snaps Applied (91)** | `snaps_count` | Number of spans modified by lexical snapping |
| **Merges Applied (16)** | `merges_count` | Number of span pairs fused under $\text{gap} \le 2$ |
| **Exact TP (1,698)** | `span_level_exact_matching.resolvable_subset.tp` | Exact string and offset match $(s, e, c)$ |
| **Exact FP (1,527)** | `span_level_exact_matching.resolvable_subset.fp` | Total preds $(3,225) - \text{Exact TP } (1,698)$ |
| **Exact FN (1,642)** | `span_level_exact_matching.resolvable_subset.fn` | Resolvable gold $(3,340) - \text{Exact TP } (1,698)$ |
| **Exact Prec (0.5265)** | `span_level_exact_matching.resolvable_subset.precision` | $1,698 / (1,698 + 1,527) = 1,698 / 3,225$ |
| **Exact Rec (0.5084)** | `span_level_exact_matching.resolvable_subset.recall` | $1,698 / (1,698 + 1,642) = 1,698 / 3,340$ |
| **Exact F1 (0.5173)** | `span_level_exact_matching.resolvable_subset.f1_score` | $2 \times 0.5265 \times 0.5084 / (0.5265 + 0.5084)$ |
| **Worst-Case Rec (0.5076)**| `span_level_exact_matching.worst_case_bound...recall` | $1,698 / (1,698 + 1,642 + 5) = 1,698 / 3,345$ |
| **Worst-Case F1 (0.5169)** | `span_level_exact_matching.worst_case_bound...f1_score` | Harmonic mean using worst-case recall |
| **Relaxed TP (2,762)** | `span_level_relaxed_matching.resolvable_subset.tp` | Maximum bipartite matching overlapping pairs |
| **Relaxed FP (463)** | `span_level_relaxed_matching.resolvable_subset.fp` | Total preds $(3,225) - \text{Relaxed TP } (2,762)$ |
| **Relaxed FN (578)** | `span_level_relaxed_matching.resolvable_subset.fn` | Resolvable gold $(3,340) - \text{Relaxed TP } (2,762)$ |
| **Relaxed Prec (0.8564)** | `span_level_relaxed_matching.resolvable_subset.precision` | $2,762 / (2,762 + 463) = 2,762 / 3,225$ |
| **Relaxed Rec (0.8269)** | `span_level_relaxed_matching.resolvable_subset.recall` | $2,762 / (2,762 + 578) = 2,762 / 3,340$ |
| **Relaxed F1 (0.8414)** | `span_level_relaxed_matching.resolvable_subset.f1_score` | $2 \times 0.8564 \times 0.8269 / (0.8564 + 0.8269)$ |
| **Boundary Errors (1,064)**| `boundary_errors_count` | Relaxed TP $(2,762) - \text{Exact TP } (1,698)$ |
| **Boundary Rate (38.52%)** | `boundary_error_rate_pct` | $1,064 / 2,762 \times 100\%$ |
| **Note Micro F1 (0.8949)** | `aggregate_classification_metrics.micro.f1_score` | Note-level multi-label category micro F1 |

---

## 8. Artifact Locations & File Map

| Artifact Purpose | Path |
|---|---|
| Track A Primary Model | [`models/baseline_a_logreg.joblib`](file:///d:/Evernorth/Clinical-DSS/models/baseline_a_logreg.joblib) |
| Track A Metrics JSON | [`reports/readmission_baselines_metrics.json`](file:///d:/Evernorth/Clinical-DSS/reports/readmission_baselines_metrics.json) |
| Track A Calibration JSON | [`reports/readmission_calibration_metrics.json`](file:///d:/Evernorth/Clinical-DSS/reports/readmission_calibration_metrics.json) |
| Track B Colab Weights | [`evernorth_m3_results/models/sbdh_transformer/best_model.pt`](file:///d:/Evernorth/Clinical-DSS/evernorth_m3_results/models/sbdh_transformer/best_model.pt) |
| Track B Frozen Baseline Metrics JSON | [`reports/sbdh_colab_transformer_metrics_corrected.json`](file:///d:/Evernorth/Clinical-DSS/reports/sbdh_colab_transformer_metrics_corrected.json) |
| Track B Arm 6A Candidate Metrics JSON | [`reports/experiments/candidate_test_evaluation/candidate_test_evaluation_metrics.json`](file:///d:/Evernorth/Clinical-DSS/reports/experiments/candidate_test_evaluation/candidate_test_evaluation_metrics.json) |
| Track B Arm 6A Candidate Technical Report | [`reports/experiments/candidate_test_evaluation/candidate_test_evaluation_report.md`](file:///d:/Evernorth/Clinical-DSS/reports/experiments/candidate_test_evaluation/candidate_test_evaluation_report.md) |
| Track B Validation Ablation Report | [`reports/experiments/span_boundary_ablation/span_boundary_ablation_report.md`](file:///d:/Evernorth/Clinical-DSS/reports/experiments/span_boundary_ablation/span_boundary_ablation_report.md) |
| Track B Rules Metrics JSON | [`reports/sbdh_rules_metrics_corrected.json`](file:///d:/Evernorth/Clinical-DSS/reports/sbdh_rules_metrics_corrected.json) |
| Candidate Metric Verification Runner | [`scripts/verify_candidate_test_metrics.py`](file:///d:/Evernorth/Clinical-DSS/scripts/verify_candidate_test_metrics.py) |
| Candidate Test Set Evaluation Runner | [`scripts/evaluate_candidate_test_set.py`](file:///d:/Evernorth/Clinical-DSS/scripts/evaluate_candidate_test_set.py) |
| Baseline Freeze Manifest | [`reports/baseline_freeze_manifest.md`](file:///d:/Evernorth/Clinical-DSS/reports/baseline_freeze_manifest.md) |
| Final Consistency Audit | [`reports/final_evidence_consistency_audit.md`](file:///d:/Evernorth/Clinical-DSS/reports/final_evidence_consistency_audit.md) |
| Portable Baseline Archive | [`reports/baseline_freeze_archive_20261026.zip`](file:///d:/Evernorth/Clinical-DSS/reports/baseline_freeze_archive_20261026.zip) |

---

## 9. Streamlit Clinical CDS MVP Execution

The interactive Streamlit application exposes both Track A (30-day readmission prediction) and Track B (SBDH entity extraction):

```powershell
streamlit run app/app.py
```

### Features & Pipeline Selection:
- **Track A:** Patient-specific readmission risk stratification using frozen Baseline A (`models/baseline_a_logreg.joblib`, ECE = 0.0061).
- **Track B Promoted Pipeline:** Runs Arm 6A (`Strategy B` active threshold 0.30 + delimiter-protected lexical snapping + conservative fusion with $\text{gap} \le 2$).
- **Track B Baseline Fallback:** User can switch via radio selector to `Strategy A (Frozen Baseline Fallback)` to reproduce original control behavior.
- **Latency & Runtime:** Model weights (~328MB) are cached once at startup via `@st.cache_resource` in ~1.39s. Real-time note inference executes via `torch.inference_mode()` in ~113 ms per note on standard CPU.

