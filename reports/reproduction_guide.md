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
*Expected Result:* **32 passed in ~27 seconds.**

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

## 7. Artifact Locations & File Map

| Artifact Purpose | Path |
|---|---|
| Track A Primary Model | [`models/baseline_a_logreg.joblib`](file:///d:/Evernorth/Clinical-DSS/models/baseline_a_logreg.joblib) |
| Track A Metrics JSON | [`reports/readmission_baselines_metrics.json`](file:///d:/Evernorth/Clinical-DSS/reports/readmission_baselines_metrics.json) |
| Track A Calibration JSON | [`reports/readmission_calibration_metrics.json`](file:///d:/Evernorth/Clinical-DSS/reports/readmission_calibration_metrics.json) |
| Track B Colab Weights | [`evernorth_m3_results/models/sbdh_transformer/best_model.pt`](file:///d:/Evernorth/Clinical-DSS/evernorth_m3_results/models/sbdh_transformer/best_model.pt) |
| Track B Colab Metrics JSON | [`reports/sbdh_colab_transformer_metrics_corrected.json`](file:///d:/Evernorth/Clinical-DSS/reports/sbdh_colab_transformer_metrics_corrected.json) |
| Track B Rules Metrics JSON | [`reports/sbdh_rules_metrics_corrected.json`](file:///d:/Evernorth/Clinical-DSS/reports/sbdh_rules_metrics_corrected.json) |
| Baseline Freeze Manifest | [`reports/baseline_freeze_manifest.md`](file:///d:/Evernorth/Clinical-DSS/reports/baseline_freeze_manifest.md) |
| Final Consistency Audit | [`reports/final_evidence_consistency_audit.md`](file:///d:/Evernorth/Clinical-DSS/reports/final_evidence_consistency_audit.md) |
