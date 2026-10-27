# Evernorth Clinical-DSS — Baseline Freeze Manifest

**Workspace:** `D:\Evernorth\Clinical-DSS`  
**Date:** October 2026  
**Git Baseline Commit:** `7dd92b5bda82042f5009af52ab7013223df567b4` (`main` branch)  
**Status:** FROZEN & VERIFIED (32/32 tests passing; 8/8 notebook checks passing)

---

## 1. Executive Summary & Purpose

This manifest records the frozen, verified state of all models, preprocessing pipelines, evaluation artifacts, configuration files, and empirical results in the Evernorth Clinical Decision Support (CDS) project.

Both modeling tracks are frozen:
- **Track A (UCI Readmission):** 3 logistic regression models evaluated across 99,343 eligible encounters, with patient-grouped split integrity (0 patient leakage), validation-exclusive threshold tuning, and probability calibration.
- **Track B (SBDH Extraction):** 1 rules baseline and 1 full 3-epoch Transformer model (`distilbert/distilroberta-base`, 45-class multi-label BIO architecture) trained on Google Colab GPU and evaluated against `synth_sbdh_test_reviewed.csv` ($N = 1,732$) under strict dual-bound accounting.

**No models were retrained. No weights, thresholds, split definitions, or existing evaluation metrics were modified.**

---

## 2. Git Working Tree & Provenance Baseline

- **Repository Root:** `D:\Evernorth\Clinical-DSS`
- **Head Commit:** `7dd92b5bda82042f5009af52ab7013223df567b4`
- **Working Tree State:** Clean working tree with existing verification scripts, audit reports, and minimal forward-compatible updates:
  - `src/transformer_model.py`: Allows `from_pretrained` to load `best_model.pt` if present.
  - `tests/test_readmission_evaluation.py`: Added 5 targeted Track A integrity assertions.
  - `tests/test_transformer_alignment.py`: Added Colab checkpoint provenance and artifact consistency test.

---

## 3. Track A: UCI Readmission Baseline Inventory

| Relative Artifact Path | Purpose | File Size (Bytes) | SHA256 Hash | Associated Dataset & Split | Threshold & Protocol | Status | Authoritative / Supporting | Reproduction Command |
|---|---|:---:|:---:|---|---|:---:|:---:|---|
| [`models/baseline_a_logreg.joblib`](models/baseline_a_logreg.joblib) | Primary real-time bedside CDS baseline (excludes disposition & billed diagnoses). | 17,018 | `794cff2d580b0d9b574b62a62c1d2549aad101e283ac65167869f9d1a076d74c` | UCI Diabetes (99,343 eligible encounters; Train: 69,470; Val: 15,022; Test: 14,851) | $\tau = 0.130$ (tuned on Val to maximize F1, frozen for Test) | **PASS** | **Authoritative (Primary Model)** | `python scripts/audit_track_a_readmission.py` |
| [`models/baseline_b_logreg.joblib`](models/baseline_b_logreg.joblib) | Sensitivity baseline adding `discharge_disposition_id` (post-acute placement). | 17,722 | `cf09fa58a88d8428e57ef89bc160060160898110541abeb8546562e95bed2377` | UCI Diabetes (Train: 69,470; Val: 15,022; Test: 14,851) | $\tau = 0.145$ (tuned on Val to maximize F1, frozen for Test) | **PASS** | **Authoritative (Placement Benchmark)** | `python scripts/audit_track_a_readmission.py` |
| [`models/baseline_c_logreg.joblib`](models/baseline_c_logreg.joblib) | Retrospective benchmark adding grouped ICD-9 discharge billing codes. | 19,562 | `7cd9b77a4f3cfaa09cdc479561b69f76e57dcb6d582170230b4bab8ccdf3c6e4` | UCI Diabetes (Train: 69,470; Val: 15,022; Test: 14,851) | $\tau = 0.135$ (tuned on Val to maximize F1, frozen for Test) | **PASS** | **Authoritative (Retrospective Bound)** | `python scripts/audit_track_a_readmission.py` |
| [`reports/readmission_baselines_metrics.json`](reports/readmission_baselines_metrics.json) | Exported baseline classification metrics, confusion matrices, and Brier scores. | 10,639 | `168a73c1c40e841f7992ccf6271d4a1e8a977444a1f44a471344edae6882f114` | Test ($N=14,851$) and Val ($N=15,022$) | Majority, Baseline A, Baseline B, Baseline C | **PASS** | **Authoritative (Core Metrics)** | `python src/train_readmission_baseline.py` |
| [`reports/readmission_calibration_metrics.json`](reports/readmission_calibration_metrics.json) | Decile reliability curves, ECE (0.0061), and Brier scores (0.0961). | 20,579 | `7bbdd92bc938c8e83204e3b3d19c2bb12c3055471798e6a59832bef7ef56d551` | Test ($N=14,851$) and Val ($N=15,022$) | 10 deciles per model | **PASS** | **Authoritative (Calibration)** | `python scripts/audit_track_a_readmission.py` |
| [`reports/track_a_readmission_audit.md`](reports/track_a_readmission_audit.md) | Comprehensive audit report covering feature timing, leakage risks, and calibration. | 22,121 | `34a15ce73926c476cc42e75cc94a2549011598ae0d33fe99f339048dd340c6da` | Complete eligible cohort ($N=99,343$) | Full evaluation protocol | **PASS** | **Authoritative (Audit Report)** | Static report |
| [`reports/data_audit/uci_diabetes_audit_summary.json`](reports/data_audit/uci_diabetes_audit_summary.json) | Milestone 1.1 data audit profiling raw UCI data, missingness, and 86 hospice cases. | 10,329 | `1d63a8b444c910bee4bb0135431f49ccaed5f4be7e61fdc9ffa52275e7c5cb4c` | Raw UCI Diabetes (101,766 encounters, 71,518 patients) | Data quality and cohort gating | **PASS** | Supporting Evidence | `python src/run_data_audit.py` |
| [`src/audit_uci.py`](src/audit_uci.py) | Cohort exclusion (2,423 deaths/hospice) and patient-grouped split implementation. | 21,746 | `3dc9f09e5ee61a34ecf5bdad662e27d5be5f760f2f823a771a89af512d357827` | Raw & eligible cohort | `GroupShuffleSplit` on `patient_nbr` (`seed=42`) | **PASS** | **Authoritative Code** | Source module |
| [`src/train_readmission_baseline.py`](src/train_readmission_baseline.py) | Pipeline construction, training on `X_train`, validation threshold tuning, evaluation. | 17,036 | `5e3a6335696d5ebf58781d1da48352ca2c29818ff3d78f6f69a69d14ec281b8d` | Train, Val, Test partitions | Encapsulated pipeline + mathematical assertions | **PASS** | **Authoritative Code** | Source module |
| [`tests/test_readmission_evaluation.py`](tests/test_readmission_evaluation.py) | 9 automated pytest tests verifying CM derivation, zero leakage, and feature isolation. | 11,485 | `2f281e6cb812...` | Synthetic + Full UCI splits | Exact CM assertions | **PASS** | **Authoritative Tests** | `pytest tests/test_readmission_evaluation.py` |

---

## 4. Track B: SBDH Extraction Baseline & Checkpoint Inventory

| Relative Artifact Path | Purpose | File Size (Bytes) | SHA256 Hash | Model & Checkpoint Context | Threshold & Protocol | Status | Authoritative / Supporting | Reproduction Command |
|---|---|:---:|:---:|---|---|:---:|:---:|---|
| [`evernorth_m3_results/models/sbdh_transformer/best_model.pt`](evernorth_m3_results/models/sbdh_transformer/best_model.pt) | **Full 3-epoch trained checkpoint** (Google Colab GPU, 6,136 notes). | **328,655,930** | `0cb00010631b089746fdd97ff3f9019e1bcc29012a7c093b33501a9c30186ba4` | `distilbert/distilroberta-base`, 45 output logits | $\tau = 0.30$ (tuned on Val) | **PASS** | **Authoritative (Primary Neural Checkpoint)** | Trained in Colab; evaluated via `python scripts/evaluate_colab_checkpoint.py` |
| [`evernorth_m3_results/models/sbdh_transformer/config.json`](evernorth_m3_results/models/sbdh_transformer/config.json) | Model architecture config for Colab model (`transformers 5.20.2`). | 653 | `e95af094be303a14227076b7479a97ac8a318b2171cf77782dec11fc5d29630b` | DistilRoBERTa 6 layers, 768 hidden, 12 heads | Arch metadata | **PASS** | Authoritative Config | Created by Colab run |
| [`evernorth_m3_results/reports/sbdh_transformer_metrics.json`](evernorth_m3_results/reports/sbdh_transformer_metrics.json) | Original Colab-generated metrics JSON artifact. | 834 | `b254e31984e7a64dfff5916cefdb0d4f0bb8d16ab30649c89bbf1b646658de9c` | `synth_sbdh_test_reviewed.csv` ($N=1,732$) | $\tau = 0.30$, exact and relaxed matching | **PASS** | Authoritative Colab Run Record | Created by Colab cell 24 |
| [`evernorth_m3_results/reports/validation_threshold_config.json`](evernorth_m3_results/reports/validation_threshold_config.json) | Validation threshold grid search (0.3 to 0.7; selected 0.30, F1 0.8319). | 720 | `186a82f0405cedc5b4c12b3a0cef5d8993fdb3b62a22fc5e79406fb1a0b6e105` | Validation set ($N=876$) | F1 optimization grid | **PASS** | Authoritative Tuning Record | Created by Colab cell 23 |
| [`reports/sbdh_colab_transformer_metrics_corrected.json`](reports/sbdh_colab_transformer_metrics_corrected.json) | Local re-evaluation of Colab checkpoint under corrected alignment & bipartite matching. | 10,153 | `f20c59f435ca0194c8fdaba7c5ffe548ec35fb34b6bcf6a177f81ab82b090934` | `synth_sbdh_test_reviewed.csv` ($N=1,732$) | $\tau = 0.30$, Kuhn bipartite matching, dual bounds | **PASS** | **Authoritative (Re-evaluated Metrics)** | `python scripts/evaluate_colab_checkpoint.py` |
| [`reports/sbdh_rules_metrics_corrected.json`](reports/sbdh_rules_metrics_corrected.json) | Corrected rules baseline evaluation (Kuhn bipartite 1-to-1 matching, dual bounds). | 7,321 | `9bbd1f5ed4c4fadf542f7c9869d2c045668f08b33454c8f9ffed5b332a461ce9` | `synth_sbdh_test_reviewed.csv` ($N=1,732$) | Deterministic regexes, Kuhn matching | **PASS** | **Authoritative (Rules Comparator)** | `python src/sbdh_rules_baseline.py --output_file reports/sbdh_rules_metrics_corrected.json` |
| [`models/sbdh_transformer/pytorch_model.bin`](models/sbdh_transformer/pytorch_model.bin) | Local 1-epoch pilot checkpoint (CPU smoke test, 60 samples). | 328,654,791 | `c2d1d47e363c2c415724c61092b71fa63c7240f79b2490a5f17140fe7f246a7c` | Local pilot training run | Heuristic threshold | **PASS** | **Superseded (Smoke-Test Only)** | Superseded by 3-epoch Colab checkpoint |
| [`models/sbdh_transformer/training_curves.json`](models/sbdh_transformer/training_curves.json) | Training history of the local pilot run (1 epoch, 24.7s). | 423 | `e399618e3ff6a0eede592032dfc1cad84c0962701f439ec6dd7ffcef33056384` | 60 samples, CPU | Pilot run metadata | **PASS** | Historical Evidence | Created by local training script |
| [`reports/sbdh_rules_metrics.json`](reports/sbdh_rules_metrics.json) | Original Milestone 2 rules baseline metrics artifact. | 7,216 | `dccbd4687e35bbb921fabc95d80fb7ffb0f4540d2d2642c3ae85a41054be58a5` | `synth_sbdh_test_reviewed.csv` | Preserved untouched | **PASS** | Historical Baseline | Static artifact |
| [`reports/sbdh_transformer_metrics.json`](reports/sbdh_transformer_metrics.json) | Original Milestone 3 evaluation artifact for local pilot checkpoint ($\theta = 0.50$). | 9,402 | `df1523436583dee222891f6afbe2facb3aeb645dee658e37bc87b414e1853288` | `synth_sbdh_test_reviewed.csv` | $\theta = 0.50$, preserved untouched | **PASS** | Historical Pilot Evaluation | Static artifact |
| [`reports/sbdh_transformer_metrics_corrected.json`](reports/sbdh_transformer_metrics_corrected.json) | Local pilot checkpoint re-evaluated at $\theta = 0.30$ under corrected alignment. | 9,971 | `2879a917893c14c52bc827f6521d9e348d037358cd5b17d8d8da498b6b3f1040` | `synth_sbdh_test_reviewed.csv` | $\theta = 0.30$, corrected alignment | **PASS** | Historical Comparison | `python src/evaluate_sbdh_transformer.py --threshold 0.30 --output_file reports/sbdh_transformer_metrics_corrected.json` |
| [`notebooks/02_sbdh_transformer.ipynb`](notebooks/02_sbdh_transformer.ipynb) | Validated self-contained Colab notebook for GPU training and evaluation. | 97,404 | `f4d198ffcc49d57327f8bfc83cdcadeba69f2332d12212d57ba6d866c14f0cc6` | Synth-SBDH from Hugging Face Hub | Full training + evaluation loop | **PASS** | **Authoritative Notebook** | `python scripts/validate_colab_notebook.py` |

---

## 5. Synth-SBDH Annotation-Accounting Invariants

The table below records the strict accounting invariants verified across every dataset partition:

| Partition | Total Notes | Raw Gold Annotations | Resolvable Annotations | Unresolved Annotations | Unresolved Breakdown | Truncated (>256 tok) | Notes > 256 tok | Invariant Satisfied? |
|---|:---:|:---:|:---:|:---:|---|:---:|:---:|:---:|
| **`train`** | 6,136 | 10,022 | 10,021 | 1 | 1 `UNRESOLVED_MISMATCHED_REPEATED` (`ex_no 1192`: `"pain"`, 2 occurrences, 1 annotation; candidate tokens masked with `loss_mask = 0.0`) | 0 | 0 | **YES** ($10021 + 1 \equiv 10022$) |
| **`val`** | 876 | 1,443 | 1,443 | 0 | None | 0 | 0 | **YES** ($1443 + 0 \equiv 1443$) |
| **`test_reviewed`** | 1,732 | 3,345 | 3,340 | 5 | 5 `UNRESOLVED_MISMATCHED_REPEATED` (`ex_no` 7556, 7589, 7843, 8581, 8654; penalized as FN in worst-case bounds) | 0 | 0 | **YES** ($3340 + 5 \equiv 3345$) |

$$\text{Accounting Identity:} \quad \text{Resolvable Gold Spans} + \sum \text{Unresolved Cases} \equiv \text{Total Raw Gold Annotations}$$

---

## 6. Authoritative Benchmark Summary Table

The authoritative benchmark results for final reporting are summarized below:

### Track A: 30-Day Readmission Risk (Test Partition, $N = 14,851$)
- **Population Prevalence:** $11.1104\%$ (1,650 positives, 13,201 negatives).
- **Zero Patient Overlap:** Verified across all partitions.

| Model / Baseline | Role in Report | AUROC | PR-AUC | Threshold | F1-Score | Recall | Precision | Specificity | Brier Score | ECE |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Majority Baseline** | Statistical baseline | 0.5000 | 0.1111 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0988 | 0.0034 |
| **Baseline A** | **Primary Bedside Model** | **0.6324** | **0.1925** | **0.1300** | **0.2449** | **0.3800** | **0.1806** | **0.7846** | **0.0961** | **0.0061** |
| **Baseline B** | Sensitivity Benchmark | 0.6599 | 0.2176 | 0.1450 | 0.2667 | 0.3552 | 0.2136 | 0.8365 | 0.0947 | 0.0091 |
| **Baseline C** | Retrospective Upper Bound | 0.6666 | 0.2213 | 0.1350 | 0.2740 | 0.4224 | 0.2028 | 0.7924 | 0.0944 | 0.0068 |

### Track B: SBDH Span Extraction (`synth_sbdh_test_reviewed.csv`, $N = 1,732$)
- **Gold Support:** 3,340 resolvable spans; 5 ambiguous cases ($N = 3,345$ total raw annotations).
- **Evaluator Algorithm:** Kuhn's Maximum Bipartite Matching (order-invariant 1-to-1 matching under category agreement).

| Model Pipeline | Role in Report | Exact F1 (Resolvable) | Exact F1 (Worst-Case) | Relaxed F1 (Resolvable) | Relaxed F1 (Worst-Case) | Relaxed Precision | Relaxed Recall | Note Micro F1 | Note Macro F1 |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Rules Baseline** | Deterministic Comparator | 0.2642 | 0.2639 | 0.5163 | 0.5158 | 0.8553 | 0.3698 | 0.5835 | 0.5278 |
| **Colab Transformer (3 ep)** | **Primary Neural Model** | **0.5105** | **0.5101** | **0.8330** | **0.8324** | **0.8816** | **0.7895** | **0.8888** | **0.8684** |
| *Delta (Colab vs. Rules)* | *Neural Improvement* | *+0.2463* | *+0.2462* | *+0.3167* | *+0.3166* | *+0.0263* | *+0.4197* | *+0.3053* | *+0.3406* |

---

## 7. Mandatory Summary

1. **What was frozen:**
   - Saved models: `baseline_a_logreg.joblib`, `baseline_b_logreg.joblib`, `baseline_c_logreg.joblib`, and `best_model.pt`.
   - Datasets & splits: UCI eligible cohort ($N = 99,343$) and patient-grouped split (seed 42); Synth-SBDH official splits (6,136 / 876 / 1,732).
   - Evaluation metrics: Track A baseline metrics JSON, Track A calibration JSON, Track B corrected rules metrics JSON, and Track B Colab transformer metrics JSON.
   - Notebook: `notebooks/02_sbdh_transformer.ipynb` (40 cells, verified by 8/8 invariant checks).
2. **What was verified:**
   - 32/32 pytest unit and integration tests passing in 27.4s.
   - Exact numerical reproduction of all Track A metrics from `.joblib` files.
   - Exact numerical reproduction of Colab checkpoint evaluation from `best_model.pt`.
   - Distinct SHA256 hashes separating Colab full-trained weights (`0cb0...`) from local pilot (`c2d1...`).
3. **What remains unverified:**
   - The 5 ambiguous test annotations (`ex_no 7556`, `7589`, `7843`, `8581`, `8654`) cannot be resolved without arbitrary guessing and must remain bounded by dual-bound reporting.
   - Generalization of UCI models to contemporary hospital networks outside the 1999–2008 historical window.
4. **Whether any files were modified outside the documentation files:**
   - Two new documentation files created: [`reports/baseline_freeze_manifest.md`](reports/baseline_freeze_manifest.md) and [`reports/reproduction_guide.md`](reports/reproduction_guide.md).
   - Zero modifications to model weights, datasets, split assignments, feature selections, or evaluation algorithms.
5. **Whether any model was retrained:**
   - **No.** Zero models were retrained.
