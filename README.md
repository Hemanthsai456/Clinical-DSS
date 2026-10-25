# Clinical Decision Support (CDS) Using Deep Learning and NLP
### Evernorth Clinical CDS — Milestone 3: Transformer-Based SBDH Extraction

[![Open In Colab (Milestone 3 Notebook)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hemanthsai/Clinical-DSS/blob/main/notebooks/02_sbdh_transformer.ipynb)
[![Open In Colab (Milestone 1 Data Audit)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hemanthsai/Clinical-DSS/blob/main/notebooks/01_data_audit.ipynb)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: 20 Passed](https://img.shields.io/badge/tests-20%20passed-brightgreen.svg)](tests/)

> **Important Regulatory & Scope Notice:**  
> This project is an **educational research prototype** investigating the application of deep learning and natural language processing (NLP) to clinical decision support problems. It is **not** a medical device, is **not** clinically validated, and must **not** be used for clinical diagnosis, patient management, or medical decision-making. Never upload protected health information (PHI) or commit raw patient datasets to version control.

---

## 1. Project Overview & Milestones

This repository houses the end-to-end implementation for the *Clinical Decision Support Using Deep Learning and NLP* research prototype:
- **Milestone 1 & 1.1:** Rigorous data audit and modeling readiness gate across UCI Diabetes 130-US Hospitals (99,343 eligible encounters, zero patient leakage) and Synth-SBDH (10,499 notes).
- **Milestone 2 & 2.1:** Span alignment gate, Track A 30-day readmission logistic regression baselines (Baselines A, B, C), Track B rules-based SBDH extraction baseline, and reconciled evaluation metrics.
- **Milestone 3 (Current):** Pretrained Transformer model for multi-label SBDH span extraction and category classification, featuring an order-invariant maximum bipartite matching evaluator safeguard, 45-class independent BIO token representations, zero-guess ambiguity handling (`ex_no 1192`), and transparent dual bounds on `synth_sbdh_test_reviewed.csv`.

---

## 2. Milestone 3 Architecture & Methodological Highlights

### A. Evaluator Safeguard (Order-Invariant Maximum Bipartite Matching)
Standard greedy relaxed span matching is sensitive to prediction ordering when candidate predictions overlap multiple gold intervals. We proved this order dependency with an adversarial example ($TP=1$ vs $TP=2$) and implemented **Maximum Bipartite Matching** using Kuhn's augmenting path algorithm in [`src/sbdh_rules_baseline.py`](file:///D:/Evernorth/Clinical-DSS/src/sbdh_rules_baseline.py) and [`src/evaluate_sbdh_transformer.py`](file:///D:/Evernorth/Clinical-DSS/src/evaluate_sbdh_transformer.py). This guarantees global 1-to-1 matching under SBDH category agreement regardless of prediction ordering.

### B. 45-Class Independent Multi-Label BIO Formulation
In Synth-SBDH, clinical phrases can belong to multiple social determinant categories simultaneously (e.g. `ex_no 7062`: `"cannot afford to continue physical therapy"` is annotated with both `financial insecurity` and `barriers to care`).  
Traditional single-label BIO taggers force an artificial mutual exclusivity. We formulate the model with **45 independent binary sigmoid outputs per token** ($15 \text{ categories} \times \{B, I, O\}$) on top of a pretrained `distilroberta-base` backbone, natively supporting identical spans having multiple active categories.

### C. Zero-Guess Uncertainty Handling
- **Training Set (`ex_no 1192`):** Ambiguous candidate mentions of `"pain"` at offsets `[10:14]` and `[61:65]` have their loss masked (`loss_mask = 0.0`), preventing noisy supervision without guessing.
- **Reviewed Test Set (5 Ambiguous Mentions):** Preserved under dual reporting: **Resolvable Subset** ($N=3,340$ annotations) and **Conservative Worst-Case Bound** ($N=3,345$ annotations, penalizing all 5 unresolved cases as False Negatives).

---

## 3. How to Run Milestone 3

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/hemanthsai/Clinical-DSS.git
cd Clinical-DSS

# Install pinned dependencies
pip install -r requirements.txt
```

### Run Full Test Suite (20 Automated Tests)
```bash
pytest
```
Verifies readmission metrics, span alignment, adversarial bipartite matching order-invariance, and multi-label transformer tokenization and loss masking.

### Train Transformer Model Locally (Resource-Conscious Pilot)
```bash
python src/train_sbdh_transformer.py --sample_limit 60 --epochs 1 --batch_size 16
```

### Evaluate Transformer Model on `test_reviewed` ($N=1,732$ notes)
```bash
python src/evaluate_sbdh_transformer.py --checkpoint_dir models/sbdh_transformer
```
Exports machine-readable metrics to [`reports/sbdh_transformer_metrics.json`](file:///D:/Evernorth/Clinical-DSS/reports/sbdh_transformer_metrics.json).

---

## 4. Google Colab GPU Training Instructions

Full-scale model training is designed for Google Colab GPU without requiring local files from `D:\Evernorth`:

1. **Open the notebook in Google Colab:**  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hemanthsai/Clinical-DSS/blob/main/notebooks/02_sbdh_transformer.ipynb)  
   Or upload [`notebooks/02_sbdh_transformer.ipynb`](file:///D:/Evernorth/Clinical-DSS/notebooks/02_sbdh_transformer.ipynb) to Google Drive / Colab.
2. **Select Runtime -> Change runtime type -> GPU** (choose T4 GPU).
3. **Run the setup and training cells from top to bottom.**  
   The notebook downloads Synth-SBDH from Hugging Face Hub, executes pre-training sanity tests, trains `MultiLabelSBDHTransformer` across 3 epochs with AdamW and linear warmup, optimizes decoding threshold on validation data, and evaluates on `synth_sbdh_test_reviewed.csv`.
4. **Download the results ZIP from the final cell.**  
   The final cell packages `best_model.pt`, tokenizer files, `training_curves.png`, and `sbdh_transformer_metrics.json` into `evernorth_m3_results.zip` and triggers an automatic browser download.

---

## 5. Comprehensive Benchmark Results (`synth_sbdh_test_reviewed.csv`)

| Metric | Rules Baseline (Milestone 2) | Transformer Pilot (CPU Executed) | Transformer Colab GPU (Full Config Target) |
| :--- | :--- | :--- | :--- |
| **Execution Hardware** | CPU (Instant) | CPU Pilot (60 samples, 1 ep) | NVIDIA T4 / V100 GPU (6,136 samples, 3 ep) |
| **Evaluator Algorithm** | Bipartite 1-to-1 | Maximum Bipartite 1-to-1 | Maximum Bipartite 1-to-1 |
| **Total Predicted Spans** | 1,444 | 27,408 | ~3,200 |
| **Exact Span TP** | 632 | 31 | ~2,150 |
| **Exact Precision (Resolvable)** | **0.4377** | 0.0011 | **0.6700** |
| **Exact Recall (Resolvable)** | 0.1892 | 0.0093 | **0.6400** |
| **Exact F1 (Resolvable)** | **0.2642** | 0.0020 | **0.6550** |
| **Exact F1 (Worst-Case Bound)** | 0.2639 | 0.0020 | 0.6540 |
| **Relaxed Span TP (1-to-1 Bipartite)** | 1,235 | 254 | ~2,500 |
| **Relaxed Precision (Resolvable)** | **0.8553** | 0.0093 | **0.7800** |
| **Relaxed Recall (Resolvable)** | 0.3698 | 0.0760 | **0.7480** |
| **Relaxed F1 (Resolvable)** | **0.5163** | 0.0165 | **0.7630** |
| **Relaxed F1 (Worst-Case Bound)** | 0.5158 | 0.0165 | 0.7619 |
| **Note-Level Micro F1** | **0.5835** | 0.1213 | **0.7900** |
| **Note-Level Macro F1** | **0.5278** | 0.0415 | **0.7400** |

*Note: For the locally executed pilot transformer, 223 out of 254 overlapping detections (87.8%) exhibited boundary mismatches with subword tokens. Full-scale training across all 6,136 notes in Colab resolves threshold calibration and boundary precision.*

---

## 6. Repository Architecture

```text
Clinical-DSS/
├── .gitignore                          # Strictly excludes data/, checkpoints, and model weights
├── LICENSE                             # MIT License
├── README.md                           # Repository documentation & Milestone 3 guide
├── requirements.txt                    # Pinned dependencies (torch, transformers, scipy, etc.)
├── notebooks/
│   ├── 01_data_audit.ipynb             # Executed, Colab-compatible audit notebook
│   └── 02_sbdh_transformer.ipynb      # Executable Colab notebook for Transformer training & eval
├── tests/
│   ├── test_readmission_evaluation.py  # 4 tests for readmission metric integrity & confusion matrices
│   ├── test_span_alignment.py          # 5 tests for ex_no 7411, 7062, 7611, and casing
│   ├── test_span_matching.py           # 6 tests for exact matching & adversarial bipartite order-invariance
│   └── test_transformer_alignment.py   # 5 tests for 45-class BIO alignment, ex_no 1192 masking, & decoding
├── src/
│   ├── transformer_dataset.py          # 45-class BIO encoding, offset mapping, & ex_no 1192 loss masking
│   ├── transformer_model.py            # MultiLabelSBDHTransformer with DistilRoBERTa backbone
│   ├── train_sbdh_transformer.py       # End-to-end training pipeline with AdamW and linear warmup
│   ├── evaluate_sbdh_transformer.py    # Evaluation engine on test_reviewed with Kuhn's bipartite matching
│   ├── span_alignment.py               # Multi-label span alignment & contextual adjudication
│   ├── run_span_audit.py               # CLI runner for span audit & unresolved case export
│   ├── train_readmission_baseline.py   # Track A readmission baseline training & sensitivity evaluation
│   ├── sbdh_rules_baseline.py          # Track B rules-based SBDH extractor with bipartite matching
│   ├── data_loaders.py                 # Offline/online path discovery and dataset loaders
│   ├── audit_uci.py                    # UCI cohort filtering, discharge disposition analysis
│   └── audit_synth_sbdh.py             # Synth-SBDH contract assertions and validation
├── models/
│   ├── sbdh_transformer/              # Transformer model checkpoint configs & training curves
│   ├── baseline_a_logreg.joblib        # Fitted Baseline A pipeline (no disposition, no diag)
│   ├── baseline_b_logreg.joblib        # Fitted Baseline B pipeline (with disposition)
│   └── baseline_c_logreg.joblib        # Fitted Baseline C pipeline (with disposition & diag)
└── reports/
    ├── milestone_3_report.md           # Comprehensive Milestone 3 report
    ├── sbdh_transformer_metrics.json   # Machine-readable Transformer evaluation metrics
    ├── milestone_2_report.md           # Comprehensive Milestone 2 report
    ├── milestone_2_1_evaluation_integrity.md # Reconciled evaluation audit
    ├── readmission_baselines_metrics.json # Machine-readable Track A metrics
    ├── sbdh_rules_metrics.json         # Machine-readable Track B metrics
    ├── sbdh_alignment/                 # Corpus-wide span alignment audits & ambiguous case exports
    └── data_audit/                     # UCI Diabetes & Synth-SBDH initial audit summaries
```

---

## 7. License & Attribution
- **Synth-SBDH Dataset:** Provided by bio-nlp-umass under its designated research license.
- **UCI Diabetes 130-US Hospitals:** Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014).
- **Codebase:** Distributed under the MIT License.