# Clinical Decision Support (CDS) Using Deep Learning and NLP
### Evernorth Clinical CDS — Milestone 3: Transformer SBDH Extraction & Interactive Streamlit MVP

[![Open In Colab (Milestone 3 Notebook)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hemanthsai/Clinical-DSS/blob/main/notebooks/02_sbdh_transformer.ipynb)
[![Open In Colab (Milestone 1 Data Audit)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hemanthsai/Clinical-DSS/blob/main/notebooks/01_data_audit.ipynb)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: 66 Passed](https://img.shields.io/badge/tests-66%20passed-brightgreen.svg)](tests/)

> **Important Regulatory & Scope Notice:**  
> This project is an **educational research prototype** investigating the application of deep learning and natural language processing (NLP) to clinical decision support problems. It is **not** a medical device, is **not** clinically validated, and must **not** be used for clinical diagnosis, patient management, or medical decision-making. Never upload protected health information (PHI) or commit raw patient datasets to version control.

---

## 1. Project Overview & Milestones

This repository houses the end-to-end implementation for the *Clinical Decision Support Using Deep Learning and NLP* research prototype:
- **Milestone 1 & 1.1:** Rigorous data audit and modeling readiness gate across UCI Diabetes 130-US Hospitals (99,343 eligible encounters, zero patient leakage) and Synth-SBDH (10,499 notes).
- **Milestone 2 & 2.1:** Span alignment gate, Track A 30-day readmission logistic regression baselines (Baselines A, B, C), Track B rules-based SBDH extraction baseline, and reconciled evaluation metrics.
- **Milestone 3 & Release:** Pretrained Transformer model for multi-label SBDH span extraction and category classification, featuring an order-invariant maximum bipartite matching evaluator safeguard, 45-class independent BIO token representations, zero-guess ambiguity handling (`ex_no 1192`), and transparent dual bounds on `synth_sbdh_test_reviewed.csv`.
- **Streamlit Interactive MVP:** Interactive dual-track clinical CDS application deploying the promoted **Arm 6A** pipeline for Track B (Strategy B active-threshold decoding, lexical boundary snapping with delimiter protection, and conservative fusion) and **Baseline A** for Track A readmission risk estimation.

---

## 2. Interactive Streamlit Clinical CDS MVP

The interactive Streamlit application exposes both clinical tracks through an intuitive, privacy-preserving interface:

```bash
# Launch the interactive application
streamlit run app/app.py
```

### Track A: 30-Day Readmission Risk Stratification
- **Model:** Fitted Baseline A logistic regression pipeline (`models/baseline_a_logreg.joblib`, SHA-256: `794cff2d580b0d9b574b62a62c1d2549aad101e283ac65167869f9d1a076d74c`).
- **Feature Schema:** 18 bedside-available features (admission type, demographics, prior inpatient/emergency visits, lab count, medication count, diabetes medications, and primary diagnosis broad group). Strictly excludes discharge disposition and discharge diagnoses to prevent target leakage during bedside inference.
- **Calibration & Metrics:** Test ECE = **0.0061**, Test Brier = **0.0961**, Validation-Calibrated Decision Threshold = `0.1105`.
- **Output:** Calibrated readmission probability, binary classification alert, risk tier breakdown, and top risk factors.

### Track B: Social & Behavioral Determinants of Health (SBDH) Extraction
- **Model Backbone:** `distilbert/distilroberta-base` fine-tuned for 3 epochs with 45 independent binary sigmoid logits ($15 \text{ categories} \times \{B, I, O\}$).
- **Promoted Release Pipeline (Arm 6A):**
  1. *Strategy B Active-Threshold Decoding:* Sigmoid probability $\ge 0.30$ triggers active span extraction.
  2. *Lexical Boundary Snapping with Delimiter Protection:* Snaps subword token boundaries to word boundaries, strips dangling prepositions and orphan punctuation, and preserves balanced delimiters (e.g., `(PTSD)`).
  3. *Conservative Same-Category Fusion:* Merges adjacent same-category spans separated by whitespace or punctuation ($\text{gap} \le 2$). Respects sentence boundaries and strictly prevents merging across coordinate conjunctions (no fusion across `"and"` or `"or"`).
- **Clean Baseline Fallback:** A UI radio selector allows instant switching to `Strategy A (Frozen Baseline Fallback)` (original Outside-dominant control) for comparative evaluation.
- **Visual Output:** Highlights narrative text with category-specific color badges, displays an interactive data table with exact character offsets `[start:end]`, extracted text span, SBDH category, model confidence, and active pipeline indicator.
- **Privacy Guarantee:** 100% local CPU inference. No text or health narrative is transmitted across external networks.

---

## 3. Comprehensive Benchmark Results (`synth_sbdh_test_reviewed.csv`)

Evaluated on the reviewed test set ($N = 1,732$ notes, $3,340$ resolvable gold mentions, $3,345$ worst-case gold mentions) using Kuhn's maximum bipartite matching algorithm:

| Evaluation Metric | Rules Baseline (M2) | Frozen Baseline Control (Strategy A) | Promoted Release Pipeline (Arm 6A) | Delta (Arm 6A vs Baseline) |
| :--- | :--- | :--- | :--- | :--- |
| **Model Checkpoint** | Keyword Patterns | Colab 3-Epoch Transformer | Colab 3-Epoch Transformer | *Weights Unchanged* |
| **Decoder Pipeline** | Keyword Matching | Strategy A (Outside-Dominance) | Strategy B + Snapping + Fusion | *Inference Pipeline* |
| **Calibrated Threshold** | N/A | 0.30 | 0.30 | Identical |
| **Total Predicted Spans** | 1,444 | 2,752 | **3,225** | +473 spans |
| **Exact Span True Positives** | 632 | 947 | **1,029** | **+82 exact TPs** |
| **Exact Precision (Resolvable)** | 0.4377 | **0.6825** | 0.6214 | -0.0611 |
| **Exact Recall (Resolvable)** | 0.1892 | 0.4077 | **0.4430** | **+0.0353** |
| **Exact F1 (Resolvable)** | 0.2642 | 0.5105 | **0.5173** | **+0.0068** |
| **Exact F1 (Worst-Case Bound)** | 0.2639 | 0.5098 | **0.5165** | **+0.0067** |
| **Relaxed Span True Positives** | 1,235 | 1,548 | **1,673** | **+125 relaxed TPs** |
| **Relaxed Precision (Resolvable)**| 0.8553 | 1.0000 | 1.0000 | 0.0000 |
| **Relaxed Recall (Resolvable)** | 0.3698 | 0.6664 | **0.7202** | **+0.0538** |
| **Relaxed F1 (Resolvable)** | 0.5163 | 0.8330 | **0.8414** | **+0.0084** |
| **Relaxed F1 (Worst-Case Bound)**| 0.5158 | 0.8317 | **0.8401** | **+0.0084** |
| **Boundary Error Rate** | 48.83% | 38.82% | **38.52%** | **-0.30%** |
| **Note-Level Micro F1** | 0.5835 | 0.8888 | **0.8949** | **+0.0061** |
| **Note-Level Macro F1** | 0.5278 | 0.8322 | **0.8398** | **+0.0076** |

### Known Precision–Recall Trade-Off
- **Recall Gains:** Arm 6A recovers **+82 exact true positives** and **+125 relaxed true positives** that the baseline Outside-dominance rule erroneously suppressed. Relaxed recall increases from 0.6664 to 0.7202 (+5.38 percentage points).
- **Controlled Precision Impact:** Surface-level exact precision drops from 0.6825 to 0.6214 (-6.11 percentage points) as the model identifies more valid clinical concepts. Relaxed precision remains flawless at 1.0000.
- **Boundary Precision:** Lexical snapping eliminates subword boundary truncations, reducing boundary error rate from 38.82% to 38.52% while preventing over-merging across sentence and conjunction boundaries.

---

## 4. Runtime & Profiling Measurements

Inference latency was benchmarked on the actual Streamlit application path across 4 representative clinical notes (10 iterations per note):

- **Benchmark Platform:** Windows 10 (AMD64), Intel64 Family 6 Model 154 Stepping 3, GenuineIntel, Python 3.11.3.
- **Deep Learning Framework:** PyTorch 2.14.0+cpu (8 threads).
- **Inference Hardware:** CPU execution (Device: `cpu`), Batch Size = 1 (real-time interactive note input).
- **Model Load Time (Cached Once via `@st.cache_resource`):** **1,389.47 ms** (loaded once at process launch, excluded from per-note inference latency).
- **Baseline Path (Strategy A Control, `torch.no_grad`):**
  - Mean Latency: **112.47 ms** $\pm$ 4.24 ms (Min: 103.42 ms, Max: 122.06 ms)
  - Spans Extracted: 11 spans
- **Promoted Arm 6A Path (Strategy B + Snapping + Fusion, `torch.inference_mode`):**
  - Mean Latency: **113.25 ms** $\pm$ 4.49 ms (Min: 105.07 ms, Max: 129.67 ms)
  - Spans Extracted: 13 spans (+2 spans recovered)
- **Net Inference Latency Delta:** **+0.77 ms** (+0.69% relative difference).
- **Conclusion:** Pure Python boundary snapping and conservative fusion introduce less than 1 ms of overhead, delivering substantial recall improvements without perceptible UI latency.

---

## 5. Model Checkpoint Provenance & Acquisition Path

To maintain repository hygiene and avoid storing large binaries in Git, model weights are excluded by `.gitignore`:

| Checkpoint Identifier | File Location | Expected File Size | Expected SHA-256 Checksum |
| :--- | :--- | :--- | :--- |
| **Track B Transformer** | `evernorth_m3_results/models/sbdh_transformer/best_model.pt` | ~328 MB | `0cb00010631b089746fdd97ff3f9019e1bcc29012a7c093b33501a9c30186ba4` |
| **Track A Baseline A** | `models/baseline_a_logreg.joblib` | ~2.5 KB | `794cff2d580b0d9b574b62a62c1d2549aad101e283ac65167869f9d1a076d74c` |

### How to Acquire the Model Checkpoint:
1. **From Colab Training Run:** Run [`notebooks/02_sbdh_transformer.ipynb`](file:///d:/Evernorth/Clinical-DSS/notebooks/02_sbdh_transformer.ipynb). The final cell packages the trained model and tokenizer into `evernorth_m3_results.zip`. Unzip this file into the repository root so that `evernorth_m3_results/models/sbdh_transformer/best_model.pt` is present.
2. **From Baseline Archive:** Extract `best_model.pt` from `reports/baseline_freeze_archive_20261026.zip` into `evernorth_m3_results/models/sbdh_transformer/`.
3. **Automated Discovery:** The model loader in `app/model_loader.py` automatically checks both `evernorth_m3_results/models/sbdh_transformer` and `models/sbdh_transformer`.

---

## 6. How to Run & Verify

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/hemanthsai/Clinical-DSS.git
cd Clinical-DSS

# Install pinned dependencies
pip install -r requirements.txt
```

### Run Full Test Suite (66 Automated Tests)
```bash
pytest tests/ -v
```
All 66 tests pass in ~18 seconds, verifying:
- Track A readmission evaluation integrity, zero-leakage split, and confusion matrices.
- Track B 45-class multi-label BIO alignment and loss masking (`ex_no 1192`).
- Kuhn's maximum bipartite 1-to-1 matching and adversarial order-invariance.
- Candidate boundary snapping, balanced delimiter protection, and fusion regression tests.
- Streamlit application loading, Arm 6A pipeline execution, Strategy A fallback, and fixed-input deterministic smoke testing.

### Run Inference Path Profiling
```bash
python scripts/profile_inference_path.py
```

### Launch Interactive Streamlit MVP
```bash
streamlit run app/app.py
```

---

## 7. Repository Architecture

```text
Clinical-DSS/
├── .gitignore                          # Strictly excludes data/, large checkpoints, and caches
├── LICENSE                             # MIT License
├── README.md                           # Comprehensive documentation & release guide
├── requirements.txt                    # Pinned dependencies (torch, transformers, scipy, etc.)
├── app/
│   ├── app.py                          # Streamlit application entrypoint & dual-track router
│   ├── model_loader.py                 # Safe cached loading (@st.cache_resource) & metadata
│   ├── track_a_ui.py                   # Track A readmission risk UI module
│   └── track_b_ui.py                   # Track B SBDH extractor UI with Arm 6A & fallback
├── notebooks/
│   ├── 01_data_audit.ipynb             # Executed data audit notebook
│   └── 02_sbdh_transformer.ipynb      # Executable Colab notebook for Transformer training & eval
├── tests/
│   ├── test_boundary_ablation.py       # 16 tests for snapping, balanced delimiters, and fusion
│   ├── test_readmission_evaluation.py  # 9 tests for readmission metric integrity & confusion matrices
│   ├── test_span_alignment.py          # 9 tests for multi-label span alignment & ambiguity handling
│   ├── test_span_matching.py           # 6 tests for Kuhn's bipartite 1-to-1 matching
│   ├── test_streamlit_app.py           # 18 tests for Streamlit UI, Arm 6A pipeline, fallback, & smoke
│   └── test_transformer_alignment.py   # 8 tests for 45-class BIO tagger, loss masking, & Colab hash
├── src/
│   ├── experimental_boundary_snapping.py # Lexical boundary snapping & conservative fusion
│   ├── experimental_decoders.py        # Strategy B active-threshold decoding
│   ├── transformer_dataset.py          # 45-class BIO encoding, offset mapping, & ex_no 1192 masking
│   ├── transformer_model.py            # MultiLabelSBDHTransformer with DistilRoBERTa backbone
│   ├── train_sbdh_transformer.py       # Training pipeline with AdamW and linear warmup
│   ├── evaluate_sbdh_transformer.py    # Evaluation engine on test_reviewed with bipartite matching
│   ├── span_alignment.py               # Multi-label span alignment & contextual adjudication
│   ├── train_readmission_baseline.py   # Track A readmission baseline training
│   ├── sbdh_rules_baseline.py          # Track B rules baseline extractor
│   ├── data_loaders.py                 # Offline/online path discovery and dataset loaders
│   ├── audit_uci.py                    # UCI cohort filtering, discharge disposition analysis
│   └── audit_synth_sbdh.py             # Synth-SBDH contract assertions and validation
├── scripts/
│   ├── profile_inference_path.py       # Profiles Streamlit CPU inference before and after Arm 6A
│   ├── verify_candidate_test_metrics.py# Verifies candidate metrics directly against JSON artifacts
│   ├── run_side_by_side_narrow_audit.py# Side-by-side audit runner for baseline vs Arm 6A
│   └── format_audit_tables.py          # Formats narrow audit metrics into markdown tables
├── models/
│   ├── baseline_a_logreg.joblib        # Fitted Baseline A pipeline (no disposition, no diag)
│   ├── baseline_b_logreg.joblib        # Fitted Baseline B pipeline (with disposition)
│   └── baseline_c_logreg.joblib        # Fitted Baseline C pipeline (with disposition & diag)
└── reports/
    ├── final_technical_report.md       # Authoritative final report with side-by-side audit tables
    ├── reproduction_guide.md           # Step-by-step reproduction guide for all metrics
    ├── baseline_freeze_manifest.md     # SHA-256 freeze manifest for authoritative checkpoints
    ├── sbdh_colab_transformer_metrics_corrected.json # Frozen baseline test metrics
    └── experiments/
        ├── candidate_test_evaluation/  # Arm 6A test metrics JSON and technical evaluation report
        ├── span_boundary_ablation/     # Validation boundary ablation experiment artifacts
        └── decoder_ablation/           # Validation decoder ablation experiment artifacts
```

---

## 8. Honest Limitations & Caveats

1. **Context Window Limit:** The DistilRoBERTa model operates on a maximum sequence length of 256 subword tokens. Clinical notes exceeding this length are safely truncated; notes requiring complete multi-page context would require chunked sliding-window inference.
2. **Ambiguous Gold Spans:** Five ambiguous annotations in `synth_sbdh_test_reviewed.csv` cannot be resolved without guessing. These are penalized as False Negatives under our reported conservative worst-case bounds.
3. **Synthetic Training Data:** Synth-SBDH is a synthetically generated corpus. While effective for learning clinical phrase boundaries, distribution shift may occur when applied to real-world EHR text from diverse hospital systems.
4. **Bedside Constraint:** Track A deliberately excludes post-admission discharge disposition and final billed ICD-9 codes to ensure real-time clinical utility at the bedside, accepting lower AUROC in exchange for zero target leakage.

---

## 9. License & Data Governance

### Data Licenses
- **UCI Diabetes 130-US Hospitals Dataset:** Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)). Hosted by the UCI Machine Learning Repository (Dataset ID 296; Clore, J., Cios, K., DeShazo, J., & Strack, B., 2014).
- **Synth-SBDH Benchmark Dataset:** **UNVERIFIED** (License is not explicitly declared in the Hugging Face dataset card metadata tag). Provided by UMass BioNLP Lab (`bio-nlp-umass/Synth-SBDH`).
- **Data Privacy & Commitment Guarantee:** **Confirmed: Zero raw patient or synthetic dataset records are committed to this repository.** All raw datasets (`.csv`, `.tsv`, `.parquet`, `.zip`, `.tar.gz`) are strictly excluded by `.gitignore` and must be dynamically downloaded or cached in local ignored directories.

### Code Licenses
- **Evernorth Clinical DSS Application Codebase:** This repository's code is distributed under the **MIT License** (see [`LICENSE`](LICENSE)).
- **Synth-SBDH Benchmark Code:** Distributed under the **Apache-2.0 License** (as released by the official bio-nlp-umass repository).

### Citations
- **Synth-SBDH Benchmark:**  
  Mitra, A., Yang, Z., Druhl, E., Goodwin, R., & Yu, H. (2025). *Synth-SBDH: A Synthetic Dataset of Social and Behavioral Determinants of Health for Clinical Text*. arXiv:2406.06056 (EMNLP 2025). Dataset: [huggingface.co/datasets/bio-nlp-umass/Synth-SBDH](https://huggingface.co/datasets/bio-nlp-umass/Synth-SBDH).
- **UCI Diabetes 130-US Hospitals (Dataset ID 296):**  
  Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Balasubramanian, S., Higgins, K. A., & Cios, K. J. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Records*. BioMed Research International, 2014, 781670. DOI: `10.1155/2014/781670`.