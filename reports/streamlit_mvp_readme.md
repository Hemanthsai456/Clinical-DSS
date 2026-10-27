# Evernorth Clinical Decision Support (CDS) — Streamlit Research Prototype

**Workspace:** `D:\Evernorth\Clinical-DSS`  
**Version:** 1.0.0 (Research & Educational MVP)  
**Authoritative Checkpoint Commit:** `7dd92b5bda82042f5009af52ab7013223df567b4`  
**Status:** Frozen Baseline Demonstration  

---

> [!CAUTION]
> **REGULATORY & NON-CLINICAL USE DISCLAIMER:**  
> This application is strictly an **educational and research prototype**.
> - **NOT A MEDICAL DEVICE:** Has not been cleared or approved by the FDA or any regulatory body.
> - **NO CLINICAL ADVICE:** Must never be used for diagnosis, clinical triage, medical decision-making, patient discharge authorization, or medication management.
> - **NO REAL PATIENT DATA:** Real protected health information (PHI) must never be entered into this application.
> - **LOCAL EXECUTION:** All inference is executed 100% locally on CPU without external API calls or telemetry.

---

## 1. Quick Start & Launch Instructions

### Prerequisites
- Python 3.10 or 3.11 (tested on Python 3.11.3 Windows)
- Modern web browser (Chrome, Firefox, Edge, Safari)

### Step 1: Install Dependencies
From the repository root (`D:\Evernorth\Clinical-DSS`), install the required dependencies:
```powershell
pip install -r requirements.txt
```

### Step 2: Launch Streamlit Application
Launch the app on the default local port (`8501`):
```powershell
streamlit run app/app.py
```

To run in headless mode or on a specific port:
```powershell
streamlit run app/app.py --server.headless true --server.port 8501
```

Once started, open your web browser and navigate to:
```
http://localhost:8501
```

---

## 2. Application Architecture & Pages

The application is structured into four modular views accessible from the sidebar navigation:

### Page 1: Track A — Inpatient 30-Day Readmission Risk
- **Model:** Primary Bedside Model (**Baseline A**; [`models/baseline_a_logreg.joblib`](../models/baseline_a_logreg.joblib)).
- **Input Constraints:** Strictly restricted to 40 clinical features legitimately accessible during discharge planning (8 numerical inpatient intensity/utilization features and 32 categorical admission, demographic, and glycemic/medication features).
- **Leakage Isolation:** Completely excludes post-acute placement (`discharge_disposition_id`) and retrospective billing ICD-9 codes (`diag_1`, `diag_2`, `diag_3`).
- **Features & Presets:**
  - **High-Risk Profile:** Prior frequent hospitalizations, elderly, insulin titration.
  - **Moderate-Risk Profile:** Average inpatient stay, single prior admission, oral antidiabetic agents.
  - **Low-Risk Profile:** Young elective admission, stable glycemic control, short length of stay.
  - **Customizable Inpatient Inputs:** Interactive sliders and dropdowns for all 40 variables.
- **Output:** Calibrated readmission probability, frozen decision threshold ($\tau = 0.130$), research risk tier (`Standard Transition` vs. `Elevated Care-Coordination Risk`), and explicit clinical advisories.

### Page 2: Track B — SBDH Narrative Text Extraction
- **Model:** 3-Epoch Colab GPU Transformer Checkpoint (local-only artifact, not committed to the repository).
- **Architecture:** `distilbert/distilroberta-base` with 45 independent binary sigmoid logits (15 SBDH categories $\times$ $\{B, I, O\}$).
- **Multi-Label Capability:** Correctly identifies multiple co-occurring social determinants mapped to identical character intervals (e.g., `"cannot afford physical therapy"` tagged as both `financial insecurity` and `barriers to care`).
- **Privacy & Security:** 100% in-memory local CPU inference. Clinical text is never transmitted over external networks.
- **Features & Controls:**
  - Presets demonstrating multi-label spans, housing/food insecurity, and social isolation.
  - Custom narrative text box with word/token counter.
  - Decision threshold slider (default frozen $\theta = 0.30$, range $0.10$ to $0.90$).
  - Sequence truncation warning if text exceeds 256 tokens.
  - Visual color-coded entity highlight view.
  - Structured span table with character offsets, category tags, and extracted snippet text.

### Page 3: Model Provenance, Checkpoint Inventory & Metrics
- Detailed side-by-side comparison of Track A (AUROC 0.6324, PR-AUC 0.1925, Brier 0.0961, ECE 0.0061) and Track B (Relaxed F1 0.8330, Exact F1 0.5105, Note Micro F1 0.8888).
- Authoritative baseline progression tables comparing Baselines A, B, and C, and the Rules baseline vs. Transformer.
- Verification status, artifact paths, and SHA256 checksums.

### Page 4: System Scope, Limitations & Ethical Considerations
- Clinical CDS setting and boundaries.
- Historical data distribution shift disclosures (UCI Diabetes collected 1999–2008).
- Regional readmission censoring nuance.
- Synthetic narrative text nuances (Synth-SBDH).
- Algorithmic fairness and care coordination resource allocation considerations.

---

## 3. Directory Layout

```text
D:\Evernorth\Clinical-DSS\
├── app/
│   ├── app.py                         # Main Streamlit application entry point
│   ├── model_loader.py                # Cached artifact loaders & metadata
│   ├── track_a_ui.py                  # Readmission risk estimation UI & presets
│   └── track_b_ui.py                  # SBDH text extraction UI & highlighter
├── models/
│   └── baseline_a_logreg.joblib       # Frozen Baseline A pipeline
├── evernorth_m3_results/
│   └── models/sbdh_transformer/
│       ├── best_model.pt              # Authoritative 3-epoch transformer weights
│       ├── config.json                # Transformer configuration
│       └── tokenizer_config.json      # FastTokenizer configuration
├── reports/
│   ├── final_technical_report.md      # Comprehensive 14-section project report
│   ├── baseline_freeze_manifest.md    # Checksum inventory of all frozen artifacts
│   ├── reproduction_guide.md          # Step-by-step reproduction instructions
│   └── streamlit_mvp_readme.md        # This application README
└── tests/
    └── test_streamlit_app.py          # Automated tests for app inference & schema
```

---

## 4. Verification & Automated Testing

To run the automated tests validating the Streamlit application inference, input schemas, and artifact preservation:
```powershell
pytest -v tests/test_streamlit_app.py
```

To run the entire test suite across both Track A and Track B:
```powershell
pytest -v
```

---

## 5. Troubleshooting & FAQ

1. **Why does Track A not include diagnosis codes or discharge disposition?**  
   In a real-time clinical discharge-planning scenario, official ICD billing codes are assigned days *after* patient discharge by medical records coders. Including them introduces lookahead coding bias. Post-acute disposition (SNF/Home Health) introduces circular placement leakage. Baseline A strictly uses pre-admission and inpatient features known before physical departure.

2. **Does Track B transmit text to cloud APIs (OpenAI, HuggingFace, etc.)?**  
   No. Inference runs entirely locally using `torch` and the local frozen weights in `evernorth_m3_results/models/sbdh_transformer/best_model.pt`. No external network calls are made.

3. **What happens if user text exceeds 256 tokens in Track B?**  
   The application displays a warning notification. The `distilroberta` tokenizer safely truncates the input to 256 tokens (`truncation=True`) and extracts spans from the first 256 tokens without crashing.
