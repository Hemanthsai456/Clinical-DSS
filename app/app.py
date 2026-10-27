"""
Main Streamlit Application: Evernorth Clinical Decision Support (CDS) Research Prototype.
Investigates Machine Learning & NLP for Inpatient Readmission Stratification and SBDH Extraction.

Educational & Research Prototype Only. Not for Clinical Use.
"""
import os
import sys
import streamlit as st

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model_loader import get_artifact_metadata
from track_a_ui import render_track_a_page
from track_b_ui import render_track_b_page

st.set_page_config(
    page_title="Evernorth Clinical CDS — Research Prototype",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_provenance_page():
    st.header("📊 Model Provenance, Checkpoint Inventory & Metrics")
    st.markdown(
        """
        Both modeling tracks operate under strict baseline freezes. All model checkpoints, preprocessing 
        pipelines, decision thresholds, and evaluation protocols are audited and locked.
        """
    )

    meta = get_artifact_metadata()
    ta = meta["track_a"]
    tb = meta["track_b"]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Track A: 30-Day Readmission Risk")
        st.markdown(
            f"""
            - **Model:** {ta['model_name']}
            - **Pipeline:** `{ta['pipeline_type']}`
            - **Features:** **{ta['features_count']} Clinical Variables** (Pre-admission & Inpatient stay)
            - **Fitted Checkpoint:** `{ta['model_path']}`
            - **Calibrated Threshold:** `\\tau = {ta['threshold']:.3f}`
            - **AUROC:** **{ta['auroc']:.4f}**
            - **PR-AUC (Average Precision):** **{ta['pr_auc']:.4f}** (Prevalence: ${ta['base_rate'] * 100:.2f}\\%$)
            - **F1-Score:** **{ta['f1_score']:.4f}** (Recall: ${ta['recall']:.2%}$, Precision: ${ta['precision']:.2%}$)
            - **Brier Score:** **{ta['brier_score']:.4f}** | **ECE:** **{ta['ece']:.4f}**
            - **Training Population:** Historical 1999–2008 UCI Diabetes (99,343 encounters)
            - **Patient Isolation:** Zero patient leakage between Train, Val, and Test
            """
        )

    with col2:
        st.subheader("Track B: SBDH Text Extraction")
        st.markdown(
            f"""
            - **Model:** {tb['model_name']}
            - **Backbone Architecture:** `{tb['backbone']}` (82M parameters, 6 layers)
            - **Active Release Pipeline:** **{tb.get('active_decoder', 'Arm 6A')}**
            - **Fallback Control:** `{tb.get('fallback_decoder', 'Strategy A')}`
            - **Output Dimension:** **{tb['output_logits']} Logits** (15 categories $\\times$ 3 BIO states)
            - **Weights File:** `{tb['checkpoint_path']}`
            - **Calibrated Threshold:** `\\theta = {tb['threshold']:.2f}`
            - **Relaxed Span F1 (Resolvable):** **{tb['relaxed_f1']:.4f}** (Baseline: {tb.get('baseline_relaxed_f1', 0.8330):.4f})
            - **Exact Span F1 (Resolvable):** **{tb['exact_f1']:.4f}** (Baseline: {tb.get('baseline_exact_f1', 0.5105):.4f})
            - **Note-Level Micro F1:** **{tb['note_micro_f1']:.4f}** (Baseline: {tb.get('baseline_note_micro_f1', 0.8888):.4f})
            - **Matching Algorithm:** Maximum Bipartite Matching (Kuhn's Augmenting Path, order-invariant)
            - **Benchmark:** Synth-SBDH reviewed test partition ($N = 1,732$ notes)
            """
        )

    st.markdown("---")
    st.subheader("Authoritative Baseline Comparison")
    
    st.markdown("**Track A Baseline Progression (Test Partition, $N = 14,851$):**")
    st.table([
        {"Model Pipeline": "Majority Class Baseline", "Features": 0, "AUROC": "0.5000", "PR-AUC": "0.1111", "Threshold": "0.500", "F1": "0.0000", "Brier": "0.0988", "Role": "Statistical Floor"},
        {"Model Pipeline": "Baseline A (Bedside Model)", "Features": 40, "AUROC": "0.6324", "PR-AUC": "0.1925", "Threshold": "0.130", "F1": "0.2449", "Brier": "0.0961", "Role": "Primary Real-Time Bedside Model"},
        {"Model Pipeline": "Baseline B (+ Disposition)", "Features": 41, "AUROC": "0.6599", "PR-AUC": "0.2176", "Threshold": "0.145", "F1": "0.2667", "Brier": "0.0947", "Role": "Placement Sensitivity Benchmark"},
        {"Model Pipeline": "Baseline C (+ Billed ICD-9s)", "Features": 44, "AUROC": "0.6666", "PR-AUC": "0.2213", "Threshold": "0.135", "F1": "0.2740", "Brier": "0.0944", "Role": "Retrospective Upper Bound"},
    ])

    st.markdown("**Track B Rules vs. Transformer Progression (`synth_sbdh_test_reviewed.csv`, $N = 1,732$):**")
    st.table([
        {"Method": "Rules Baseline (Milestone 2)", "Exact F1 (Resolvable)": "0.2642", "Exact F1 (Worst-Case)": "0.2639", "Relaxed F1 (Resolvable)": "0.5163", "Relaxed F1 (Worst-Case)": "0.5158", "Note Micro F1": "0.5835"},
        {"Method": "3-Epoch Colab Transformer (Control)", "Exact F1 (Resolvable)": "0.5105", "Exact F1 (Worst-Case)": "0.5101", "Relaxed F1 (Resolvable)": "0.8330", "Relaxed F1 (Worst-Case)": "0.8324", "Note Micro F1": "0.8888"},
        {"Method": "Arm 6A Promoted Release Pipeline", "Exact F1 (Resolvable)": "0.5173", "Exact F1 (Worst-Case)": "0.5169", "Relaxed F1 (Resolvable)": "0.8414", "Relaxed F1 (Worst-Case)": "0.8408", "Note Micro F1": "0.8949"},
        {"Method": "Net Release Improvement (Arm 6A vs Control)", "Exact F1 (Resolvable)": "+0.0068", "Exact F1 (Worst-Case)": "+0.0068", "Relaxed F1 (Resolvable)": "+0.0084", "Relaxed F1 (Worst-Case)": "+0.0084", "Note Micro F1": "+0.0061"},
    ])


def render_ethics_page():
    st.header("📖 System Scope, Limitations & Ethical Considerations")
    st.markdown(
        """
        ### 1. Intended Use & Boundaries
        - **Intended Use:** Exploratory machine learning research into care-coordination risk stratification and clinical entity extraction.
        - **Contraindicated Use:** Clinical triage, emergency care, automated discharge denial, independent billing coding, unsupervised clinical alerting.
        
        ### 2. Historical & Synthetic Dataset Biases
        - **Temporal Obsolescence:** The UCI dataset reflects care from **1999 to 2008**. Outpatient management protocols, sliding-scale insulin standards, and modern therapies (SGLT2 inhibitors, GLP-1 receptor agonists) are absent.
        - **Out-of-System Censoring:** Readmissions outside the 130 consortium hospitals were not captured, undercounting true readmissions in regional networks.
        - **Synthetic Text Regularity:** Synth-SBDH notes are generated by large language models. They lack authentic EHR formatting noise, typing misspellings, fragmented bullet points, and copy-forward clutter.
        
        ### 3. Patient Privacy & Data Safeguards
        - This prototype executes all neural and statistical inferences **locally on CPU**.
        - No clinical narrative, demographic parameter, or session data is transmitted to external cloud APIs, tracking servers, or remote third parties.
        - Protected health information (PHI) must never be entered into this prototype.
        """
    )


def main():
    # Regulatory Banner
    st.error(
        "⚠️ **EDUCATIONAL & RESEARCH PROTOTYPE ONLY — NOT A MEDICAL DEVICE**  \n"
        "This application is developed solely for machine learning research and educational demonstration. "
        "It is **not** cleared by the FDA, has **not** undergone clinical trial validation, and must **never** be used for clinical diagnosis, patient management, or medical decision-making. "
        "Do **not** enter Protected Health Information (PHI)."
    )

    st.title("🏥 Evernorth Clinical Decision Support (CDS) Research Prototype")
    st.markdown(
        "*Investigating Machine Learning and NLP for Inpatient Readmission Stratification and Social Determinants Extraction.*"
    )

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    st.sidebar.markdown("**Evernorth Clinical CDS**  \n*Educational Prototype v1.0*")
    
    app_mode = st.sidebar.radio(
        "Select Functional Module:",
        [
            "Track A: Readmission Risk Estimation",
            "Track B: SBDH Text Extraction",
            "Model Provenance & Metrics",
            "Scope, Limitations & Ethics",
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Technical Artifacts:**\n"
        "- Track A Model: Baseline A LogReg (40 features)\n"
        "- Track B Pipeline: Arm 6A Promoted (Strategy B + Snap + Fuse)\n"
        "- Track B Fallback: Strategy A Baseline Control\n"
        "- Local Execution: CPU (Zero external network calls)"
    )

    if app_mode == "Track A: Readmission Risk Estimation":
        render_track_a_page()
    elif app_mode == "Track B: SBDH Text Extraction":
        render_track_b_page()
    elif app_mode == "Model Provenance & Metrics":
        render_provenance_page()
    elif app_mode == "Scope, Limitations & Ethics":
        render_ethics_page()

    # Footer
    st.markdown("---")
    f_col1, f_col2 = st.columns([3, 1])
    with f_col1:
        st.caption(
            "© 2026 Evernorth Clinical Decision Support Research Prototype. Built with Streamlit, scikit-learn, and Hugging Face Transformers. "
            "MIT Open Source License. Not for clinical diagnostic use."
        )
    with f_col2:
        st.caption("Git Commit: `7dd92b5b` (Clean Baseline)")


if __name__ == "__main__":
    main()
