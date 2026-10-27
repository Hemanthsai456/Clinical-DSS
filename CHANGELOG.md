# Changelog

All notable changes made to the Evernorth Clinical Decision Support (CDS) prototype to ensure mathematical correctness, methodological transparency, honest error disclosure, and supervisor defensibility.

---

## Session status (2026-10-27)
- **Git status audit:**
  - Modified working tree files: `README.md`, `app/app.py`, `app/model_loader.py`, `app/track_a_ui.py`, `app/track_b_ui.py`, `reports/final_technical_report.md`, `reports/sbdh_error_analysis_experiment_1.md`, `scripts/run_span_boundary_ablation.py`, `src/experimental_boundary_snapping.py`, `tests/test_streamlit_app.py`.
  - Untracked files: `CHANGELOG.md`, `tests/test_track_a_display.py`.
- **Implementation Status:**
  - **Step 1 (Track A Risk Panel): COMPLETED**
    - Metrics loaded once dynamically from `reports/readmission_baselines_metrics.json` via helper `load_track_a_metrics()`. Fails with clear `FileNotFoundError` or `KeyError` if file or keys are absent. Zero hardcoded fallbacks.
    - Plain statement computed as a population fact about the flagged group: *"On the frozen test set, about 8 of every 10 patients flagged at tau = 0.130 were not readmitted within 30 days (precision 18.06%)."* Placed in separate clearly labelled `Population Operating Context` alert block, distinct from patient prediction.
    - Base rate dynamically loaded as 0.1111 (displayed as 11.11%).
    - Relative risk computed dynamically as `probability / base_rate` (yields 0.87x for 9.62% input).
    - "Model Operating Range" line dynamically populated with test recall (38.00%), precision (18.06%), specificity (78.46%), AUROC (0.6324).
    - Threshold labeled as validation-tuned ($F_1$) and test metrics kept separate.
    - "Transitional Care Trigger" labeled as an illustrative hypothetical workflow that has not been clinically evaluated.
  - **Step 2 (Track B Known Limitations): COMPLETED**
    - Added visible note directly beneath Track B preset selector: *"Known limitations on these examples: 'life' and 'in a rented house' are false positives; 'access to healthcare' is a truncated span (should read 'poor access to healthcare'). See the Track B error analysis report ([reports/sbdh_error_analysis_experiment_1.md](file:///d:/Evernorth/Clinical-DSS/reports/sbdh_error_analysis_experiment_1.md))."*
    - Decoder pipeline (Arm 6A) preserved without modification.
  - **Step 3 (Unit Tests): COMPLETED**
    - Added comprehensive unit test suite in `tests/test_track_a_display.py` (9 tests):
      1. `test_track_a_metrics_source_of_truth_values`: verifies source-of-truth JSON values.
      2. `test_track_a_plain_statement_formula_exact`: verifies exact formula `round((1 - precision) * 10)`.
      3. `test_track_a_plain_statement_changes_with_temp_metrics_file`: verifies that changing a copy of metrics in a temp directory updates the computed statement.
      4. `test_track_a_relative_risk_calculation`: verifies `prob / 0.1111` (0.0962 -> 0.87x).
      5. `test_track_a_relative_risk_invalid_base_rate`: verifies `ValueError` on non-positive base rate.
      6. `test_track_a_risk_classification_boundary`: verifies STANDARD strictly below tau and ELEVATED at or above tau.
      7. `test_track_a_missing_file_raises_error`: verifies clear `FileNotFoundError`.
      8. `test_track_a_missing_key_raises_error`: verifies clear `KeyError`.
      9. `test_track_a_corrupted_json_raises_error`: verifies clear `ValueError`.
  - **Step 4 (Verification & App Testing): COMPLETED**
    - Verified Streamlit application at `http://localhost:8501`.
    - Computed plain statement displayed accurately: *"On the frozen test set, about 8 of every 10 patients flagged at tau = 0.130 were not readmitted within 30 days (precision 18.06%)."*
    - Relative risk for 9.62% test input verified as `0.0962 / 0.1111 = 0.87x`.
    - Low-risk preset verified: Predicted risk 9.11%, population base rate 11.11%, relative risk 0.82x, classification STANDARD RISK.
    - Model Operating Range block verified with dynamic values: Recall 38.00%, Precision 18.06%, Specificity 78.46%, AUROC 0.6324.
    - High-risk preset verified: Predicted risk 23.21% (> 13%), relative risk 2.09x, classification ELEVATED RISK.
    - Track B preset selector verified: Visible note displayed linking to `reports/sbdh_error_analysis_experiment_1.md`.
    - Screenshots captured in artifact directory:
      - `track_a_low_risk.png`
      - `track_a_high_risk.png`
      - `track_a_model_operating_range.png`
      - `track_b_preset_note.png`
  - **Dependency & Governance Updates:**
    - Pinned Streamlit in `requirements.txt` to `streamlit==1.56.0` to support modern container width parameters (`width="stretch"`).
    - Verified license documentation in `README.md`: codebase is licensed under MIT, Synth-SBDH code is licensed under Apache-2.0, and Synth-SBDH dataset license remains explicitly marked UNVERIFIED.
    - Corrected Arm 7A vs. Arm 7B validation ablation figures in `CHANGELOG.md`: Arm 7A Exact gain is +0.0001 and Relaxed drop is −0.0012; Arm 7B Exact gain is +0.0004 and Relaxed drop is −0.0007.
- **Test Suite Results:**
  - Before changes: 69 passed, 0 failed.
  - After changes: **78 passed**, 0 failed in 33.31s. Zero failures.

---

## [Unreleased] - 2026-10-26

### Track A: Inpatient 30-Day Readmission Risk (`app/track_a_ui.py`)
- **Fixed Admission-Type and Demographic Selectbox Defaults:**
  - Resolved preset selection bug where `admission_type_id` index was computed as `0 if value == "1" else 1`, causing the Low-Risk preset (`"3" - Elective`) to mis-select `"2 - Urgent"`.
  - Replaced index lookups with explicit key-based indexing: `list(adm_types.keys()).index(value)`.
  - Implemented the identical safe key-based indexing for `gender`, `race`, and `admission_source_id`.
  - Added unit test `test_track_a_low_risk_preset_admission_type_id` in `tests/test_streamlit_app.py` verifying that the Low-Risk preset submits `admission_type_id "3"`.
- **Exposed All 23 Inpatient Diabetes Medications:**
  - Added an interactive expander exposing all 23 diabetes medications (`metformin`, `repaglinide`, `nateglinide`, `chlorpropamide`, `glimepiride`, `acetohexamide`, `glipizide`, `glyburide`, `tolbutamide`, `pioglitazone`, `rosiglitazone`, `acarbose`, `miglitol`, `troglitazone`, `tolazamide`, `examide`, `citoglipton`, `insulin`, `glyburide-metformin`, `glipizide-metformin`, `glimepiride-pioglitazone`, `metformin-rosiglitazone`, `metformin-pioglitazone`).
  - Labeled defaults explicitly as `(Default: No)` to eliminate silent hardcoded feature assumptions while preserving full compatibility with presets.
- **Risk Stratification & Honest Clinical Communication:**
  - Visibly separated the validation-tuned decision threshold ($\tau = 0.130$) from evaluated frozen test metrics, clearly labeling the evaluation splits.
  - Displayed the population base rate ($11.11\%$) and relative risk ($\text{prob} / 0.111$).
  - Added an unambiguous plain statement: *"About 8 of every 10 flagged patients will not be readmitted within 30 days (precision 18.06% on test)."*
  - Re-labeled the progress bar explicitly as an estimated readmission probability rather than a patient safety score.
  - Labeled the "Transitional Care Trigger" description as a hypothetical example workflow that has not been prospectively or clinically evaluated.
- **Fixed LaTeX Escape Warnings:**
  - Converted f-strings containing LaTeX math symbols (`\%`, `\tau`) to raw f-strings (`rf"""..."""`), eliminating invalid escape sequence warnings.

---

### Track B: SBDH Narrative Text Extraction (`app/track_b_ui.py`, `src/`, `scripts/`)
- **Span Confidence Score Calculation:**
  - Replaced arbitrary 0.50 fallback and $B$-only confidence calculation with $\max(P_B, P_I)$ over all active subword tokens overlapping the span.
  - Labeled UI table column as `"Span score (max token probability)"`.
  - Added warning logging and `"n/a"` display if no active tokens match a span.
  - Added unit test `test_track_b_displayed_span_scores_above_threshold` verifying every extracted span displayed has score $\ge$ decision threshold (0.30).
- **Highlight Rendering Rewrite & Exact Character Preservation:**
  - Rewrote `render_highlighted_text` to split narrative text at all unique span boundary cut-points.
  - Renders each text slice exactly once, with overlapping multi-label categories rendered as stacked badges.
  - Added unit test `test_render_highlighted_text_exact_character_preservation` verifying that the rendered HTML contains each character of the input narrative exactly once without duplication or omission.
- **Boundary Snapping & Fusion Ablation (Validation-Only Experiments):**
  - Added extended punctuation blockers (`","`, `";"`, `":"`) and stopword filtering (`"life"`, `"home"`, `"stay"`, `"care"`) to `src/experimental_boundary_snapping.py` as new optional parameters, preserving Arm 6A backward compatibility.
  - Added Arm 7A and Arm 7B to `scripts/run_span_boundary_ablation.py`.
  - Evaluated on 876 validation notes (`reports/cached_val_notes.pt`):
    - **Arm 6A (Current Promoted):** Exact F1 = 0.5297, Relaxed F1 = 0.8362, Merges = 10.
    - **Arm 7A (Extended Blockers):** Exact F1 = 0.5298 (+0.0001), Relaxed F1 = 0.8350 (−0.0012), Merges = 3 (blocked 7 punctuation merges).
    - **Arm 7B (Extended Blockers + Stopword Filter):** Exact F1 = 0.5301 (+0.0004), Relaxed F1 = 0.8355 (−0.0007), Merges = 3 (filtered 2 false-positive stopword spans).
  - **Decision:** Arm 7A showed an Exact gain of only +0.0001 and a Relaxed drop of −0.0012; Arm 7B showed an Exact gain of +0.0004 and a Relaxed drop of −0.0007. Because exact F1 gains were negligible while relaxed F1 dropped in both configurations, Arm 7A and 7B were retained as validation experiments and NOT promoted to the release pipeline. Arm 6A remains the promoted release. No frozen test metrics in `reports/` were changed.
- **Dual-Protocol Metrics Reporting:**
  - Updated UI performance display to present both evaluation protocols matching `reports/final_technical_report.md`:
    - **Resolvable Annotations Protocol ($N=3,340$):** Arm 6A Exact F1 = 0.5173, Relaxed F1 = 0.8414, Note Micro F1 = 0.8949 (vs Baseline Exact F1 = 0.5105, Relaxed F1 = 0.8330).
    - **Conservative Worst-Case Bound ($N=3,345$, 5 unresolved annotations penalized as FN):** Arm 6A Exact F1 = 0.5169, Relaxed F1 = 0.8408 (vs Baseline Exact F1 = 0.5101, Relaxed F1 = 0.8324).
- **Dynamic Model Name Configuration:**
  - Added `get_track_b_base_model_name()` to `app/model_loader.py` to read architecture from `models/sbdh_transformer/config.json` (`distilbert/distilroberta-base`).
  - Aligned model name across UI and reports.
- **Privacy Text Update:**
  - Updated privacy text in Track B to: *"When run locally, no narrative is transmitted over any external network or cloud service."*
- **Streamlit Container Width Modernization:**
  - Replaced deprecated `use_container_width=True` with `width="stretch"` for `st.dataframe`.

---

### Reports and Documentation
- **Synth-SBDH Citation Update:**
  - Corrected Synth-SBDH citation across `README.md`, `reports/final_technical_report.md`, and `reports/sbdh_error_analysis_experiment_1.md` to:
    > Mitra, A., Yang, Z., Druhl, E., Goodwin, R., & Yu, H. (2025). Synth-SBDH: A Synthetic Dataset of Social and Behavioral Determinants of Health for Clinical Text. arXiv:2406.06056 (EMNLP 2025). Dataset: huggingface.co/datasets/bio-nlp-umass/Synth-SBDH.
- **Track A Patient Count Audit:**
  - Verified mathematical derivation explaining why Track A validation and test splits both contain 10,499 unique patients:
    - 69,990 eligible patients split via `GroupShuffleSplit(test_size=0.15)` produces $69,990 \times 0.15 = 10,498.5 \rightarrow 10,499$ test patients.
    - Subsequent relative validation split ($0.15 / 0.85 \approx 0.17647$) on remaining 59,491 patients produces $59,491 \times 0.17647 = 10,498.41 \rightarrow 10,499$ validation patients.
    - Confirmed that Synth-SBDH having 10,499 total records is a pure numerical coincidence between two entirely independent datasets.
- **License & Data Governance (`README.md`):**
  - Added Section 9 documenting:
    - UCI Diabetes 130-US Hospitals dataset license: Creative Commons Attribution 4.0 International (CC BY 4.0).
    - Synth-SBDH dataset license: Marked **UNVERIFIED** (not explicitly declared in Hugging Face dataset card metadata tag).
    - Synth-SBDH code license: Apache-2.0 License.
    - Application codebase license: MIT License.
    - Verified and confirmed that zero raw dataset files are committed to Git.

---

### Test Suite Verification
- **Test Results:**
  - Before changes: 66 passed.
  - After changes: **69 passed**, 0 failed in 18.79s.
  - Added tests:
    1. `test_track_a_low_risk_preset_admission_type_id`: verifies Low-Risk preset submits `admission_type_id "3"`.
    2. `test_track_b_displayed_span_scores_above_threshold`: verifies every displayed span score is $\ge$ decision threshold.
    3. `test_render_highlighted_text_exact_character_preservation`: verifies character-for-character preservation without duplication.
