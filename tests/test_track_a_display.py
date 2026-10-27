"""
Unit tests for Track A dynamic risk panel display and metric computation.
Verifies:
1. Dynamic loading of frozen metrics from reports/readmission_baselines_metrics.json.
2. Exact formula computation of population plain statement:
   "On the frozen test set, about {round((1 - precision) * 10)} of every 10 patients flagged at tau = {tau:.3f} were not readmitted within 30 days (precision {precision:.2%})."
3. Verification that changing a copy of the metrics file in a temporary directory changes the statement.
4. Relative risk computation (p / base_rate where base_rate = 0.1111).
5. Exact classification boundary (STANDARD below tau, ELEVATED at or above tau).
6. Clear error raising on missing file or missing keys.
7. Validation threshold alignment with TRACK_A_FROZEN_THRESHOLD.
"""
import os
import sys
import json
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
APP_DIR = os.path.join(PROJECT_ROOT, "app")
if APP_DIR not in sys.path:
    sys.path.append(APP_DIR)

from app.model_loader import (
    load_track_a_metrics,
    compute_track_a_plain_statement,
    compute_relative_risk,
    classify_readmission_risk,
    TRACK_A_FROZEN_THRESHOLD,
    TRACK_A_METRICS_PATH,
)


def test_track_a_metrics_source_of_truth_values():
    """Verify that load_track_a_metrics extracts exact frozen baseline values from reports/."""
    metrics = load_track_a_metrics()
    assert metrics["base_rate"] == 0.1111
    assert metrics["precision"] == 0.1806
    assert metrics["recall"] == 0.38
    assert metrics["specificity"] == 0.7846
    assert metrics["auroc"] == 0.6324
    assert metrics["val_threshold"] == 0.13
    assert abs(metrics["val_threshold"] - TRACK_A_FROZEN_THRESHOLD) < 1e-4


def test_track_a_plain_statement_formula_exact():
    """
    Verify computed plain statement matches required population wording and formula:
    'On the frozen test set, about {round((1 - precision) * 10)} of every 10 patients flagged at tau = {tau:.3f} were not readmitted within 30 days (precision {precision:.2%}).'
    """
    metrics = load_track_a_metrics()
    precision = metrics["precision"]  # 0.1806
    tau = TRACK_A_FROZEN_THRESHOLD     # 0.130

    statement = compute_track_a_plain_statement(precision=precision, tau=tau)
    expected_unadmitted = round((1 - precision) * 10)  # round(0.8194 * 10) = 8
    assert expected_unadmitted == 8

    expected_statement = (
        "On the frozen test set, about 8 of every 10 patients flagged at tau = 0.130 "
        "were not readmitted within 30 days (precision 18.06%)."
    )
    assert statement == expected_statement


def test_track_a_plain_statement_changes_with_temp_metrics_file(tmp_path):
    """
    Verify that copying the metrics file to a temporary directory and modifying
    the precision value causes compute_track_a_plain_statement to produce a changed sentence.
    """
    with open(TRACK_A_METRICS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Modify precision from 0.1806 to 0.4000
    data["sensitivity_experiments"]["Baseline_A_No_Disposition_No_Diag"]["test"]["precision"] = 0.40

    temp_metrics_file = tmp_path / "temp_readmission_metrics.json"
    with open(temp_metrics_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    loaded = load_track_a_metrics(metrics_path=str(temp_metrics_file))
    assert loaded["precision"] == 0.40

    statement = compute_track_a_plain_statement(precision=loaded["precision"], tau=TRACK_A_FROZEN_THRESHOLD)
    # round((1 - 0.40) * 10) = round(6.0) = 6
    expected_statement = (
        "On the frozen test set, about 6 of every 10 patients flagged at tau = 0.130 "
        "were not readmitted within 30 days (precision 40.00%)."
    )
    assert statement == expected_statement
    assert "about 6 of every 10" in statement
    assert "precision 40.00%" in statement


def test_track_a_relative_risk_calculation():
    """
    Verify relative risk calculation: probability / base_rate.
    For probability 9.62% (0.0962) and base rate 0.1111:
    0.0962 / 0.1111 = 0.86588... -> 0.87x.
    """
    base_rate = 0.1111

    # Preset 3 / Low-risk check: 9.62% -> 0.87x
    prob_low = 0.0962
    rr_low = compute_relative_risk(prob_low, base_rate)
    assert abs(rr_low - (prob_low / base_rate)) < 1e-9
    assert f"{rr_low:.2f}x" == "0.87x"

    # Exactly at threshold tau = 0.130: 0.130 / 0.1111 -> 1.17x
    prob_tau = 0.130
    rr_tau = compute_relative_risk(prob_tau, base_rate)
    assert f"{rr_tau:.2f}x" == "1.17x"

    # Zero risk
    assert compute_relative_risk(0.0, base_rate) == 0.0

    # Probability equals base rate -> 1.00x
    assert compute_relative_risk(base_rate, base_rate) == 1.0


def test_track_a_relative_risk_invalid_base_rate():
    """Verify that zero or negative base rate raises ValueError."""
    with pytest.raises(ValueError, match="base_rate must be positive"):
        compute_relative_risk(0.15, 0.0)

    with pytest.raises(ValueError, match="base_rate must be positive"):
        compute_relative_risk(0.15, -0.1111)


def test_track_a_risk_classification_boundary():
    """
    Verify decision threshold classification:
    - Probability strictly below tau: STANDARD RISK
    - Probability at or above tau: ELEVATED RISK
    """
    tau = TRACK_A_FROZEN_THRESHOLD  # 0.130

    # Just below tau
    assert classify_readmission_risk(tau - 1e-6, tau) == "STANDARD"
    assert classify_readmission_risk(tau - 0.01, tau) == "STANDARD"
    assert classify_readmission_risk(0.0962, tau) == "STANDARD"
    assert classify_readmission_risk(0.1299, tau) == "STANDARD"

    # Exactly at tau
    assert classify_readmission_risk(tau, tau) == "ELEVATED"

    # Just above tau
    assert classify_readmission_risk(tau + 1e-6, tau) == "ELEVATED"
    assert classify_readmission_risk(0.131, tau) == "ELEVATED"
    assert classify_readmission_risk(0.25, tau) == "ELEVATED"


def test_track_a_missing_file_raises_error(tmp_path):
    """Verify FileNotFoundError is raised with clear message if metrics file does not exist."""
    missing_path = str(tmp_path / "non_existent_metrics.json")
    with pytest.raises(FileNotFoundError, match="Track A frozen metrics file not found"):
        load_track_a_metrics(metrics_path=missing_path)


def test_track_a_missing_key_raises_error(tmp_path):
    """Verify KeyError is raised with clear message if any required key is missing."""
    with open(TRACK_A_METRICS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Missing majority_class_baseline
    bad_data_1 = {k: v for k, v in data.items() if k != "majority_class_baseline"}
    file_1 = tmp_path / "bad_1.json"
    with open(file_1, "w", encoding="utf-8") as f:
        json.dump(bad_data_1, f)
    with pytest.raises(KeyError, match="missing required key"):
        load_track_a_metrics(metrics_path=str(file_1))

    # 2. Missing precision in test metrics
    bad_data_2 = json.loads(json.dumps(data))
    del bad_data_2["sensitivity_experiments"]["Baseline_A_No_Disposition_No_Diag"]["test"]["precision"]
    file_2 = tmp_path / "bad_2.json"
    with open(file_2, "w", encoding="utf-8") as f:
        json.dump(bad_data_2, f)
    with pytest.raises(KeyError, match="missing required key: 'precision'"):
        load_track_a_metrics(metrics_path=str(file_2))


def test_track_a_corrupted_json_raises_error(tmp_path):
    """Verify ValueError is raised if the metrics file contains malformed JSON."""
    bad_json_file = tmp_path / "corrupted.json"
    bad_json_file.write_text("{ this is not valid json", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON in metrics file"):
        load_track_a_metrics(metrics_path=str(bad_json_file))
