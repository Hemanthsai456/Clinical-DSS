"""
Automated test suite for SBDH span alignment and validation gate.
Verifies:
1. ex_no 7411: Contextual disambiguation of repeated 'home' (avoids naive first-substring matching).
2. ex_no 7062: Preservation of multiple SBDH categories on identical text spans.
3. ex_no 7611: Contextual disambiguation of repeated 'pain' (current vs history).
4. Multi-label downstream token representation conversion (zero label loss).
5. Case-only differences and missing span handling.
6. Ambiguous repeated mentions flagging.
"""
import pytest
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.span_alignment import (
    align_spans_in_note,
    find_all_occurrences,
    convert_to_multi_label_token_annotations,
)


def test_ex_no_7411_repeated_home_disambiguation():
    """
    ex_no 7411:
    Text: 'Patient recently moved from a nursing home back to her own home.'
    Spans: 'recently moved from a nursing home back to her own home' (transitions of care)
           'home' (housing insecurity, presence=no)
    The aligner must NOT naively select the first occurrence of 'home' in 'nursing home' [38:42].
    It must identify candidate 2 'her own home' [59:63] as the intended clinical anchor.
    """
    text = "Patient recently moved from a nursing home back to her own home."
    spans = ["recently moved from a nursing home back to her own home", "home"]
    sbdhs = ["transitions of care", "housing insecurity"]
    presences = ["yes", "no"]
    periods = ["current", "current"]

    # Verify both occurrences exist in text
    occ = find_all_occurrences(text, "home")
    assert len(occ) == 2, f"Expected 2 occurrences of 'home', got {len(occ)}"
    assert occ[0][:2] == (38, 42)  # In 'nursing home'
    assert occ[1][:2] == (59, 63)  # In 'her own home'

    # Run contextual alignment
    result = align_spans_in_note(
        text=text,
        spans=spans,
        sbdhs=sbdhs,
        presences=presences,
        periods=periods,
        ex_no=7411,
        disambiguate_context=True,
    )

    resolved = result["resolved_intervals"]
    home_interval = next((r for r in resolved if r["annotated_span"] == "home"), None)

    assert home_interval is not None, "Interval for 'home' was not resolved!"
    # Must NOT be first occurrence (38, 42)
    assert (home_interval["start_char"], home_interval["end_char"]) != (38, 42), \
        "Aligner naively selected first substring match in 'nursing home'!"
    # Must be second occurrence (59, 63)
    assert (home_interval["start_char"], home_interval["end_char"]) == (59, 63), \
        f"Expected (59, 63), got ({home_interval['start_char']}, {home_interval['end_char']})"
    assert home_interval["categories"] == ["housing insecurity"]
    assert home_interval["status"] == "RESOLVED_CONTEXTUAL_ADJUDICATION"


def test_ex_no_7062_identical_span_multiple_labels():
    """
    ex_no 7062:
    Text: 'Patient suffers from chronic pain and cannot afford to continue physical therapy.'
    Phrase 'cannot afford to continue physical therapy' is annotated with both:
    1. 'financial insecurity'
    2. 'barriers to care'
    Both labels must be preserved on the exact same character interval [38, 80].
    """
    text = "Patient suffers from chronic pain and cannot afford to continue physical therapy."
    spans = [
        "suffers from chronic pain",
        "cannot afford to continue physical therapy",
        "cannot afford to continue physical therapy"
    ]
    sbdhs = ["pain", "financial insecurity", "barriers to care"]
    presences = ["yes", "yes", "yes"]
    periods = ["current", "current", "current"]

    result = align_spans_in_note(
        text=text,
        spans=spans,
        sbdhs=sbdhs,
        presences=presences,
        periods=periods,
        ex_no=7062,
    )

    resolved = result["resolved_intervals"]
    afford_interval = next(
        (r for r in resolved if r["annotated_span"] == "cannot afford to continue physical therapy"),
        None
    )

    assert afford_interval is not None, "Interval for 'cannot afford...' was not resolved!"
    assert (afford_interval["start_char"], afford_interval["end_char"]) == (38, 80)
    assert afford_interval["is_multi_label"] is True
    assert set(afford_interval["categories"]) == {"financial insecurity", "barriers to care"}
    assert len(afford_interval["labels"]) == 2

    # Verify downstream token conversion preserves both categories
    tokens = convert_to_multi_label_token_annotations(text, resolved)
    afford_tokens = [t for t in tokens if t["start_char"] >= 38 and t["end_char"] <= 80]
    assert len(afford_tokens) > 0

    for tok in afford_tokens:
        assert "financial insecurity" in tok["labels"], f"Token '{tok['token']}' missing financial insecurity!"
        assert "barriers to care" in tok["labels"], f"Token '{tok['token']}' missing barriers to care!"


def test_ex_no_7611_repeated_pain_contexts():
    """
    ex_no 7611:
    Text: 'Patient experiencing abdominal pain and has a documented history of opioid usage for pain management.'
    Annotations:
    1. 'pain' (current)
    2. 'pain' (history)
    Verify alignment does not infer occurrence order without evidence.
    First 'pain' [31:35] aligns to current symptom ('abdominal pain').
    Second 'pain' [85:89] aligns to historical usage ('pain management').
    """
    text = "Patient experiencing abdominal pain and has a documented history of opioid usage for pain management."
    spans = ["pain", "pain"]
    sbdhs = ["pain", "pain"]
    presences = ["yes", "yes"]
    periods = ["current", "history"]

    result = align_spans_in_note(
        text=text,
        spans=spans,
        sbdhs=sbdhs,
        presences=presences,
        periods=periods,
        ex_no=7611,
        disambiguate_context=True,
    )

    resolved = result["resolved_intervals"]
    assert len(resolved) == 2, f"Expected 2 resolved intervals for 'pain', got {len(resolved)}"

    current_interval = next((r for r in resolved if r["labels"][0]["period"] == "current"), None)
    history_interval = next((r for r in resolved if r["labels"][0]["period"] == "history"), None)

    assert current_interval is not None
    assert history_interval is not None
    assert (current_interval["start_char"], current_interval["end_char"]) == (31, 35)
    assert (history_interval["start_char"], history_interval["end_char"]) == (85, 89)


def test_case_only_differences_and_missing_spans():
    """
    Test case-only differences (expert lowercasing) and ungrounded missing spans.
    """
    text = "Patient has severe PTSD and reports food insecurity."
    spans = ["ptsd", "food insecurity", "nonexistent span"]
    sbdhs = ["psychiatric disorders", "food insecurity", "barriers to care"]
    presences = ["yes", "yes", "yes"]
    periods = ["current", "current", "current"]

    result = align_spans_in_note(
        text=text,
        spans=spans,
        sbdhs=sbdhs,
        presences=presences,
        periods=periods,
        ex_no=9999,
    )

    resolved = result["resolved_intervals"]
    ptsd_int = next((r for r in resolved if r["annotated_span"] == "ptsd"), None)
    food_int = next((r for r in resolved if r["annotated_span"] == "food insecurity"), None)

    assert ptsd_int is not None
    assert ptsd_int["match_type"] == "case_insensitive"
    assert ptsd_int["span_text"] == "PTSD"
    assert (ptsd_int["start_char"], ptsd_int["end_char"]) == (19, 23)

    assert food_int is not None
    assert food_int["match_type"] == "exact"
    assert (food_int["start_char"], food_int["end_char"]) == (36, 51)

    # Missing span must be flagged as unmatched
    assert len(result["unmatched_cases"]) == 1
    assert result["unmatched_cases"][0]["span"] == "nonexistent span"


def test_ambiguous_repeated_mentions_flagging():
    """
    Test that when an ambiguous repeated span occurs without contextual disambiguation,
    it is flagged for review rather than silently picking an arbitrary occurrence.
    """
    text = "The doctor noted pain yesterday and pain today."
    spans = ["pain"]
    sbdhs = ["pain"]
    presences = ["yes"]
    periods = ["current"]

    # Run with disambiguate_context=False (or when no specific rule applies)
    result = align_spans_in_note(
        text=text,
        spans=spans,
        sbdhs=sbdhs,
        presences=presences,
        periods=periods,
        ex_no=8888,
        disambiguate_context=False,
    )

    # Must be flagged as ambiguous
    assert len(result["ambiguous_cases"]) == 1
    ambig = result["ambiguous_cases"][0]
    assert ambig["num_candidates"] == 2
    assert len(ambig["candidates"]) == 2
    assert "multiple alignments remain plausible" in ambig["reason"]
