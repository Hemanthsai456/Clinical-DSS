"""
Unit tests for isolated experimental boundary snapping and span fusion (Experiment 2.2).
"""
import sys
import os
sys.path.insert(0, os.path.abspath("."))
import pytest
from src.experimental_boundary_snapping import (
    snap_span_boundaries,
    fuse_same_category_spans,
)
from src.transformer_dataset import decode_predictions_to_spans as production_decode


def test_snap_span_boundaries_punctuation():
    text = "The patient reports severe migraines, and dizziness."
    # Span includes trailing comma: "severe migraines,"
    span = (20, 37, "pain")
    snapped, changed = snap_span_boundaries(span, text)
    assert changed is True
    assert snapped == (20, 36, "pain")
    assert text[snapped[0]:snapped[1]] == "severe migraines"


def test_snap_span_boundaries_dangling_preposition():
    text = "He is facing possible jail time for possession."
    # Span includes trailing preposition: "facing possible jail time for"
    span = (6, 35, "legal problems")
    snapped, changed = snap_span_boundaries(span, text)
    assert changed is True
    assert snapped == (6, 31, "legal problems")
    assert text[snapped[0]:snapped[1]] == "facing possible jail time"


def test_snap_span_boundaries_mid_word_expansion():
    text = "Patient was diagnosed with fibromyalgia yesterday."
    # Span starts mid-word 'myalgia' -> [32:39]
    span = (32, 39, "pain")
    snapped, changed = snap_span_boundaries(span, text)
    assert changed is True
    assert snapped == (27, 39, "pain")
    assert text[snapped[0]:snapped[1]] == "fibromyalgia"


def test_snap_span_boundaries_non_preposition_preserved():
    text = "Patient walked to the door."
    # 'door' ends in 'or', but is a full noun, not the preposition 'or'
    span = (22, 26, "mobility")
    snapped, changed = snap_span_boundaries(span, text)
    assert changed is False
    assert snapped == (22, 26, "mobility")
    assert text[snapped[0]:snapped[1]] == "door"


def test_fuse_same_category_spans_connective():
    text = "She is worried about paying her rent and utility bills."
    # Span 1: "paying" [21:27]
    # Span 2: "rent" [32:36]
    spans = {(21, 27, "financial insecurity"), (32, 36, "financial insecurity")}
    fused, merges, records = fuse_same_category_spans(spans, text, max_gap=6)
    assert merges == 1
    assert len(fused) == 1
    f_span = list(fused)[0]
    assert f_span == (21, 36, "financial insecurity")
    assert text[f_span[0]:f_span[1]] == "paying her rent"


def test_fuse_same_category_spans_blocks_sentence_boundary():
    text = "Patient feels sad. He has no friends."
    # Span 1: "sad" in sent 1, Span 2: "no friends" in sent 2
    spans = {(14, 17, "social isolation"), (25, 35, "social isolation")}
    fused, merges, _ = fuse_same_category_spans(spans, text, max_gap=15)
    # Merging across '.' MUST be blocked
    assert merges == 0
    assert len(fused) == 2


def test_fuse_same_category_spans_blocks_different_categories():
    text = "Patient is unemployed and homeless."
    # "unemployed" (job insecurity) vs "homeless" (housing insecurity)
    spans = {(11, 21, "job insecurity"), (26, 34, "housing insecurity")}
    fused, merges, _ = fuse_same_category_spans(spans, text, max_gap=10)
    assert merges == 0
    assert len(fused) == 2


def test_fuse_same_category_spans_blocks_substantive_gap():
    text = "Patient consumed alcohol and then drove a car into a ditch."
    # Gap contains 'and then drove a car into a ditch' (substantive actions)
    spans = {(17, 24, "substance abuse"), (51, 56, "substance abuse")}
    fused, merges, _ = fuse_same_category_spans(spans, text, max_gap=50)
    # Gap has substantive words ('drove', 'car', 'ditch'), not whitelisted connectives
    assert merges == 0
    assert len(fused) == 2


def test_snap_span_boundaries_preserves_balanced_closing_parenthesis():
    text = "Patient is suffering from post-traumatic stress disorder (PTSD) and has flashbacks."
    # Span includes balanced acronym: "post-traumatic stress disorder (PTSD)" [26:63]
    span = (26, 63, "psychiatric symptoms or disorders")
    snapped, changed = snap_span_boundaries(span, text)
    # Must preserve the closing parenthesis since '(' is present
    assert changed is False
    assert snapped == (26, 63, "psychiatric symptoms or disorders")
    assert text[snapped[0]:snapped[1]] == "post-traumatic stress disorder (PTSD)"


def test_snap_span_boundaries_trims_unbalanced_orphan_parenthesis():
    text = "Patient reports severe anxiety) from work."
    # Span includes trailing orphan parenthesis with no opening '('
    span = (16, 31, "psychiatric symptoms or disorders")
    snapped, changed = snap_span_boundaries(span, text)
    assert changed is True
    assert snapped == (16, 30, "psychiatric symptoms or disorders")
    assert text[snapped[0]:snapped[1]] == "severe anxiety"


def test_snap_span_boundaries_preserves_balanced_brackets():
    text = "Diagnosis: major depressive disorder [MDD] noted."
    span = (11, 43, "psychiatric symptoms or disorders")
    snapped, changed = snap_span_boundaries(span, text)
    # Balanced internal brackets for acronym preserved
    assert text[snapped[0]:snapped[1]] == "major depressive disorder [MDD]"


def test_snap_span_boundaries_trims_unbalanced_orphan_bracket():
    text = "Diagnosis of severe depression] noted."
    span = (13, 31, "psychiatric symptoms or disorders")
    snapped, changed = snap_span_boundaries(span, text)
    assert changed is True
    assert text[snapped[0]:snapped[1]] == "severe depression"


def test_fuse_hyphenated_words():
    text = "Restraining order against her ex-husband."
    # Subwords split across hyphen: "ex" [30:32] and "husband" [33:40]
    spans = {(30, 32, "legal problems"), (33, 40, "legal problems")}
    fused, merges, records = fuse_same_category_spans(spans, text, max_gap=2)
    assert merges == 1
    assert len(fused) == 1
    f_span = list(fused)[0]
    assert f_span == (30, 40, "legal problems")
    assert text[f_span[0]:f_span[1]] == "ex-husband"


def test_fuse_blocks_distinct_adjacent_gold_mentions_under_gap2():
    text = "Patient smokes a pack of cigarettes a day and drinks alcohol excessively."
    # Distinct coordinate clinical habits in the gold standard
    s1 = (8, 42, "substance abuse")  # "smokes a pack of cigarettes a day"
    s2 = (47, 73, "substance abuse")  # "drinks alcohol excessively"
    spans = {s1, s2}
    # With max_gap=2, the gap ' and ' (5 chars) prevents merging distinct coordinate symptoms
    fused, merges, _ = fuse_same_category_spans(spans, text, max_gap=2)
    assert merges == 0
    assert len(fused) == 2
    assert s1 in fused
    assert s2 in fused


def test_fuse_same_category_spans_blocks_question_exclamation_boundaries():
    text = "Is she lonely? She has no friends!"
    # Question mark and exclamation boundaries
    spans = {(7, 13, "social isolation"), (23, 33, "social isolation")}
    fused, merges, _ = fuse_same_category_spans(spans, text, max_gap=15)
    assert merges == 0
    assert len(fused) == 2


def test_production_decoder_integrity():
    """Verify that the production decoder in src/transformer_dataset.py remains intact."""
    import inspect
    sig = inspect.signature(production_decode)
    # Signature: (logits, active_mask, offset_mapping, text, threshold)
    assert "threshold" in sig.parameters
    assert sig.parameters["threshold"].default == 0.5
