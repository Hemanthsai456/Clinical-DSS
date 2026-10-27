"""
Experimental Lexical Boundary Snapping and Span Fusion for Track B Experiment 2.2.

This module provides deterministic, rule-based post-processing methods for SBDH spans:
1. Lexical Boundary Snapping:
   - Expands spans that cut through alphanumeric words.
   - Trims leading and trailing punctuation and whitespace.
   - Trims dangling trailing prepositions/conjunctions (e.g., 'for', 'to', 'and', 'or', 'of').
2. Conservative Same-Category Span Fusion:
   - Merges adjacent or overlapping spans of the identical SBDH category.
   - Enforces sentence boundary integrity (strictly prohibits merging across sentence terminators).
   - Restricts gap length and requires gap text to consist solely of whitespace, punctuation,
     or allowed grammatical connectives.

IMPORTANT: This module is purely experimental and does NOT modify or replace the production
decoder in src/transformer_dataset.py or the Streamlit application in app/.
"""
from typing import Set, Tuple, List, Optional, Dict, Any
import re

# Dangling prepositions and conjunctions that cannot legitimately terminate a standalone SBDH mention
DANGLING_PREPOSITIONS = {
    "for", "to", "and", "or", "with", "in", "at", "from", "of", "by", "as", "about", "into", "on", "due"
}

PUNCT_CHARS = '.,;:!?"\'()[]{}-/'

# Whitelist of allowed non-substantive connectives for same-category span fusion
DEFAULT_ALLOWED_CONNECTIVES = {
    "and", "or", "her", "his", "their", "my", "the", "a", "an", "of", "to", "in", "for", "with"
}

# Boundary punctuation blockers for span fusion
DEFAULT_BLOCKED_PUNCTUATION = {".", "!", "?", "\n"}
EXTENDED_BLOCKED_PUNCTUATION = {".", "!", "?", "\n", ",", ";", ":"}

# Non-specific generic stopwords for validation ablation testing
SBDH_STOPWORDS = {"life", "home", "stay", "care"}


def _is_trimmable_trailing_punct(char: str, s: int, e: int, text: str) -> bool:
    """
    Checks if a trailing punctuation character is safe to trim.
    Preserves closing delimiters (e.g. ')', ']', '}') when they balance an opening
    delimiter inside the span (e.g., 'post-traumatic stress disorder (PTSD)').
    """
    if char not in PUNCT_CHARS:
        return False
    sub = text[s:e]
    if char == ")" and sub.count("(") >= sub.count(")"):
        return False
    if char == "]" and sub.count("[") >= sub.count("]"):
        return False
    if char == "}" and sub.count("{") >= sub.count("}"):
        return False
    return True


def _is_trimmable_leading_punct(char: str, s: int, e: int, text: str) -> bool:
    """
    Checks if a leading punctuation character is safe to trim.
    Preserves opening delimiters when balanced with a closing delimiter inside the span.
    """
    if char not in PUNCT_CHARS:
        return False
    sub = text[s:e]
    if char == "(" and sub.count(")") >= sub.count("(") and len(sub) > 2 and sub[-1] != ")":
        return False
    if char == "[" and sub.count("]") >= sub.count("[") and len(sub) > 2 and sub[-1] != "]":
        return False
    return True


def snap_span_boundaries(
    span: Tuple[int, int, str],
    text: Optional[str] = None,
    snap_words: bool = True,
    trim_punct: bool = True,
    trim_dangling_prep: bool = True,
) -> Tuple[Tuple[int, int, str], bool]:
    """
    Applies deterministic lexical boundary snapping to a single (start_char, end_char, category) tuple.

    Returns:
        ((new_start, new_end, category), was_modified)
    """
    s, e, cat = span
    orig_s, orig_e = s, e

    if s >= e or not text or s < 0 or e > len(text):
        return span, False

    # 1. Word boundary expansion if cut mid-word by subword tokenization
    if snap_words:
        while s > 0 and text[s - 1].isalnum() and text[s].isalnum():
            s -= 1
        while e < len(text) and text[e - 1].isalnum() and text[e].isalnum():
            e += 1

    # 2. Whitespace trimming
    sub = text[s:e]
    lead_ws = len(sub) - len(sub.lstrip())
    s += lead_ws
    sub = text[s:e]
    trail_ws = len(sub) - len(sub.rstrip())
    e -= trail_ws

    if s >= e:
        return (orig_s, orig_e, cat), False

    # 3. Punctuation trimming from span boundaries (preserving balanced delimiters)
    if trim_punct:
        while s < e and _is_trimmable_leading_punct(text[s], s, e, text):
            s += 1
        while e > s and _is_trimmable_trailing_punct(text[e - 1], s, e, text):
            e -= 1

    if s >= e:
        return (orig_s, orig_e, cat), False

    # 4. Dangling trailing preposition trimming
    # e.g., "facing possible jail time for" -> "facing possible jail time"
    if trim_dangling_prep:
        sub = text[s:e]
        words = sub.split()
        if len(words) > 1 and words[-1].lower() in DANGLING_PREPOSITIONS:
            last_word = words[-1]
            cand_e = e - len(last_word)
            while cand_e > s and text[cand_e - 1].isspace():
                cand_e -= 1
            if cand_e > s:
                e = cand_e

    # 5. Final whitespace & punctuation check
    sub = text[s:e]
    lead_ws = len(sub) - len(sub.lstrip())
    s += lead_ws
    sub = text[s:e]
    trail_ws = len(sub) - len(sub.rstrip())
    e -= trail_ws
    if trim_punct:
        while s < e and _is_trimmable_leading_punct(text[s], s, e, text):
            s += 1
        while e > s and _is_trimmable_trailing_punct(text[e - 1], s, e, text):
            e -= 1

    changed = (s != orig_s or e != orig_e) and (s < e)
    if s >= e:
        return (orig_s, orig_e, cat), False
    return (s, e, cat), changed


def fuse_same_category_spans(
    spans: Set[Tuple[int, int, str]],
    text: Optional[str] = None,
    max_gap: int = 6,
    allowed_connectives: Optional[Set[str]] = None,
    blocked_punctuation: Optional[Set[str]] = None,
) -> Tuple[Set[Tuple[int, int, str]], int, List[Dict[str, Any]]]:
    """
    Applies conservative same-category span fusion to a set of spans.

    Rules:
    - Only spans of the identical category are candidates.
    - Spans must reside within the same sentence (no blocked punctuation in gap).
    - Character gap (next_start - curr_end) must be <= max_gap (or abutting/overlapping).
    - Intervening gap text must be strictly whitespace, punctuation (e.g. hyphens),
      or whitelisted short grammatical connectives.

    Returns:
        (fused_spans_set, total_merges_count, audit_records_list)
    """
    if not spans or not text:
        return spans, 0, []

    if allowed_connectives is None:
        allowed_connectives = DEFAULT_ALLOWED_CONNECTIVES

    if blocked_punctuation is None:
        blocked_punctuation = DEFAULT_BLOCKED_PUNCTUATION

    sorted_spans = sorted(list(spans), key=lambda x: (x[2], x[0], x[1]))
    fused_spans: List[Tuple[int, int, str]] = []
    merges_count = 0
    merge_records: List[Dict[str, Any]] = []

    i = 0
    while i < len(sorted_spans):
        curr_s, curr_e, curr_c = sorted_spans[i]
        j = i + 1
        while j < len(sorted_spans):
            next_s, next_e, next_c = sorted_spans[j]
            if next_c != curr_c:
                break

            # Check overlap or abutting
            if next_s <= curr_e:
                orig_first = (curr_s, curr_e, curr_c)
                orig_second = (next_s, next_e, next_c)
                curr_e = max(curr_e, next_e)
                merges_count += 1
                merge_records.append({
                    "category": curr_c,
                    "span1": orig_first,
                    "span2": orig_second,
                    "merged_span": (curr_s, curr_e, curr_c),
                    "span1_text": text[orig_first[0]:orig_first[1]],
                    "span2_text": text[orig_second[0]:orig_second[1]],
                    "merged_text": text[curr_s:curr_e],
                    "reason": "overlap_or_abutting",
                    "gap": 0,
                    "gap_text": "",
                })
                j += 1
                continue

            gap = next_s - curr_e
            if gap <= max_gap:
                gap_text = text[curr_e:next_s]
                # Sentence / clause boundary check
                if any(punct in gap_text for punct in blocked_punctuation):
                    break

                # Connective check
                gap_stripped = gap_text.strip().lower()
                gap_words = set(re.findall(r"\b\w+\b", gap_stripped))
                if not gap_words or gap_words.issubset(allowed_connectives):
                    orig_first = (curr_s, curr_e, curr_c)
                    orig_second = (next_s, next_e, next_c)
                    curr_e = next_e
                    merges_count += 1
                    merge_records.append({
                        "category": curr_c,
                        "span1": orig_first,
                        "span2": orig_second,
                        "merged_span": (curr_s, curr_e, curr_c),
                        "span1_text": text[orig_first[0]:orig_first[1]],
                        "span2_text": text[orig_second[0]:orig_second[1]],
                        "merged_text": text[curr_s:curr_e],
                        "reason": f"gap_connective",
                        "gap": gap,
                        "gap_text": gap_text,
                    })
                    j += 1
                    continue
                else:
                    break
            else:
                break

        fused_spans.append((curr_s, curr_e, curr_c))
        i = j

    return set(fused_spans), merges_count, merge_records


def filter_stopword_spans(
    spans: Set[Tuple[int, int, str]],
    text: Optional[str] = None,
    stopwords: Optional[Set[str]] = None,
) -> Set[Tuple[int, int, str]]:
    """
    Validation-only ablation filter: removes single-word generic stopwords
    (e.g., 'life', 'home', 'stay', 'care') that lack SBDH specificity.
    """
    if not spans or not text:
        return spans
    if stopwords is None:
        stopwords = SBDH_STOPWORDS
    filtered = set()
    for s, e, c in spans:
        span_str = text[s:e].strip().lower()
        if span_str in stopwords:
            continue
        filtered.add((s, e, c))
    return filtered
