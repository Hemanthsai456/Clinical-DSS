"""
Milestone 3: Multi-Label SBDH Tokenization, BIO Target Alignment, and Dataset Encoding.

Key Architectural Guarantees:
1. Multi-Label Support: Independent BIO labels for 15 SBDH categories (15 * 3 = 45 binary targets per token).
   Tokens overlapping spans carrying multiple categories (e.g., ex_no 7062) receive active B/I targets for
   all assigned categories simultaneously without label collision.
2. Character Offset Recovery: Uses Hugging Face FastTokenizer offset_mapping to map every token directly
   to exact character boundaries in the raw clinical note.
3. Explicit Handling of Ambiguous Training Annotations:
   ex_no 1192 in the training set contains an ungrounded repeated 'pain' span with two plausible locations.
   Rather than guessing an arbitrary offset, the candidate tokens for 'pain' in ex_no 1192 are explicitly
   masked out from the loss calculation (loss_mask = 0 for the 'pain' category on those tokens).
4. Subword Alignment: First subword token receives B_{cat}, subsequent subwords receive I_{cat}.
   Non-span tokens receive O_{cat} for all categories.
   Special tokens (<s>, </s>, <pad>) are masked from loss (active_mask = 0).
"""
import os
import sys
from typing import Dict, Any, List, Tuple, Optional, Set
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.span_alignment import align_spans_in_note, parse_delimited_field


SBDH_CATEGORIES: List[str] = [
    "barriers to care",
    "financial insecurity",
    "food insecurity",
    "housing insecurity",
    "job insecurity",
    "legal problems",
    "loss of relationship",
    "pain",
    "patient disability",
    "physical isolation",
    "psychiatric symptoms or disorders",
    "social isolation",
    "substance abuse",
    "transitions of care",
    "violence",
]

NUM_CATEGORIES: int = len(SBDH_CATEGORIES)  # 15
NUM_LABELS: int = NUM_CATEGORIES * 3        # 45 (B, I, O per category)

CAT_TO_IDX: Dict[str, int] = {cat: idx for idx, cat in enumerate(SBDH_CATEGORIES)}
IDX_TO_CAT: Dict[int, str] = {idx: cat for idx, cat in enumerate(SBDH_CATEGORIES)}

# Label indices for category k:
# B_k = 3*k + 0
# I_k = 3*k + 1
# O_k = 3*k + 2
def get_label_indices(cat_idx: int) -> Tuple[int, int, int]:
    base = cat_idx * 3
    return base, base + 1, base + 2


def encode_single_note(
    text: str,
    gold_intervals: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = 256,
    ex_no: Optional[int] = None,
    ambiguous_cases: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, np.ndarray]:
    """
    Encode a clinical note and its resolved gold intervals into token IDs, attention masks,
    token offset mappings, active token masks, multi-label BIO targets, and loss masks.

    Returns dict containing:
    - input_ids: (max_length,) int64
    - attention_mask: (max_length,) int64
    - offset_mapping: (max_length, 2) int64
    - active_mask: (max_length,) float32 (1.0 for real text tokens, 0.0 for special/pad)
    - targets: (max_length, 45) float32 (binary indicator for B, I, O per category)
    - loss_mask: (max_length, 15) float32 (1.0 for valid supervision, 0.0 for ambiguous spans)
    """
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
    )

    input_ids = np.array(encoding["input_ids"], dtype=np.int64)
    attention_mask = np.array(encoding["attention_mask"], dtype=np.int64)
    offset_mapping = np.array(encoding["offset_mapping"], dtype=np.int64)
    seq_len = len(input_ids)

    # Active mask: 1.0 for non-special text tokens, 0.0 for special tokens and padding
    active_mask = np.zeros(seq_len, dtype=np.float32)
    for i in range(seq_len):
        start, end = offset_mapping[i]
        # Real text tokens have start < end
        if attention_mask[i] == 1 and start < end:
            active_mask[i] = 1.0

    # Initialize targets: For each category, default all tokens to O (Outside)
    # targets shape: (seq_len, 45)
    targets = np.zeros((seq_len, NUM_LABELS), dtype=np.float32)
    for cat_idx in range(NUM_CATEGORIES):
        _, _, o_idx = get_label_indices(cat_idx)
        # Default active tokens to O_k = 1.0
        targets[active_mask == 1.0, o_idx] = 1.0

    # Loss mask per category: shape (seq_len, 15)
    # Allows category-specific masking for ungrounded/ambiguous cases
    loss_mask = np.ones((seq_len, NUM_CATEGORIES), dtype=np.float32)

    # Track any intervals that fall outside tokenizer max_length (truncation)
    max_char_offset = max([e for s, e in offset_mapping if e > s]) if any(e > s for s, e in offset_mapping) else 0
    truncated_intervals = []

    # Populate B and I targets from resolved gold intervals
    for interval in gold_intervals:
        int_start = interval["start_char"]
        int_end = interval["end_char"]
        cats = interval.get("categories", [])

        # Find all active tokens that overlap with [int_start, int_end]
        overlapping_token_indices = []
        for i in range(seq_len):
            if active_mask[i] == 1.0:
                t_start, t_end = offset_mapping[i]
                # Non-empty overlap condition
                if max(t_start, int_start) < min(t_end, int_end):
                    overlapping_token_indices.append(i)

        if not overlapping_token_indices:
            if int_start >= max_char_offset:
                truncated_intervals.append(interval)
            continue

        for cat in cats:
            if cat not in CAT_TO_IDX:
                continue
            cat_idx = CAT_TO_IDX[cat]
            b_idx, i_idx, o_idx = get_label_indices(cat_idx)

            first_tok = overlapping_token_indices[0]
            # First subword token gets B, unset O
            targets[first_tok, b_idx] = 1.0
            targets[first_tok, o_idx] = 0.0

            # Subsequent subword tokens get I, unset O
            for sub_tok in overlapping_token_indices[1:]:
                targets[sub_tok, i_idx] = 1.0
                targets[sub_tok, o_idx] = 0.0

    # Explicit Handling of Ambiguous Annotations (e.g. ex_no 1192):
    # If the note contains ambiguous repeated mentions, do NOT guess the offset.
    # Mask out candidate tokens for that specific category in loss_mask.
    if ambiguous_cases:
        for amb in ambiguous_cases:
            amb_cat = amb.get("sbdh")
            if amb_cat in CAT_TO_IDX:
                amb_cat_idx = CAT_TO_IDX[amb_cat]
                for cand in amb.get("candidates", []):
                    c_start = cand["start_char"]
                    c_end = cand["end_char"]
                    for i in range(seq_len):
                        if active_mask[i] == 1.0:
                            t_start, t_end = offset_mapping[i]
                            if max(t_start, c_start) < min(t_end, c_end):
                                # Mask out this token for this specific category
                                loss_mask[i, amb_cat_idx] = 0.0

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "offset_mapping": offset_mapping,
        "active_mask": active_mask,
        "targets": targets,
        "loss_mask": loss_mask,
        "truncated_intervals": truncated_intervals,
    }


def decode_predictions_to_spans(
    logits: np.ndarray,
    active_mask: np.ndarray,
    offset_mapping: np.ndarray,
    text: Optional[str] = None,
    threshold: float = 0.5,
) -> Set[Tuple[int, int, str]]:
    """
    Decode multi-label token logits into character-level spans and SBDH categories.

    Decision Rules:
    1. Outside (O) Condition:
       Token t is O if p_O >= max(p_B, p_I) or max(p_B, p_I) <= threshold.
       Any currently active span is closed and emitted.
    2. Active Beginning (B) Condition:
       Token t is B if max(p_B, p_I) > threshold and p_B >= p_I.
       Any previously open span is closed and emitted; a new span starts at t.
    3. Active Continuation (I) Condition:
       Token t is I if max(p_B, p_I) > threshold and p_I > p_B.
       If a span is open, it continues. If no span is open (orphan I), deterministic
       BIO repair treats it as B.
    4. Exact Character Boundary Refinement:
       Subword token offsets that span whitespace (e.g. leading spaces in BPE tokens)
       are cleanly trimmed against the raw text to match exact word boundaries.

    Returns:
    - Set of (start_char, end_char, category) tuples.
    """
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
    seq_len = len(active_mask)
    predicted_spans: Set[Tuple[int, int, str]] = set()

    for cat_idx, cat in enumerate(SBDH_CATEGORIES):
        b_idx, i_idx, o_idx = get_label_indices(cat_idx)

        in_span = False
        span_start_tok = -1
        span_end_tok = -1

        for t in range(seq_len):
            if active_mask[t] != 1.0:
                if in_span:
                    s_char = int(offset_mapping[span_start_tok][0])
                    e_char = int(offset_mapping[span_end_tok][1])
                    if s_char < e_char:
                        if text:
                            sub = text[s_char:e_char]
                            s_char += len(sub) - len(sub.lstrip())
                            e_char -= len(sub) - len(sub.rstrip())
                        if s_char < e_char:
                            predicted_spans.add((s_char, e_char, cat))
                    in_span = False
                continue

            p_b = probs[t, b_idx]
            p_i = probs[t, i_idx]
            p_o = probs[t, o_idx]

            # 1. Outside condition
            if p_o >= max(p_b, p_i) or max(p_b, p_i) <= threshold:
                if in_span:
                    s_char = int(offset_mapping[span_start_tok][0])
                    e_char = int(offset_mapping[span_end_tok][1])
                    if s_char < e_char:
                        if text:
                            sub = text[s_char:e_char]
                            s_char += len(sub) - len(sub.lstrip())
                            e_char -= len(sub) - len(sub.rstrip())
                        if s_char < e_char:
                            predicted_spans.add((s_char, e_char, cat))
                    in_span = False
                continue

            # 2. Active Beginning condition
            if p_b >= p_i:
                if in_span:
                    s_char = int(offset_mapping[span_start_tok][0])
                    e_char = int(offset_mapping[span_end_tok][1])
                    if s_char < e_char:
                        if text:
                            sub = text[s_char:e_char]
                            s_char += len(sub) - len(sub.lstrip())
                            e_char -= len(sub) - len(sub.rstrip())
                        if s_char < e_char:
                            predicted_spans.add((s_char, e_char, cat))
                in_span = True
                span_start_tok = t
                span_end_tok = t
            else:
                # 3. Active Inside condition (with orphan I repair)
                if in_span:
                    span_end_tok = t
                else:
                    in_span = True
                    span_start_tok = t
                    span_end_tok = t

        if in_span:
            s_char = int(offset_mapping[span_start_tok][0])
            e_char = int(offset_mapping[span_end_tok][1])
            if s_char < e_char:
                if text:
                    sub = text[s_char:e_char]
                    s_char += len(sub) - len(sub.lstrip())
                    e_char -= len(sub) - len(sub.rstrip())
                if s_char < e_char:
                    predicted_spans.add((s_char, e_char, cat))

    return predicted_spans


class SynthSBDHDataset(Dataset):
    """
    PyTorch Dataset for Synth-SBDH Multi-Label Span Extraction.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 256,
        is_training: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training
        self.samples = []

        # Pre-align all notes
        for idx, row in self.df.iterrows():
            ex_no = int(row["ex_no"]) if pd.notna(row.get("ex_no")) else None
            text = str(row["Text"])
            spans = parse_delimited_field(row.get("Textspan"))
            sbdhs = parse_delimited_field(row.get("SBDH"))
            presences = parse_delimited_field(row.get("Presence"))
            periods = parse_delimited_field(row.get("Period"))

            # Align spans with contextual disambiguation
            gold_aligned = align_spans_in_note(
                text=text,
                spans=spans,
                sbdhs=sbdhs,
                presences=presences,
                periods=periods,
                ex_no=ex_no,
                disambiguate_context=True,
            )

            encoded = encode_single_note(
                text=text,
                gold_intervals=gold_aligned["resolved_intervals"],
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                ex_no=ex_no,
                ambiguous_cases=gold_aligned["ambiguous_cases"],
            )

            self.samples.append({
                "ex_no": ex_no,
                "text": text,
                "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
                "offset_mapping": torch.tensor(encoded["offset_mapping"], dtype=torch.long),
                "active_mask": torch.tensor(encoded["active_mask"], dtype=torch.float32),
                "targets": torch.tensor(encoded["targets"], dtype=torch.float32),
                "loss_mask": torch.tensor(encoded["loss_mask"], dtype=torch.float32),
                "gold_intervals": gold_aligned["resolved_intervals"],
                "ambiguous_cases": gold_aligned["ambiguous_cases"],
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]
