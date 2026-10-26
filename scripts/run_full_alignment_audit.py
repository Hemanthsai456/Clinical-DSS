import sys
import os
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from src.data_loaders import load_synth_sbdh
from src.span_alignment import parse_delimited_field, find_all_occurrences
from src.transformer_dataset import SBDH_CATEGORIES, CAT_TO_IDX

tokenizer = AutoTokenizer.from_pretrained("distilroberta-base", add_prefix_space=True)

dfs, _ = load_synth_sbdh()

splits_to_audit = ["train", "val", "test_reviewed"]

audit_summary = {}

for split_name in splits_to_audit:
    df = dfs[split_name]
    print(f"\n========================================================")
    print(f"AUDITING SPLIT: {split_name} ({len(df)} notes)")
    print(f"========================================================")
    
    total_raw_annotations = 0
    resolved_annotations = 0
    unresolved_by_reason = {
        "UNRESOLVED_MISMATCHED_REPEATED": 0,
        "UNRESOLVED_SPAN_NOT_FOUND": 0,
        "UNRESOLVED_DUPLICATE_ANNOTATION": 0,
        "UNRESOLVED_AMBIGUOUS_ORDER": 0,
    }
    
    co_occurring_multilabel_annotations = 0
    unique_resolved_intervals = set()
    truncated_annotations_count = 0
    notes_exceeding_256_tokens = 0
    max_tokens_seen = 0
    
    detailed_unresolved_records = []
    
    for idx, row in df.iterrows():
        ex_no = int(row["ex_no"]) if pd.notna(row.get("ex_no")) else idx
        text = str(row["Text"])
        spans = parse_delimited_field(row.get("Textspan"))
        sbdhs = parse_delimited_field(row.get("SBDH"))
        presences = parse_delimited_field(row.get("Presence"))
        periods = parse_delimited_field(row.get("Period"))
        
        n_spans = len(spans)
        total_raw_annotations += n_spans
        
        # Token length and truncation check
        enc = tokenizer(text, max_length=256, truncation=True, return_offsets_mapping=True)
        raw_tokens = tokenizer.tokenize(text)
        token_count = len(raw_tokens)
        if token_count > max_tokens_seen:
            max_tokens_seen = token_count
        if token_count > 256:
            notes_exceeding_256_tokens += 1
            
        # Get maximum character offset covered by the 256 tokens
        offsets = enc["offset_mapping"]
        max_char_covered = max([e for s, e in offsets]) if offsets else 0
        
        # Group annotations by normalized span
        annotations = []
        for i in range(n_spans):
            annotations.append({
                "ann_idx": i,
                "span": spans[i].strip(),
                "sbdh": sbdhs[i].strip().lower() if i < len(sbdhs) else "",
                "presence": presences[i].strip() if i < len(presences) else "",
                "period": periods[i].strip() if i < len(periods) else "",
            })
            
        span_groups = {}
        for ann in annotations:
            norm_span = ann["span"].lower()
            if norm_span not in span_groups:
                span_groups[norm_span] = []
            span_groups[norm_span].append(ann)
            
        note_resolved_count = 0
        note_unresolved_count = 0
        
        for norm_span, ann_list in span_groups.items():
            if not norm_span:
                continue
                
            raw_span_rep = ann_list[0]["span"]
            candidates = find_all_occurrences(text, raw_span_rep)
            num_candidates = len(candidates)
            num_annotations = len(ann_list)
            
            # Case 1: Span not found in note
            if num_candidates == 0:
                for ann in ann_list:
                    unresolved_by_reason["UNRESOLVED_SPAN_NOT_FOUND"] += 1
                    note_unresolved_count += 1
                    detailed_unresolved_records.append({
                        "split": split_name,
                        "ex_no": ex_no,
                        "span": ann["span"],
                        "sbdh": ann["sbdh"],
                        "reason": "UNRESOLVED_SPAN_NOT_FOUND",
                        "detail": f"Span '{ann['span']}' not found in note text."
                    })
                continue
                
            # Case 2: Exactly 1 candidate occurrence in note
            if num_candidates == 1:
                cand_s, cand_e, mtype = candidates[0]
                
                # Check for duplicate annotations (same category for same candidate)
                seen_cats = set()
                for ann in ann_list:
                    cat = ann["sbdh"]
                    if cat in seen_cats:
                        # Duplicate identical annotation!
                        unresolved_by_reason["UNRESOLVED_DUPLICATE_ANNOTATION"] += 1
                        note_unresolved_count += 1
                        detailed_unresolved_records.append({
                            "split": split_name,
                            "ex_no": ex_no,
                            "span": ann["span"],
                            "sbdh": cat,
                            "reason": "UNRESOLVED_DUPLICATE_ANNOTATION",
                            "detail": f"Duplicate annotation of category '{cat}' on same span interval [{cand_s}:{cand_e}]."
                        })
                    else:
                        seen_cats.add(cat)
                        resolved_annotations += 1
                        note_resolved_count += 1
                        unique_resolved_intervals.add((ex_no, cand_s, cand_e, cat))
                        
                        # Truncation check
                        if cand_e > max_char_covered:
                            truncated_annotations_count += 1
                            
                if len(seen_cats) > 1:
                    co_occurring_multilabel_annotations += len(seen_cats)
                continue
                
            # Case 3: Multiple candidates (num_candidates > 1)
            # Check contextual adjudication
            if ex_no == 7411 and norm_span == "home" and num_candidates == 2 and num_annotations == 1:
                # Disambiguated contextually: candidate 1 is 'her own home'
                cand_target = candidates[1]
                resolved_annotations += 1
                note_resolved_count += 1
                unique_resolved_intervals.add((ex_no, cand_target[0], cand_target[1], ann_list[0]["sbdh"]))
                if cand_target[1] > max_char_covered:
                    truncated_annotations_count += 1
                continue
                
            if ex_no == 7611 and norm_span == "pain" and num_candidates == 2 and num_annotations == 2:
                # Disambiguated contextually:
                # cand 0: current pain [31:35], cand 1: history pain [85:89]
                for c in candidates:
                    resolved_annotations += 1
                    note_resolved_count += 1
                    unique_resolved_intervals.add((ex_no, c[0], c[1], "pain"))
                    if c[1] > max_char_covered:
                        truncated_annotations_count += 1
                continue
                
            # If candidate count matches annotation count:
            if num_candidates == num_annotations:
                # Check if all annotations have the same category
                ann_cats = [a["sbdh"] for a in ann_list]
                if len(set(ann_cats)) == 1:
                    # Same category, e.g. 2 pain annotations and 2 pain mentions in text
                    for c in candidates:
                        resolved_annotations += 1
                        note_resolved_count += 1
                        unique_resolved_intervals.add((ex_no, c[0], c[1], ann_cats[0]))
                        if c[1] > max_char_covered:
                            truncated_annotations_count += 1
                else:
                    # Different categories, order is ambiguous without contextual evidence!
                    for ann in ann_list:
                        unresolved_by_reason["UNRESOLVED_AMBIGUOUS_ORDER"] += 1
                        note_unresolved_count += 1
                        detailed_unresolved_records.append({
                            "split": split_name,
                            "ex_no": ex_no,
                            "span": ann["span"],
                            "sbdh": ann["sbdh"],
                            "reason": "UNRESOLVED_AMBIGUOUS_ORDER",
                            "detail": f"N_cand=N_ann={num_candidates} with different categories {ann_cats}; occurrence mapping ambiguous."
                        })
                continue
                
            # If num_candidates != num_annotations:
            # DO NOT fall back to candidates[0]! Explicitly flag as UNRESOLVED_MISMATCHED_REPEATED.
            for ann in ann_list:
                unresolved_by_reason["UNRESOLVED_MISMATCHED_REPEATED"] += 1
                note_unresolved_count += 1
                detailed_unresolved_records.append({
                    "split": split_name,
                    "ex_no": ex_no,
                    "span": ann["span"],
                    "sbdh": ann["sbdh"],
                    "reason": "UNRESOLVED_MISMATCHED_REPEATED",
                    "detail": f"Span '{ann['span']}' appears {num_candidates} times in text, but has {num_annotations} annotation(s). No unambiguous contextual anchor."
                })

        # Note-level accounting assertion
        assert note_resolved_count + note_unresolved_count == n_spans, (
            f"Note {ex_no} accounting violation: resolved ({note_resolved_count}) + "
            f"unresolved ({note_unresolved_count}) != total ({n_spans})"
        )

    # Split-level accounting assertion
    total_unresolved = sum(unresolved_by_reason.values())
    assert resolved_annotations + total_unresolved == total_raw_annotations, (
        f"Split {split_name} accounting violation: resolved ({resolved_annotations}) + "
        f"unresolved ({total_unresolved}) != total raw ({total_raw_annotations})"
    )
    
    audit_summary[split_name] = {
        "notes": len(df),
        "total_raw_annotations": total_raw_annotations,
        "resolved_annotations": resolved_annotations,
        "unique_resolved_intervals": len(unique_resolved_intervals),
        "total_unresolved": total_unresolved,
        "unresolved_by_reason": unresolved_by_reason,
        "truncated_annotations": truncated_annotations_count,
        "notes_exceeding_256_tokens": notes_exceeding_256_tokens,
        "max_tokens_seen": max_tokens_seen,
        "detailed_unresolved": detailed_unresolved_records
    }
    
    print(f"Total Raw Annotations:       {total_raw_annotations:,}")
    print(f"Resolved Annotations:        {resolved_annotations:,}")
    print(f"Unique Resolved Intervals:   {len(unique_resolved_intervals):,}")
    print(f"Total Unresolved:            {total_unresolved:,}")
    for reason, cnt in unresolved_by_reason.items():
        print(f"  - {reason}: {cnt}")
    print(f"Truncated Annotations (>256 tok): {truncated_annotations_count}")
    print(f"Notes exceeding 256 tokens:       {notes_exceeding_256_tokens}")
    print(f"Max token length in split:        {max_tokens_seen}")
    print(f"Accounting Invariant: {resolved_annotations} + {total_unresolved} == {total_raw_annotations} -> PASSED!")

print("\n========================================================")
print("AUDIT COMPLETE FOR ALL SPLITS")
print("========================================================")
