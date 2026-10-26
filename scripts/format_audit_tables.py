import json

d = json.load(open('reports/experiments/candidate_test_evaluation/side_by_side_narrow_audit_metrics.json'))

print("### Table 1: Note-Level Category Classification (Baseline vs Arm 6A Candidate)")
print("Evaluation Procedure: Binary classification per note-category pair (1 = category present in note, 0 = absent). Evaluated across all 1,732 reviewed test notes (2,827 total note-category occurrences).\n")
print("| SBDH Category | Note Support | Base TP | Cand TP | TP Delta | Base P | Cand P | Base R | Cand R | Base F1 | Cand F1 | F1 Delta |")
print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
for r in d['table_1_note_level']:
    c = r['category']
    diff_tp = r['cand_tp'] - r['base_tp']
    diff_f1 = r['cand_f1'] - r['base_f1']
    print(f"| **{c}** | {r['notes_support']} | {r['base_tp']} | {r['cand_tp']} | {diff_tp:+d} | {r['base_p']:.4f} | {r['cand_p']:.4f} | {r['base_r']:.4f} | {r['cand_r']:.4f} | {r['base_f1']:.4f} | {r['cand_f1']:.4f} | {diff_f1:+.4f} |")
ag = d['aggregates']['note_level']
print(f"| **NOTE MICRO-SUM** | **2,827** | **2,357** | **2,445** | **+88** | **{ag['baseline_micro']['p']:.4f}** | **{ag['candidate_micro']['p']:.4f}** | **{ag['baseline_micro']['r']:.4f}** | **{ag['candidate_micro']['r']:.4f}** | **{ag['baseline_micro']['f1']:.4f}** | **{ag['candidate_micro']['f1']:.4f}** | **+{ag['candidate_micro']['f1'] - ag['baseline_micro']['f1']:.4f}** |")
print(f"| **NOTE MACRO-AVG** | — | — | — | — | — | — | — | — | **{ag['baseline_macro']['f1']:.4f}** | **{ag['candidate_macro']['f1']:.4f}** | **+{ag['candidate_macro']['f1'] - ag['baseline_macro']['f1']:.4f}** |")

print("\n### Table 2: Span-Level Exact Matching (Baseline vs Arm 6A Candidate)")
print("Evaluation Procedure: Exact boundary and category agreement: (start_char, end_char, category). Evaluated across all 3,340 resolvable gold spans.\n")
print("| SBDH Category | Gold Support | Base Pred | Cand Pred | Base TP | Cand TP | TP Delta | Base P | Cand P | Base R | Cand R | Base F1 | Cand F1 | F1 Delta |")
print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
for r in d['table_2_span_level_exact']:
    c = r['category']
    diff_tp = r['cand_tp'] - r['base_tp']
    diff_f1 = r['cand_f1'] - r['base_f1']
    print(f"| **{c}** | {r['gold_support']} | {r['base_preds']} | {r['cand_preds']} | {r['base_tp']} | {r['cand_tp']} | {diff_tp:+d} | {r['base_p']:.4f} | {r['cand_p']:.4f} | {r['base_r']:.4f} | {r['cand_r']:.4f} | {r['base_f1']:.4f} | {r['cand_f1']:.4f} | {diff_f1:+.4f} |")
ag_ex = d['aggregates']['span_level_exact']
print(f"| **EXACT MICRO-SUM** | **3,340** | **2,991** | **3,225** | **{ag_ex['baseline_micro']['tp']}** | **{ag_ex['candidate_micro']['tp']}** | **+{ag_ex['candidate_micro']['tp'] - ag_ex['baseline_micro']['tp']}** | **{ag_ex['baseline_micro']['p']:.4f}** | **{ag_ex['candidate_micro']['p']:.4f}** | **{ag_ex['baseline_micro']['r']:.4f}** | **{ag_ex['candidate_micro']['r']:.4f}** | **{ag_ex['baseline_micro']['f1']:.4f}** | **{ag_ex['candidate_micro']['f1']:.4f}** | **+{ag_ex['candidate_micro']['f1'] - ag_ex['baseline_micro']['f1']:.4f}** |")
print(f"| **EXACT MACRO-AVG** | — | — | — | — | — | — | — | — | — | — | **{ag_ex['baseline_macro']['f1']:.4f}** | **{ag_ex['candidate_macro']['f1']:.4f}** | **+{ag_ex['candidate_macro']['f1'] - ag_ex['baseline_macro']['f1']:.4f}** |")

print("\n### Table 3: Span-Level Relaxed Matching (Baseline vs Arm 6A Candidate)")
print("Evaluation Procedure: Maximum Bipartite 1-to-1 Matching under category agreement & character overlap >= 1. Evaluated across all 3,340 resolvable gold spans.\n")
print("| SBDH Category | Gold Support | Base Pred | Cand Pred | Base TP | Cand TP | TP Delta | Base P | Cand P | Base R | Cand R | Base F1 | Cand F1 | F1 Delta |")
print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
for r in d['table_3_span_level_relaxed']:
    c = r['category']
    diff_tp = r['cand_tp'] - r['base_tp']
    diff_f1 = r['cand_f1'] - r['base_f1']
    print(f"| **{c}** | {r['gold_support']} | {r['base_preds']} | {r['cand_preds']} | {r['base_tp']} | {r['cand_tp']} | {diff_tp:+d} | {r['base_p']:.4f} | {r['cand_p']:.4f} | {r['base_r']:.4f} | {r['cand_r']:.4f} | {r['base_f1']:.4f} | {r['cand_f1']:.4f} | {diff_f1:+.4f} |")
ag_rel = d['aggregates']['span_level_relaxed']
print(f"| **RELAXED MICRO-SUM** | **3,340** | **2,991** | **3,225** | **{ag_rel['baseline_micro']['tp']}** | **{ag_rel['candidate_micro']['tp']}** | **+{ag_rel['candidate_micro']['tp'] - ag_rel['baseline_micro']['tp']}** | **{ag_rel['baseline_micro']['p']:.4f}** | **{ag_rel['candidate_micro']['p']:.4f}** | **{ag_rel['baseline_micro']['r']:.4f}** | **{ag_rel['candidate_micro']['r']:.4f}** | **{ag_rel['baseline_micro']['f1']:.4f}** | **{ag_rel['candidate_micro']['f1']:.4f}** | **+{ag_rel['candidate_micro']['f1'] - ag_rel['baseline_micro']['f1']:.4f}** |")
print(f"| **RELAXED MACRO-AVG** | — | — | — | — | — | — | — | — | — | — | **{ag_rel['baseline_macro']['f1']:.4f}** | **{ag_rel['candidate_macro']['f1']:.4f}** | **+{ag_rel['candidate_macro']['f1'] - ag_rel['baseline_macro']['f1']:.4f}** |")
