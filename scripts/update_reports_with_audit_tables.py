import os
import json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
audit_json = os.path.join(ROOT, "reports", "experiments", "candidate_test_evaluation", "side_by_side_narrow_audit_metrics.json")
with open(audit_json, "r") as f:
    d = json.load(f)

# Build the three tables
t1 = []
t1.append("#### Table 12.3A: Note-Level Category Classification (Baseline vs Arm 6A Candidate)")
t1.append("*Procedure: Binary multi-label classification per note-category pair (1 = category present in note, 0 = absent). Evaluated across all 1,732 reviewed test notes (2,827 total note-category occurrences).*\n")
t1.append("| SBDH Category | Note Support | Base TP | Cand TP | TP Delta | Base P | Cand P | Base R | Cand R | Base F1 | Cand F1 | F1 Delta |")
t1.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
for r in d["table_1_note_level"]:
    c = r["category"]
    diff_tp = r["cand_tp"] - r["base_tp"]
    diff_f1 = r["cand_f1"] - r["base_f1"]
    t1.append(f"| **{c}** | {r['notes_support']} | {r['base_tp']} | {r['cand_tp']} | {diff_tp:+d} | {r['base_p']:.4f} | {r['cand_p']:.4f} | {r['base_r']:.4f} | {r['cand_r']:.4f} | {r['base_f1']:.4f} | {r['cand_f1']:.4f} | {diff_f1:+.4f} |")
ag_note = d["aggregates"]["note_level"]
t1.append(f"| **NOTE MICRO-SUM** | **2,827** | **2,357** | **2,445** | **+88** | **{ag_note['baseline_micro']['p']:.4f}** | **{ag_note['candidate_micro']['p']:.4f}** | **{ag_note['baseline_micro']['r']:.4f}** | **{ag_note['candidate_micro']['r']:.4f}** | **{ag_note['baseline_micro']['f1']:.4f}** | **{ag_note['candidate_micro']['f1']:.4f}** | **+{ag_note['candidate_micro']['f1'] - ag_note['baseline_micro']['f1']:.4f}** |")
t1.append(f"| **NOTE MACRO-AVG** | — | — | — | — | — | — | — | — | **{ag_note['baseline_macro']['f1']:.4f}** | **{ag_note['candidate_macro']['f1']:.4f}** | **+{ag_note['candidate_macro']['f1'] - ag_note['baseline_macro']['f1']:.4f}** |\n")

t2 = []
t2.append("#### Table 12.3B: Span-Level Exact Matching (Baseline vs Arm 6A Candidate)")
t2.append("*Procedure: Exact character offset and category agreement: `(start_char, end_char, category)`. Evaluated across all 3,340 resolvable gold spans.*\n")
t2.append("| SBDH Category | Gold Support | Base Pred | Cand Pred | Base TP | Cand TP | TP Delta | Base P | Cand P | Base R | Cand R | Base F1 | Cand F1 | F1 Delta |")
t2.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
for r in d["table_2_span_level_exact"]:
    c = r["category"]
    diff_tp = r["cand_tp"] - r["base_tp"]
    diff_f1 = r["cand_f1"] - r["base_f1"]
    t2.append(f"| **{c}** | {r['gold_support']} | {r['base_preds']} | {r['cand_preds']} | {r['base_tp']} | {r['cand_tp']} | {diff_tp:+d} | {r['base_p']:.4f} | {r['cand_p']:.4f} | {r['base_r']:.4f} | {r['cand_r']:.4f} | {r['base_f1']:.4f} | {r['cand_f1']:.4f} | {diff_f1:+.4f} |")
ag_ex = d["aggregates"]["span_level_exact"]
t2.append(f"| **EXACT MICRO-SUM** | **3,340** | **2,991** | **3,225** | **{ag_ex['baseline_micro']['tp']}** | **{ag_ex['candidate_micro']['tp']}** | **+{ag_ex['candidate_micro']['tp'] - ag_ex['baseline_micro']['tp']}** | **{ag_ex['baseline_micro']['p']:.4f}** | **{ag_ex['candidate_micro']['p']:.4f}** | **{ag_ex['baseline_micro']['r']:.4f}** | **{ag_ex['candidate_micro']['r']:.4f}** | **{ag_ex['baseline_micro']['f1']:.4f}** | **{ag_ex['candidate_micro']['f1']:.4f}** | **+{ag_ex['candidate_micro']['f1'] - ag_ex['baseline_micro']['f1']:.4f}** |")
t2.append(f"| **EXACT MACRO-AVG** | — | — | — | — | — | — | — | — | — | — | **{ag_ex['baseline_macro']['f1']:.4f}** | **{ag_ex['candidate_macro']['f1']:.4f}** | **+{ag_ex['candidate_macro']['f1'] - ag_ex['baseline_macro']['f1']:.4f}** |\n")

t3 = []
t3.append("#### Table 12.3C: Span-Level Relaxed Matching (Baseline vs Arm 6A Candidate)")
t3.append("*Procedure: Kuhn's Maximum Bipartite 1-to-1 Matching under category agreement & character overlap $\\ge 1$. Evaluated across all 3,340 resolvable gold spans.*\n")
t3.append("| SBDH Category | Gold Support | Base Pred | Cand Pred | Base TP | Cand TP | TP Delta | Base P | Cand P | Base R | Cand R | Base F1 | Cand F1 | F1 Delta |")
t3.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
for r in d["table_3_span_level_relaxed"]:
    c = r["category"]
    diff_tp = r["cand_tp"] - r["base_tp"]
    diff_f1 = r["cand_f1"] - r["base_f1"]
    t3.append(f"| **{c}** | {r['gold_support']} | {r['base_preds']} | {r['cand_preds']} | {r['base_tp']} | {r['cand_tp']} | {diff_tp:+d} | {r['base_p']:.4f} | {r['cand_p']:.4f} | {r['base_r']:.4f} | {r['cand_r']:.4f} | {r['base_f1']:.4f} | {r['cand_f1']:.4f} | {diff_f1:+.4f} |")
ag_rel = d["aggregates"]["span_level_relaxed"]
t3.append(f"| **RELAXED MICRO-SUM** | **3,340** | **2,991** | **3,225** | **{ag_rel['baseline_micro']['tp']}** | **{ag_rel['candidate_micro']['tp']}** | **+{ag_rel['candidate_micro']['tp'] - ag_rel['baseline_micro']['tp']}** | **{ag_rel['baseline_micro']['p']:.4f}** | **{ag_rel['candidate_micro']['p']:.4f}** | **{ag_rel['baseline_micro']['r']:.4f}** | **{ag_rel['candidate_micro']['r']:.4f}** | **{ag_rel['baseline_micro']['f1']:.4f}** | **{ag_rel['candidate_micro']['f1']:.4f}** | **+{ag_rel['candidate_micro']['f1'] - ag_rel['baseline_micro']['f1']:.4f}** |")
t3.append(f"| **RELAXED MACRO-AVG** | — | — | — | — | — | — | — | — | — | — | **{ag_rel['baseline_macro']['f1']:.4f}** | **{ag_rel['candidate_macro']['f1']:.4f}** | **+{ag_rel['candidate_macro']['f1'] - ag_rel['baseline_macro']['f1']:.4f}** |\n")

new_section_content = (
    "### 12.3 Per-Category Breakdown (Reviewed Test Set)\n\n"
    "To maintain absolute methodological clarity, per-category evaluation metrics are partitioned into three distinct, unambiguous side-by-side comparisons using identical evaluation procedures for both the frozen baseline and the Arm 6A candidate.\n\n"
    + "\n".join(t1) + "\n"
    + "\n".join(t2) + "\n"
    + "\n".join(t3) + "\n"
)

# 1. Update final_technical_report.md
final_report_path = os.path.join(ROOT, "reports", "final_technical_report.md")
with open(final_report_path, "r", encoding="utf-8") as f:
    final_text = f.read()

# Locate section 12.3 to 12.4
sec12_start = final_text.find("### 12.3 Per-Category Breakdown (Reviewed Test Set)")
sec12_end = final_text.find("### 12.4 Precision–Recall Trade-Off Analysis")
if sec12_start != -1 and sec12_end != -1:
    final_text = final_text[:sec12_start] + new_section_content + final_text[sec12_end:]
    with open(final_report_path, "w", encoding="utf-8") as f:
        f.write(final_text)
    print("Successfully updated reports/final_technical_report.md!")
else:
    print(f"Error locating boundaries in final_technical_report.md (start={sec12_start}, end={sec12_end})")

# 2. Update candidate_test_evaluation_report.md
cand_report_path = os.path.join(ROOT, "reports", "experiments", "candidate_test_evaluation", "candidate_test_evaluation_report.md")
with open(cand_report_path, "r", encoding="utf-8") as f:
    cand_text = f.read()

# Update Table names for Section 5
cand_sec_content = new_section_content.replace("12.3A", "5A").replace("12.3B", "5B").replace("12.3C", "5C").replace("### 12.3 Per-Category Breakdown (Reviewed Test Set)", "## 5. Per-Category Test Comparison")
sec5_start = cand_text.find("## 5. Per-Category Test Comparison")
sec5_end = cand_text.find("## 6. Detailed Analysis of Test Performance")
if sec5_start != -1 and sec5_end != -1:
    cand_text = cand_text[:sec5_start] + cand_sec_content + "---\n\n" + cand_text[sec5_end:]
    with open(cand_report_path, "w", encoding="utf-8") as f:
        f.write(cand_text)
    print("Successfully updated reports/experiments/candidate_test_evaluation/candidate_test_evaluation_report.md!")
else:
    print(f"Error locating boundaries in candidate_test_evaluation_report.md (start={sec5_start}, end={sec5_end})")
