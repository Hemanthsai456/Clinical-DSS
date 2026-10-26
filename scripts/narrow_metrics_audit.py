"""
Comprehensive Narrow Audit:
1. Independently recompute per-category TP, FP, FN, P, R, F1 for Arm 6A and Baseline.
2. Show exact mathematical summation to aggregate span-level metrics (Exact & Relaxed).
3. Verify note-level classification metrics separately (Micro and Macro).
"""
import os
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CAND_PATH = os.path.join(PROJECT_ROOT, "reports", "experiments", "candidate_test_evaluation", "candidate_test_evaluation_metrics.json")
BASE_PATH = os.path.join(PROJECT_ROOT, "reports", "sbdh_colab_transformer_metrics_corrected.json")

def f1(p, r):
    return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

def run_audit():
    with open(CAND_PATH) as f:
        cand = json.load(f)
    with open(BASE_PATH) as f:
        base = json.load(f)

    print("=" * 100)
    print("PART 1: SPAN-LEVEL PER-CATEGORY AUDIT (ARM 6A CANDIDATE)")
    print("=" * 100)

    cand_cats = cand["per_category_metrics"]
    
    # Table header
    print(f"{'Category':<32} | {'Gold':<5} | {'Pred':<5} | {'Ex TP':<5} | {'Ex FP':<5} | {'Ex FN':<5} | {'Ex P':<6} | {'Ex R':<6} | {'Ex F1':<6} | {'Rel TP':<6} | {'Rel FP':<6} | {'Rel FN':<6} | {'Rel P':<6} | {'Rel R':<6} | {'Rel F1':<6}")
    print("-" * 140)

    tot_gold = 0
    tot_pred = 0
    tot_ex_tp = 0
    tot_ex_fp = 0
    tot_ex_fn = 0
    tot_rel_tp = 0
    tot_rel_fp = 0
    tot_rel_fn = 0

    ex_f1_list = []
    rel_f1_list = []

    for cat in sorted(cand_cats.keys()):
        c = cand_cats[cat]
        gold = c["gold_support"]
        pred = c["predicted_count"]
        
        # Exact
        e_tp = c["exact_tp"]
        e_fp = c["exact_fp"]
        e_fn = c["exact_fn"]
        e_p = e_tp / (e_tp + e_fp) if (e_tp + e_fp) > 0 else 0.0
        e_r = e_tp / (e_tp + e_fn) if (e_tp + e_fn) > 0 else 0.0
        e_f = f1(e_p, e_r)
        
        # Verify internal category assertions
        assert e_tp + e_fn == gold, f"{cat} exact TP+FN != gold"
        assert e_tp + e_fp == pred, f"{cat} exact TP+FP != pred"
        assert abs(c["exact_precision"] - round(e_p, 4)) < 1e-6
        assert abs(c["exact_recall"] - round(e_r, 4)) < 1e-6
        assert abs(c["exact_f1"] - round(e_f, 4)) < 1e-6

        # Relaxed
        r_tp = c["relaxed_tp"]
        r_fp = c["relaxed_fp"]
        r_fn = c["relaxed_fn"]
        r_p = r_tp / (r_tp + r_fp) if (r_tp + r_fp) > 0 else 0.0
        r_r = r_tp / (r_tp + r_fn) if (r_tp + r_fn) > 0 else 0.0
        r_f = f1(r_p, r_r)

        assert r_tp + r_fn == gold, f"{cat} relaxed TP+FN != gold"
        assert r_tp + r_fp == pred, f"{cat} relaxed TP+FP != pred"
        assert abs(c["relaxed_precision"] - round(r_p, 4)) < 1e-6
        assert abs(c["relaxed_recall"] - round(r_r, 4)) < 1e-6
        assert abs(c["relaxed_f1"] - round(r_f, 4)) < 1e-6

        tot_gold += gold
        tot_pred += pred
        tot_ex_tp += e_tp
        tot_ex_fp += e_fp
        tot_ex_fn += e_fn
        tot_rel_tp += r_tp
        tot_rel_fp += r_fp
        tot_rel_fn += r_fn

        ex_f1_list.append(e_f)
        rel_f1_list.append(r_f)

        print(f"{cat:<32} | {gold:>5} | {pred:>5} | {e_tp:>5} | {e_fp:>5} | {e_fn:>5} | {e_p:>6.4f} | {e_r:>6.4f} | {e_f:>6.4f} | {r_tp:>6} | {r_fp:>6} | {r_fn:>6} | {r_p:>6.4f} | {r_r:>6.4f} | {r_f:>6.4f}")

    print("-" * 140)
    
    # Micro aggregation
    agg_ex_p = tot_ex_tp / (tot_ex_tp + tot_ex_fp)
    agg_ex_r = tot_ex_tp / (tot_ex_tp + tot_ex_fn)
    agg_ex_f = f1(agg_ex_p, agg_ex_r)

    agg_rel_p = tot_rel_tp / (tot_rel_tp + tot_rel_fp)
    agg_rel_r = tot_rel_tp / (tot_rel_tp + tot_rel_fn)
    agg_rel_f = f1(agg_rel_p, agg_rel_r)

    print(f"{'MICRO-SUM / AGGREGATE':<32} | {tot_gold:>5} | {tot_pred:>5} | {tot_ex_tp:>5} | {tot_ex_fp:>5} | {tot_ex_fn:>5} | {agg_ex_p:>6.4f} | {agg_ex_r:>6.4f} | {agg_ex_f:>6.4f} | {tot_rel_tp:>6} | {tot_rel_fp:>6} | {tot_rel_fn:>6} | {agg_rel_p:>6.4f} | {agg_rel_r:>6.4f} | {agg_rel_f:>6.4f}")
    
    macro_ex_f = float(np.mean(ex_f1_list))
    macro_rel_f = float(np.mean(rel_f1_list))
    print(f"{'MACRO-AVERAGE':<32} | {'-':>5} | {'-':>5} | {'-':>5} | {'-':>5} | {'-':>5} | {'-':>6} | {'-':>6} | {macro_ex_f:>6.4f} | {'-':>6} | {'-':>6} | {'-':>6} | {'-':>6} | {'-':>6} | {macro_rel_f:>6.4f}")

    print("\n" + "=" * 100)
    print("PART 2: NOTE-LEVEL CLASSIFICATION AUDIT (BASELINE VS ARM 6A)")
    print("=" * 100)

    base_note_cats = base["per_category_metrics"]
    print(f"{'Category':<32} | {'Base Notes':<10} | {'Base TP':<7} | {'Base FP':<7} | {'Base FN':<7} | {'Base P':<7} | {'Base R':<7} | {'Base F1':<7}")
    print("-" * 105)

    b_tot_sup = 0
    b_tot_tp = 0
    b_tot_fp = 0
    b_tot_fn = 0
    b_macro_p = []
    b_macro_r = []
    b_macro_f = []

    for cat in sorted(base_note_cats.keys()):
        bc = base_note_cats[cat]
        b_sup = bc["support"]
        b_tp = bc["tp"]
        b_fp = bc["fp"]
        b_fn = bc["fn"]

        bp = b_tp / (b_tp + b_fp) if (b_tp + b_fp) > 0 else 0.0
        br = b_tp / (b_tp + b_fn) if (b_tp + b_fn) > 0 else 0.0
        bf = f1(bp, br)

        assert abs(bc["precision"] - round(bp, 4)) < 1e-6
        assert abs(bc["recall"] - round(br, 4)) < 1e-6
        assert abs(bc["f1_score"] - round(bf, 4)) < 1e-6
        assert b_tp + b_fn == b_sup

        b_tot_sup += b_sup
        b_tot_tp += b_tp
        b_tot_fp += b_fp
        b_tot_fn += b_fn
        b_macro_p.append(bp)
        b_macro_r.append(br)
        b_macro_f.append(bf)

        print(f"{cat:<32} | {b_sup:>10} | {b_tp:>7} | {b_fp:>7} | {b_fn:>7} | {bp:>7.4f} | {br:>7.4f} | {bf:>7.4f}")

    print("-" * 105)
    b_micro_p = b_tot_tp / (b_tot_tp + b_tot_fp)
    b_micro_r = b_tot_tp / (b_tot_tp + b_tot_fn)
    b_micro_f = f1(b_micro_p, b_micro_r)
    b_mac_p = float(np.mean(b_macro_p))
    b_mac_r = float(np.mean(b_macro_r))
    b_mac_f = float(np.mean(b_macro_f))

    print(f"{'BASELINE NOTE MICRO-SUM':<32} | {b_tot_sup:>10} | {b_tot_tp:>7} | {b_tot_fp:>7} | {b_tot_fn:>7} | {b_micro_p:>7.4f} | {b_micro_r:>7.4f} | {b_micro_f:>7.4f}")
    print(f"{'BASELINE NOTE MACRO-AVG':<32} | {'-':>10} | {'-':>7} | {'-':>7} | {'-':>7} | {b_mac_p:>7.4f} | {b_mac_r:>7.4f} | {b_mac_f:>7.4f}")

    # Check against baseline JSON
    assert abs(base["aggregate_classification_metrics"]["micro"]["precision"] - round(b_micro_p, 4)) < 1e-6
    assert abs(base["aggregate_classification_metrics"]["micro"]["recall"] - round(b_micro_r, 4)) < 1e-6
    assert abs(base["aggregate_classification_metrics"]["micro"]["f1_score"] - round(b_micro_f, 4)) < 1e-6
    assert abs(base["aggregate_classification_metrics"]["macro"]["precision"] - round(b_mac_p, 4)) < 1e-6
    assert abs(base["aggregate_classification_metrics"]["macro"]["recall"] - round(b_mac_r, 4)) < 1e-6
    assert abs(base["aggregate_classification_metrics"]["macro"]["f1_score"] - round(b_mac_f, 4)) < 1e-6
    print("\n[VERIFIED] Baseline Note Micro (0.8888) and Macro (0.8684) exactly derived from per-category note counts!")

    print("\n" + "=" * 100)
    print("PART 3: NOTE-LEVEL COMPARISON SUMMARY")
    print("=" * 100)
    print(f"Metric                        | Frozen Baseline | Arm 6A Candidate | Absolute Delta")
    print("-" * 75)
    print(f"Note Micro Precision          | {base['aggregate_classification_metrics']['micro']['precision']:<15.4f} | {cand['aggregate_classification_metrics']['micro']['precision']:<16.4f} | {cand['aggregate_classification_metrics']['micro']['precision'] - base['aggregate_classification_metrics']['micro']['precision']:+.4f}")
    print(f"Note Micro Recall             | {base['aggregate_classification_metrics']['micro']['recall']:<15.4f} | {cand['aggregate_classification_metrics']['micro']['recall']:<16.4f} | {cand['aggregate_classification_metrics']['micro']['recall'] - base['aggregate_classification_metrics']['micro']['recall']:+.4f}")
    print(f"Note Micro F1-Score           | {base['aggregate_classification_metrics']['micro']['f1_score']:<15.4f} | {cand['aggregate_classification_metrics']['micro']['f1_score']:<16.4f} | {cand['aggregate_classification_metrics']['micro']['f1_score'] - base['aggregate_classification_metrics']['micro']['f1_score']:+.4f}")
    print(f"Note Macro Precision          | {base['aggregate_classification_metrics']['macro']['precision']:<15.4f} | {cand['aggregate_classification_metrics']['macro']['precision']:<16.4f} | {cand['aggregate_classification_metrics']['macro']['precision'] - base['aggregate_classification_metrics']['macro']['precision']:+.4f}")
    print(f"Note Macro Recall             | {base['aggregate_classification_metrics']['macro']['recall']:<15.4f} | {cand['aggregate_classification_metrics']['macro']['recall']:<16.4f} | {cand['aggregate_classification_metrics']['macro']['recall'] - base['aggregate_classification_metrics']['macro']['recall']:+.4f}")
    print(f"Note Macro F1-Score           | {base['aggregate_classification_metrics']['macro']['f1_score']:<15.4f} | {cand['aggregate_classification_metrics']['macro']['f1_score']:<16.4f} | {cand['aggregate_classification_metrics']['macro']['f1_score'] - base['aggregate_classification_metrics']['macro']['f1_score']:+.4f}")
    print("=" * 100)

if __name__ == "__main__":
    run_audit()
