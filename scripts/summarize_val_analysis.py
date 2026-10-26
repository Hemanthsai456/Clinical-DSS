import json

data = json.load(open('reports/validation_error_analysis_experiment_1.json'))
sens = data['sensitivity_analysis']

print("=" * 105)
print("SENSITIVITY ANALYSIS SUMMARY ACROSS THRESHOLDS")
print("=" * 105)
print(f"{'Threshold':<10} | {'Exact F1':<10} | {'Exact P':<10} | {'Exact R':<10} | {'Relaxed F1':<10} | {'Relaxed P':<10} | {'Relaxed R':<10} | {'Boundary Errs':<15}")
print("-" * 105)
for th_key in ['threshold_0.20', 'threshold_0.30', 'threshold_0.40', 'threshold_0.50']:
    item = sens[th_key]
    th = item['threshold']
    ex = item['exact_matching']
    rel = item['relaxed_matching']
    b_err = item['boundary_errors_count']
    b_pct = item['boundary_error_rate_pct']
    print(f"{th:<10.2f} | {ex['f1_score']:<10.4f} | {ex['precision']:<10.4f} | {ex['recall']:<10.4f} | {rel['f1_score']:<10.4f} | {rel['precision']:<10.4f} | {rel['recall']:<10.4f} | {b_err} ({b_pct}%)")

print("\n" + "=" * 115)
print("PER-CATEGORY BREAKDOWN AT THRESHOLD 0.30 (VALIDATION SET, N = 876)")
print("=" * 115)
pcat = sens['threshold_0.30']['per_category']
print(f"{'Category':<35} | {'Gold':<5} | {'Pred':<5} | {'TP':<5} | {'FP':<5} | {'FN':<5} | {'Prec':<7} | {'Recall':<7} | {'F1':<7} | {'ExactTP':<7} | {'ExactRate':<9}")
print("-" * 115)
for cat, m in pcat.items():
    exact_rate = (m['exact_tp'] / m['relaxed_tp'] * 100) if m['relaxed_tp'] > 0 else 0.0
    print(f"{cat:<35} | {m['gold_support']:>5} | {m['predicted_count']:>5} | {m['relaxed_tp']:>5} | {m['relaxed_fp']:>5} | {m['relaxed_fn']:>5} | {m['relaxed_precision']:>7.4f} | {m['relaxed_recall']:>7.4f} | {m['relaxed_f1']:>7.4f} | {m['exact_tp']:>7} | {exact_rate:>8.1f}%")

print("\n" + "=" * 115)
print("WEAKEST CATEGORIES BY RELAXED F1 AT THRESHOLD 0.30")
print("=" * 115)
sorted_cats = sorted(pcat.items(), key=lambda x: x[1]['relaxed_f1'])
for cat, m in sorted_cats[:6]:
    print(f"{cat:<35}: F1={m['relaxed_f1']:.4f} (Rec={m['relaxed_recall']:.4f}, Prec={m['relaxed_precision']:.4f}, Gold={m['gold_support']}, TP={m['relaxed_tp']}, FN={m['relaxed_fn']}, FP={m['relaxed_fp']})")
