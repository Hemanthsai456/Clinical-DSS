import json

d = json.load(open('reports/experiments/decoder_ablation/decoder_ablation_metrics.json'))
a = d['validation_benchmarks']['strategy_a_th_0.30']['per_category']
b = d['validation_benchmarks']['strategy_b_th_0.30']['per_category']

print("=" * 115)
print("STRATEGY A (CONTROL) VS STRATEGY B (ACTIVE THRESHOLD) PER-CATEGORY COMPARISON (TH=0.30)")
print("=" * 115)
print(f"{'Category':<32} | {'A-Rec':<6} | {'B-Rec':<6} | {'DeltaRec':<8} | {'A-Prec':<6} | {'B-Prec':<6} | {'A-F1':<6} | {'B-F1':<6} | {'DeltaF1':<8} | {'Exact A':<7} | {'Exact B':<7}")
print("-" * 115)
for c in sorted(a.keys()):
    rec_a, rec_b = a[c]['relaxed_recall'], b[c]['relaxed_recall']
    prec_a, prec_b = a[c]['relaxed_precision'], b[c]['relaxed_precision']
    f1_a, f1_b = a[c]['relaxed_f1'], b[c]['relaxed_f1']
    d_rec = rec_b - rec_a
    d_f1 = f1_b - f1_a
    ex_a, ex_b = a[c]['exact_tp'], b[c]['exact_tp']
    print(f"{c:<32} | {rec_a:>6.4f} | {rec_b:>6.4f} | {d_rec:>+8.4f} | {prec_a:>6.4f} | {prec_b:>6.4f} | {f1_a:>6.4f} | {f1_b:>6.4f} | {d_f1:>+8.4f} | {ex_a:>7} | {ex_b:>7}")

print("\n" + "=" * 115)
print("AGGREGATE COMPARISON")
print("=" * 115)
for strat in ['strategy_a_th_0.30', 'strategy_b_th_0.30', 'strategy_b_th_0.40', 'strategy_c_margin_+0.00_th_0.30', 'strategy_c_margin_+0.05_th_0.30', 'strategy_c_margin_+0.10_th_0.30']:
    item = d['validation_benchmarks'][strat]
    name = item['strategy_name']
    ex = item['exact_matching']
    rel = item['relaxed_matching']
    preds = item['total_pred_spans']
    print(f"{name:<38} | Exact F1: {ex['f1_score']:.4f} (P={ex['precision']:.4f}, R={ex['recall']:.4f}, TP={ex['tp']}, FP={ex['fp']}, FN={ex['fn']}) | Relaxed F1: {rel['f1_score']:.4f} (P={rel['precision']:.4f}, R={rel['recall']:.4f}, TP={rel['tp']}, FP={rel['fp']}, FN={rel['fn']}, Preds={preds})")
