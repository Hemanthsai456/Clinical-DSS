import json

data = json.load(open('reports/validation_error_analysis_experiment_1.json'))['detailed_030_analysis']
fps = data['fp_examples']
fns = data['fn_examples']
bnds = data['boundary_examples']

print("=" * 80)
print("REPRESENTATIVE ERROR EXAMPLES (VALIDATION SET, THRESHOLD 0.30)")
print("=" * 80)

for cat in ['barriers to care', 'physical isolation', 'patient disability', 'food insecurity', 'pain']:
    print(f"\n>>> Category: '{cat}'")
    print("  FALSE NEGATIVES (Missed gold spans):")
    for ex in fns.get(cat, [])[:3]:
        print(f"    - ex_no {ex['ex_no']}: Gold={ex['gold_span']}")
        print(f"      Context: \"...{ex['surrounding_context']}...\"")
    
    print("  FALSE POSITIVES (Spurious model predictions):")
    for ex in fps.get(cat, [])[:3]:
        print(f"    - ex_no {ex['ex_no']}: Pred={ex['pred_span']}")
        print(f"      Context: \"...{ex['surrounding_context']}...\"")

    print("  BOUNDARY ERRORS (Relaxed match but exact mismatch):")
    for ex in bnds.get(cat, [])[:3]:
        print(f"    - ex_no {ex['ex_no']}: Gold={ex['gold_span']} vs Pred={ex['pred_span']}")
