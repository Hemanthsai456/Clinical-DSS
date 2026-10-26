"""
Benchmark Script: Profiles the actual Track B Streamlit inference path.
Measures latency before (Strategy A Control) and after (Arm 6A Promoted Pipeline).

Hardware: Local CPU
Timing: Measures model load time and per-inference latency (mean, std, min, max).
"""
import time
import os
import sys
import platform
import hashlib
import torch
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer
from src.transformer_model import MultiLabelSBDHTransformer
from src.transformer_dataset import decode_predictions_to_spans
from src.experimental_decoders import decode_strategy_b_active_threshold
from src.experimental_boundary_snapping import snap_span_boundaries, fuse_same_category_spans

def hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

SAMPLE_NOTES = [
    (
        "Example 1",
        "Patient suffers from chronic back pain and cannot afford to continue physical therapy. "
        "Reports missing appointments due to lack of copay funds."
    ),
    (
        "Example 2",
        "Patient recently experienced job loss and faced foreclosure on his apartment. "
        "Currently staying in temporary homeless shelter and relies on local emergency food banks."
    ),
    (
        "Example 3",
        "Patient uses a motorized wheelchair due to severe neuropathy. Expresses profound loneliness "
        "and social isolation living alone, with a past history of heavy alcohol abuse."
    ),
    (
        "Example 4",
        "Discharged home after hospitalization. Patient lives alone with no transportation and reports food insecurity."
    ),
]

def profile():
    colab_dir = os.path.join(PROJECT_ROOT, "evernorth_m3_results", "models", "sbdh_transformer")
    weights_path = os.path.join(colab_dir, "best_model.pt")
    ckpt_hash = hash_file(weights_path)

    print("=" * 80, flush=True)
    print("STREAMLIT INFERENCE PATH RUNTIME PROFILING", flush=True)
    print("=" * 80, flush=True)
    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})", flush=True)
    print(f"Processor: {platform.processor()}", flush=True)
    print(f"Python Version: {platform.python_version()}", flush=True)
    print(f"PyTorch Version: {torch.__version__} (Threads: {torch.get_num_threads()})", flush=True)
    print(f"Device: CPU", flush=True)
    print(f"Checkpoint SHA-256: {ckpt_hash}", flush=True)
    print("-" * 80, flush=True)

    # 1. Model & Tokenizer Initialization (Cached once per Streamlit process)
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(colab_dir)
    model = MultiLabelSBDHTransformer.from_pretrained(colab_dir)
    model.eval()
    model.to("cpu")
    model_load_ms = (time.perf_counter() - t0) * 1000
    print(f"Model & Tokenizer Load Time (cached once): {model_load_ms:.2f} ms", flush=True)

    # Warmup
    warmup_text = "Patient has severe pain."
    warm_enc = tokenizer(warmup_text, max_length=256, padding="max_length", truncation=True, return_offsets_mapping=True, return_tensors="pt")
    with torch.no_grad():
        _ = model(input_ids=warm_enc["input_ids"], attention_mask=warm_enc["attention_mask"])

    # 2. Benchmark Baseline Path (Strategy A Control: torch.no_grad + Strategy A)
    base_latencies = []
    base_spans_total = 0
    N_RUNS = 10

    for name, text in SAMPLE_NOTES:
        for run_idx in range(N_RUNS):
            t_start = time.perf_counter()
            encoded = tokenizer(text, max_length=256, padding="max_length", truncation=True, return_offsets_mapping=True, return_tensors="pt")
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            offset_mapping = encoded["offset_mapping"].squeeze(0).cpu().numpy()
            active_mask = np.zeros(len(offset_mapping), dtype=float)
            for t in range(len(offset_mapping)):
                if offset_mapping[t][0] != 0 or offset_mapping[t][1] != 0:
                    active_mask[t] = 1.0

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs["logits"].squeeze(0).cpu().numpy()

            spans = decode_predictions_to_spans(logits, active_mask, offset_mapping, text, threshold=0.30)
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            base_latencies.append(elapsed_ms)
            if run_idx == 0:
                base_spans_total += len(spans)

    # 3. Benchmark Arm 6A Promoted Path (torch.inference_mode + Strategy B + Snap + Fuse)
    arm6a_latencies = []
    arm6a_spans_total = 0

    for name, text in SAMPLE_NOTES:
        for run_idx in range(N_RUNS):
            t_start = time.perf_counter()
            encoded = tokenizer(text, max_length=256, padding="max_length", truncation=True, return_offsets_mapping=True, return_tensors="pt")
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            offset_mapping = encoded["offset_mapping"].squeeze(0).cpu().numpy()
            active_mask = np.zeros(len(offset_mapping), dtype=float)
            for t in range(len(offset_mapping)):
                if offset_mapping[t][0] != 0 or offset_mapping[t][1] != 0:
                    active_mask[t] = 1.0

            with torch.inference_mode():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs["logits"].squeeze(0).cpu().numpy()

            raw_spans = decode_strategy_b_active_threshold(logits, active_mask, offset_mapping, text, threshold=0.30)
            snapped = set(snap_span_boundaries(p, text)[0] for p in raw_spans)
            fused, _, _ = fuse_same_category_spans(snapped, text, max_gap=2, allowed_connectives=set())
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            arm6a_latencies.append(elapsed_ms)
            if run_idx == 0:
                arm6a_spans_total += len(fused)

    base_mean = float(np.mean(base_latencies))
    base_std = float(np.std(base_latencies))
    base_min = float(np.min(base_latencies))
    base_max = float(np.max(base_latencies))

    arm6a_mean = float(np.mean(arm6a_latencies))
    arm6a_std = float(np.std(arm6a_latencies))
    arm6a_min = float(np.min(arm6a_latencies))
    arm6a_max = float(np.max(arm6a_latencies))

    diff_ms = arm6a_mean - base_mean
    diff_pct = (diff_ms / base_mean) * 100

    print("-" * 80, flush=True)
    print(f"Baseline Path (Strategy A Control, torch.no_grad, batch_size=1):", flush=True)
    print(f"  Mean Latency: {base_mean:.2f} ms +/- {base_std:.2f} ms (Min: {base_min:.2f} ms, Max: {base_max:.2f} ms)", flush=True)
    print(f"  Spans Extracted (4 sample notes): {base_spans_total}", flush=True)
    print("-" * 80, flush=True)
    print(f"Promoted Arm 6A Path (Strategy B + Snap + Fuse, torch.inference_mode, batch_size=1):", flush=True)
    print(f"  Mean Latency: {arm6a_mean:.2f} ms +/- {arm6a_std:.2f} ms (Min: {arm6a_min:.2f} ms, Max: {arm6a_max:.2f} ms)", flush=True)
    print(f"  Spans Extracted (4 sample notes): {arm6a_spans_total} (+{arm6a_spans_total - base_spans_total} spans)", flush=True)
    print("-" * 80, flush=True)
    print(f"Inference Latency Delta: {diff_ms:+.2f} ms ({diff_pct:+.2f}%)", flush=True)
    print(f"Model Load Included in Timing? NO (Model is cached once at app startup in {model_load_ms:.2f} ms)", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    profile()
