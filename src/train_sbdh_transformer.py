"""
Milestone 3: SBDH Multi-Label Transformer Training Pipeline.

Features:
- Deterministic random seed initialization for reproducibility.
- Resource-conscious training loop supporting full GPU execution or lightweight CPU execution.
- Multi-label Binary Cross-Entropy loss with token-level active masking and category-level loss masking
  (handles ungrounded ambiguous training annotations like ex_no 1192).
- Validation tracking on synth_sbdh_val.csv at each epoch.
- Best model checkpointing based on Validation relaxed span F1.
- Logging of training curves and metric progression.
- Clean export of model weights and tokenizer to models/sbdh_transformer/.
"""
import os
import sys
import argparse
import time
import json
import random
from typing import Dict, Any, List, Optional, Tuple, Set
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loaders import load_synth_sbdh
from src.transformer_dataset import SynthSBDHDataset, SBDH_CATEGORIES
from src.transformer_model import MultiLabelSBDHTransformer
from src.evaluate_sbdh_transformer import evaluate_transformer_on_dataset
from src.utils import save_json, NpEncoder


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    active_mask = torch.stack([b["active_mask"] for b in batch])
    targets = torch.stack([b["targets"] for b in batch])
    loss_mask = torch.stack([b["loss_mask"] for b in batch])
    offset_mapping = torch.stack([b["offset_mapping"] for b in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "active_mask": active_mask,
        "targets": targets,
        "loss_mask": loss_mask,
        "offset_mapping": offset_mapping,
        "raw_samples": batch,
    }


def train_sbdh_transformer(
    model_name: str = "distilbert/distilroberta-base",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 3e-5,
    max_length: int = 256,
    sample_limit: int = 0,
    output_dir: str = "models/sbdh_transformer",
    seed: int = 42,
    device: str = "auto",
) -> Dict[str, Any]:
    """
    Train and validate MultiLabelSBDHTransformer.
    """
    set_seed(seed)

    if device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = device

    print("=" * 80)
    print("TRACK B: MULTI-LABEL TRANSFORMER SBDH EXTRACTION TRAINING")
    print("=" * 80)
    print(f"Device: {dev.upper()}")
    print(f"Base Model: {model_name}")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Learning Rate: {learning_rate}")
    print(f"Max Length: {max_length} | Seed: {seed}")
    if sample_limit > 0:
        print(f"Resource Constraint: Training subset limited to {sample_limit} samples.")

    # 1. Load Synth-SBDH partitions
    dfs, data_dir = load_synth_sbdh()
    train_df = dfs["train"]
    val_df = dfs["val"]

    if sample_limit > 0:
        train_df = train_df.iloc[:sample_limit].copy()
        val_df = val_df.iloc[:min(len(val_df), max(50, sample_limit // 4))].copy()

    print(f"Training notes: {len(train_df):,d} | Validation notes: {len(val_df):,d}")

    # 2. Tokenizer and Datasets
    print("\nInitializing FastTokenizer and encoding dataset partitions...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = SynthSBDHDataset(train_df, tokenizer, max_length=max_length, is_training=True)
    val_dataset = SynthSBDHDataset(val_df, tokenizer, max_length=max_length, is_training=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # 3. Model & Optimizer
    print(f"Building MultiLabelSBDHTransformer with 45 multi-label BIO heads...")
    model = MultiLabelSBDHTransformer(model_name=model_name, num_labels=45, dropout_rate=0.1)
    model.to(dev)

    # Optimizer with weight decay
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # 4. Training Loop
    os.makedirs(output_dir, exist_ok=True)
    training_history = []
    best_val_f1 = -1.0
    start_time = time.time()

    print("\n" + "-" * 80)
    print("BEGINNING TRAINING WITH VALIDATION TRACKING")
    print("-" * 80)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        total_train_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(dev)
            attention_mask = batch["attention_mask"].to(dev)
            active_mask = batch["active_mask"].to(dev)
            targets = batch["targets"].to(dev)
            loss_mask = batch["loss_mask"].to(dev)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                active_mask=active_mask,
                targets=targets,
                loss_mask=loss_mask,
            )

            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_train_loss / max(1, num_batches)
        epoch_elapsed = time.time() - epoch_start

        # Validation evaluation
        print(f"\n[Epoch {epoch}/{epochs}] Evaluating on validation set ({len(val_df)} notes)...")
        val_eval = evaluate_transformer_on_dataset(
            model=model,
            df=val_df,
            tokenizer=tokenizer,
            device=dev,
            max_length=max_length,
            dataset_name="validation",
        )

        val_exact_f1 = val_eval["span_level_exact_matching"]["resolvable_subset"]["f1_score"]
        val_rel_f1 = val_eval["span_level_relaxed_matching"]["resolvable_subset"]["f1_score"]
        val_micro_f1 = val_eval["aggregate_classification_metrics"]["micro"]["f1_score"]

        print(
            f"  Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Exact F1: {val_exact_f1:.4f} | Val Relaxed F1: {val_rel_f1:.4f} | "
            f"Val Micro F1: {val_micro_f1:.4f} | Time: {epoch_elapsed:.1f}s"
        )

        training_history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_exact_f1": round(val_exact_f1, 4),
            "val_relaxed_f1": round(val_rel_f1, 4),
            "val_micro_f1": round(val_micro_f1, 4),
            "epoch_duration_sec": round(epoch_elapsed, 1),
        })

        # Save checkpoint if best validation relaxed F1
        if val_rel_f1 > best_val_f1 or epoch == 1:
            best_val_f1 = val_rel_f1
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"  --> Saved new best model checkpoint to {output_dir}/ (Val Relaxed F1: {val_rel_f1:.4f})")

    total_training_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE in {total_training_time:.1f}s | Best Val Relaxed F1: {best_val_f1:.4f}")
    print("=" * 80)

    # Save training curves metadata
    curves_file = os.path.join(output_dir, "training_curves.json")
    save_json({
        "model_name": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "sample_limit": sample_limit,
        "best_val_relaxed_f1": best_val_f1,
        "total_training_duration_sec": round(total_training_time, 1),
        "history": training_history,
    }, curves_file)
    print(f"Training curves saved to: {curves_file}")

    return {
        "output_dir": output_dir,
        "best_val_f1": best_val_f1,
        "training_history": training_history,
        "total_duration_sec": total_training_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Multi-Label SBDH Transformer")
    parser.add_argument("--model_name", type=str, default="distilbert/distilroberta-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--sample_limit", type=int, default=0, help="0 for full dataset, >0 for subset")
    parser.add_argument("--output_dir", type=str, default="models/sbdh_transformer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    train_sbdh_transformer(
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        sample_limit=args.sample_limit,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
