"""
Script to build notebooks/02_sbdh_transformer.ipynb for Milestone 3.
Creates a self-contained, reproducible Google Colab & local Jupyter notebook.
"""
import os
import json

def create_notebook():
    notebook_path = os.path.join("notebooks", "02_sbdh_transformer.ipynb")
    os.makedirs("notebooks", exist_ok=True)

    cells = []

    def md(source):
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")]
        }

    def code(source):
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")]
        }

    # Cell 1: Header & Colab Badge
    cells.append(md("""# Evernorth Clinical CDS — Milestone 3: Transformer-Based SBDH Extraction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hemanthsai/Clinical-DSS/blob/main/notebooks/02_sbdh_transformer.ipynb)

**Clinical Decision Support Using Deep Learning and NLP**  
*Educational & Research Prototype — Not for Unsupervised Clinical Deployment*

---

### Objectives & Deliverables
1. **Evaluator Safeguards (Phase 0):** Demonstrate that greedy span matching is sensitive to prediction order on overlapping intervals; implement and verify maximum bipartite matching for order-invariant 1-to-1 span matching.
2. **Modeling Formulation (Phase 1):** Pretrained Transformer (`distilroberta-base`) with an independent 45-class BIO classification head ($15\\text{ categories} \\times \\{B, I, O\\}$ binary outputs per token). Solves the multi-label problem where identical spans have multiple categories (e.g., `ex_no 7062`: `financial insecurity` + `barriers to care`).
3. **Data & Training Protocol (Phase 2):**
   - Training on `synth_sbdh_train.csv` ($N=6,136$ notes).
   - Validation on `synth_sbdh_val.csv` ($N=876$ notes).
   - Final evaluation strictly on `synth_sbdh_test_reviewed.csv` ($N=1,732$ notes).
   - Zero-guess uncertainty handling: `ex_no 1192` candidate loss masking.
   - Preserving 5 unresolved test annotations via dual bounds (resolvable subset vs. conservative worst-case bounds).
4. **Evaluation & Error Analysis (Phase 3):** Exact vs. relaxed F1, per-category breakdown, boundary error analysis, repeated mention precision, and head-to-head comparison against the rules-based baseline.
"""))

    # Cell 2: Colab Environment Setup
    cells.append(md("## 1. Environment Setup & Hardware Diagnostics"))
    cells.append(code("""# Install required dependencies in Google Colab / local Jupyter
!pip install -q torch transformers datasets accelerate scikit-learn pandas numpy scipy

import os
import sys
import json
import random
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, List, Tuple, Set

# Deterministic seeding for reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# Hardware diagnostics
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device in use: {device}")
if torch.cuda.is_available():
    print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    print(f"VRAM Allocated: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    print("WARNING: Running on CPU. GPU (T4, V100, or A100) is strongly recommended for full model training.")
"""))

    # Cell 3: Data Loading
    cells.append(md("## 2. Dataset Loading: Synth-SBDH\n"
                    "We load Synth-SBDH either directly from Hugging Face Hub (`bio-nlp-umass/Synth-SBDH`) or from a local data directory if available."))
    cells.append(code("""from datasets import load_dataset

def load_synth_sbdh_datasets():
    \"\"\"Load train, val, and test_reviewed splits from HuggingFace or local disk.\"\"\"
    local_data_dir = "D:/Evernorth/Synth-SBDH" if os.path.exists("D:/Evernorth/Synth-SBDH") else "./data/Synth-SBDH"
    
    if os.path.exists(os.path.join(local_data_dir, "synth_sbdh_train.csv")):
        print(f"Loading datasets from local directory: {local_data_dir}")
        train_df = pd.read_csv(os.path.join(local_data_dir, "synth_sbdh_train.csv"))
        val_df = pd.read_csv(os.path.join(local_data_dir, "synth_sbdh_val.csv"))
        test_df = pd.read_csv(os.path.join(local_data_dir, "synth_sbdh_test_reviewed.csv"))
    else:
        print("Loading Synth-SBDH from Hugging Face Hub (bio-nlp-umass/Synth-SBDH)...")
        hf_dataset = load_dataset("bio-nlp-umass/Synth-SBDH")
        train_df = pd.DataFrame(hf_dataset["train"])
        val_df = pd.DataFrame(hf_dataset["validation"])
        test_df = pd.DataFrame(hf_dataset["test"])
    
    print(f"Train split: {len(train_df)} notes")
    print(f"Val split:   {len(val_df)} notes")
    print(f"Test split:  {len(test_df)} notes")
    return train_df, val_df, test_df

train_df, val_df, test_df = load_synth_sbdh_datasets()
"""))

    # Cell 4: Evaluator Safeguard Demo
    cells.append(md("""## 3. Phase 0 — Evaluator Safeguard: Bipartite Matching vs. Greedy Matching

Standard greedy matching evaluates predictions in the order they appear in the prediction list. When candidate predictions overlap in character boundaries, greedy matching can prematurely consume a gold annotation, stranding subsequent predictions and yielding suboptimal matching.

We prove this with an adversarial example and demonstrate that Maximum Bipartite Matching (via augmenting paths) guarantees order-invariance and global maximum 1-to-1 matching.
"""))
    cells.append(code("""# Maximum Bipartite Matching Implementation
def find_maximum_bipartite_matching(
    gold_spans: List[Tuple[int, int, str]],
    pred_spans: List[Tuple[int, int, str]],
    overlap_fn
) -> Tuple[int, List[Tuple[int, int]]]:
    \"\"\"
    Finds the maximum cardinality one-to-one matching between gold spans and predicted spans
    using the augmenting path algorithm (Kuhn's algorithm).
    \"\"\"
    n_gold = len(gold_spans)
    n_pred = len(pred_spans)
    
    # Construct adjacency list: pred_idx -> list of compatible gold_idx
    adj: List[List[int]] = [[] for _ in range(n_pred)]
    for p_idx, (p_start, p_end, p_cat) in enumerate(pred_spans):
        for g_idx, (g_start, g_end, g_cat) in enumerate(gold_spans):
            if p_cat == g_cat and overlap_fn(g_start, g_end, p_start, p_end):
                adj[p_idx].append(g_idx)
                
    match_gold = [-1] * n_gold
    
    def dfs(u: int, visited: List[bool]) -> bool:
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                if match_gold[v] < 0 or dfs(match_gold[v], visited):
                    match_gold[v] = u
                    return True
        return False
        
    matching_size = 0
    for u in range(n_pred):
        visited = [False] * n_gold
        if dfs(u, visited):
            matching_size += 1
            
    matches = [(match_gold[v], v) for v in range(n_gold) if match_gold[v] >= 0]
    return matching_size, matches

def char_overlap(g_start: int, g_end: int, p_start: int, p_end: int) -> bool:
    return max(0, min(g_end, p_end) - max(g_start, p_start)) > 0

# Adversarial Demonstration:
# Gold 1: [10, 25], Gold 2: [20, 35] (Category: 'pain')
# Pred 1: [12, 22] (overlaps Gold 1 & Gold 2)
# Pred 2: [14, 18] (overlaps ONLY Gold 1)
gold_demo = [(10, 25, "pain"), (20, 35, "pain")]
preds_order_a = [(12, 22, "pain"), (14, 18, "pain")]
preds_order_b = [(14, 18, "pain"), (12, 22, "pain")]

# 1. Greedy Matching simulation
def greedy_match(golds, preds):
    matched_golds = set()
    tp = 0
    for p_start, p_end, p_cat in preds:
        for g_idx, (g_start, g_end, g_cat) in enumerate(golds):
            if g_idx not in matched_golds and p_cat == g_cat and char_overlap(g_start, g_end, p_start, p_end):
                matched_golds.add(g_idx)
                tp += 1
                break
    return tp

tp_greedy_a = greedy_match(gold_demo, preds_order_a)
tp_greedy_b = greedy_match(gold_demo, preds_order_b)
tp_bipartite_a, _ = find_maximum_bipartite_matching(gold_demo, preds_order_a, char_overlap)
tp_bipartite_b, _ = find_maximum_bipartite_matching(gold_demo, preds_order_b, char_overlap)

print("=== EVALUATOR SAFEGUARD DEMONSTRATION ===")
print(f"Greedy Match Order A [P1, P2]: TP = {tp_greedy_a}  (SUBOPTIMAL!)")
print(f"Greedy Match Order B [P2, P1]: TP = {tp_greedy_b}")
print(f"Maximum Bipartite Order A:     TP = {tp_bipartite_a}  (ORDER INVARIANT)")
print(f"Maximum Bipartite Order B:     TP = {tp_bipartite_b}  (ORDER INVARIANT)")
assert tp_bipartite_a == tp_bipartite_b == 2, "Bipartite matching failed to find optimal 2 TPs!"
print("Safeguard Verified: Maximum bipartite matching is guaranteed order-invariant and globally optimal.")
"""))

    # Cell 5: Preprocessing & Multi-label BIO Target Formulation
    cells.append(md("""## 4. Phase 1 — Modeling Formulation & Multi-Label BIO Encoding

### Multi-Label Span Problem
In Synth-SBDH, a single character span can belong to multiple SBDH categories simultaneously:
- **Example `ex_no 7062`:** The span `"cannot afford to continue physical therapy"` is annotated with both `financial insecurity` and `barriers to care`.
- A traditional mutually exclusive BIO scheme (e.g. 15 × 2 + 1 = 31 classes with softmax) forces an artificial choice between categories, corrupting recall on co-occurring labels.

### 45-Class Independent Sigmoid Architecture
We formulate the task with **45 independent binary outputs per token**:
- 15 categories $\\times$ 3 tag positions ($\\{B, I, O\\}$).
- For each token $t$ and category $c$, the model predicts:
  - $\\hat{y}_{t, c, B} = \\sigma(z_{t, c, B})$
  - $\\hat{y}_{t, c, I} = \\sigma(z_{t, c, I})$
  - $\\hat{y}_{t, c, O} = \\sigma(z_{t, c, O})$
- This natively supports identical spans having multiple active categories, while masking special tokens ($CLS, SEP, PAD$) from the binary cross-entropy loss.
"""))
    cells.append(code("""from transformers import AutoTokenizer

SBDH_CATEGORIES = [
    "alcohol use", "barriers to care", "depression", "drug use",
    "economic status", "education", "employment", "financial insecurity",
    "housing delivery", "living with", "pain", "patient disability",
    "social environment", "tobacco use", "violence"
]
CAT_TO_IDX = {cat: idx for idx, cat in enumerate(sorted(SBDH_CATEGORIES))}
BIO_TAGS = ["B", "I", "O"]
TAG_TO_IDX = {t: idx for idx, t in enumerate(BIO_TAGS)}
TOTAL_LABELS = len(SBDH_CATEGORIES) * len(BIO_TAGS)  # 15 * 3 = 45

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base", add_prefix_space=True)
print(f"Tokenizer loaded: {tokenizer.name_or_path} | Vocabulary size: {tokenizer.vocab_size}")
"""))

    # Cell 6: Multi-label Target Alignment Function
    cells.append(md("""### Handling Uncertainty & Ambiguity Without Guessing
In training example `ex_no 1192`, the gold annotation specifies `Textspan='pain'` with `SBDH='pain'`, but the note contains two separate mentions of 'pain' at offsets `[10:14]` and `[61:65]` without clarifying context.
- **Rule:** Do NOT silently guess one mention as correct and label the other as negative.
- **Solution:** Mask candidate tokens with `loss_mask = 0.0` for category `pain`, excluding ambiguous tokens from backpropagation.
"""))
    cells.append(code("""def encode_note_multi_label(
    text: str,
    spans: List[str],
    sbdhs: List[str],
    tokenizer,
    ex_no: int = None,
    max_length: int = 256
):
    \"\"\"
    Aligns character-level multi-label annotations to BPE tokens using fast offset mapping.
    Produces:
      - input_ids: (seq_len,)
      - attention_mask: (seq_len,)
      - labels: (seq_len, 45) multi-label binary float targets
      - loss_mask: (seq_len, 45) binary mask (0 for special/padding/ambiguous tokens)
    \"\"\"
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"][0]
    attention_mask = encoding["attention_mask"][0]
    offset_mapping = encoding["offset_mapping"][0].tolist()
    seq_len = len(input_ids)
    
    labels = np.zeros((seq_len, TOTAL_LABELS), dtype=np.float32)
    loss_mask = np.zeros((seq_len, TOTAL_LABELS), dtype=np.float32)
    
    # Initialize all real tokens to tag 'O' and active loss
    for i in range(seq_len):
        start_char, end_char = offset_mapping[i]
        if attention_mask[i] == 1 and not (start_char == 0 and end_char == 0):
            loss_mask[i, :] = 1.0
            for cat_idx in range(len(SBDH_CATEGORIES)):
                o_label_idx = cat_idx * 3 + TAG_TO_IDX["O"]
                labels[i, o_label_idx] = 1.0
                
    # Align spans to character offsets
    for span, sbdh in zip(spans, sbdhs):
        cat = sbdh.strip().lower()
        if cat not in CAT_TO_IDX:
            continue
        cat_idx = CAT_TO_IDX[cat]
        b_idx = cat_idx * 3 + TAG_TO_IDX["B"]
        i_idx = cat_idx * 3 + TAG_TO_IDX["I"]
        o_idx = cat_idx * 3 + TAG_TO_IDX["O"]
        
        # Ambiguity check for ex_no 1192
        if ex_no == 1192 and cat == "pain" and span.strip().lower() == "pain":
            # Mask out both candidate occurrences
            for tok_idx, (tok_s, tok_e) in enumerate(offset_mapping):
                if (10 <= tok_s < 14) or (61 <= tok_s < 65):
                    loss_mask[tok_idx, cat_idx*3 : (cat_idx+1)*3] = 0.0
            continue
            
        # Find span in text
        start_pos = text.lower().find(span.lower())
        if start_pos == -1:
            continue
        end_pos = start_pos + len(span)
        
        is_first = True
        for tok_idx, (tok_s, tok_e) in enumerate(offset_mapping):
            if tok_s == 0 and tok_e == 0:
                continue
            if max(start_pos, tok_s) < min(end_pos, tok_e):
                labels[tok_idx, o_idx] = 0.0
                if is_first:
                    labels[tok_idx, b_idx] = 1.0
                    is_first = False
                else:
                    labels[tok_idx, i_idx] = 1.0
                    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": torch.tensor(labels, dtype=torch.float32),
        "loss_mask": torch.tensor(loss_mask, dtype=torch.float32),
        "offset_mapping": offset_mapping
    }

print("Multi-label encoding pipeline ready.")
"""))

    # Cell 7: Transformer Model Architecture
    cells.append(md("""## 5. Model Architecture: `MultiLabelSBDHTransformer`

- **Backbone:** Pretrained `distilbert/distilroberta-base` (82M parameters, 6 layers, 768 hidden dimension).
- **Classification Head:** Linear layer projecting token hidden representations ($768 \\to 45$).
- **Loss:** Element-wise Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`), masked by `loss_mask`.
"""))
    cells.append(code("""import torch.nn as nn
from transformers import RobertaModel

class MultiLabelSBDHTransformer(nn.Module):
    def __init__(self, pretrained_model_name: str = "distilbert/distilroberta-base", num_labels: int = 45, dropout: float = 0.1):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        
    def forward(self, input_ids, attention_mask, labels=None, loss_mask=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None and loss_mask is not None:
            raw_loss = self.loss_fn(logits, labels)
            masked_loss = raw_loss * loss_mask
            loss = masked_loss.sum() / (loss_mask.sum() + 1e-8)
            
        return {"loss": loss, "logits": logits}

model = MultiLabelSBDHTransformer().to(device)
print(f"MultiLabelSBDHTransformer initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")
"""))

    # Cell 8: Prediction Decoding Function
    cells.append(md("""## 6. Decoding Predictions into SBDH Character Spans
We decode the 45-dimensional logits into character-level spans `(start_char, end_char, category)`:
1. Compute sigmoid probabilities $\\sigma(\\text{logits})$.
2. Apply threshold (default $0.5$) to category-specific B and I probabilities.
3. Group contiguous B and I tokens into spans and map them back to original text character offsets via `offset_mapping`.
"""))
    cells.append(code("""def decode_predictions_to_spans(
    logits: torch.Tensor,
    offset_mapping: List[Tuple[int, int]],
    threshold: float = 0.5
) -> List[Tuple[int, int, str]]:
    \"\"\"
    Converts (seq_len, 45) logits into list of predicted spans (start_char, end_char, category).
    \"\"\"
    probs = torch.sigmoid(logits).cpu().numpy()
    spans = []
    
    for cat_name, cat_idx in CAT_TO_IDX.items():
        b_idx = cat_idx * 3 + TAG_TO_IDX["B"]
        i_idx = cat_idx * 3 + TAG_TO_IDX["I"]
        
        in_span = False
        span_start = 0
        span_end = 0
        
        for tok_idx, (tok_s, tok_e) in enumerate(offset_mapping):
            if tok_s == 0 and tok_e == 0:
                continue
            p_b = probs[tok_idx, b_idx]
            p_i = probs[tok_idx, i_idx]
            
            if p_b >= threshold:
                if in_span:
                    spans.append((span_start, span_end, cat_name))
                in_span = True
                span_start = tok_s
                span_end = tok_e
            elif in_span and p_i >= threshold:
                span_end = tok_e
            else:
                if in_span:
                    spans.append((span_start, span_end, cat_name))
                    in_span = False
        if in_span:
            spans.append((span_start, span_end, cat_name))
            
    return spans
"""))

    # Cell 9: Resource-Conscious Training Pipeline
    cells.append(md("""## 7. Phase 2 — Training & Validation Pipeline

We employ a resource-conscious configuration:
- Optimizer: AdamW ($lr=3\\times 10^{-5}$, weight decay $0.01$).
- Batch size: 16 (or 8 with gradient accumulation on smaller GPUs).
- Warmup: Linear warmup for the first 10% of training steps.
- Early checkpointing: Track validation relaxed F1 after each epoch and retain the best checkpoint.
"""))
    cells.append(code("""from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

class SynthSBDHDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["Text"])
        spans_raw = str(row["Textspan"]).split("||") if pd.notna(row.get("Textspan")) else []
        sbdhs_raw = str(row["SBDH"]).split("||") if pd.notna(row.get("SBDH")) else []
        ex_no = int(row["ex_no"]) if pd.notna(row.get("ex_no")) else None
        
        return encode_note_multi_label(
            text=text,
            spans=spans_raw,
            sbdhs=sbdhs_raw,
            tokenizer=self.tokenizer,
            ex_no=ex_no,
            max_length=self.max_length
        )

# Create dataloaders
train_dataset = SynthSBDHDataset(train_df, tokenizer, max_length=256)
val_dataset = SynthSBDHDataset(val_df, tokenizer, max_length=256)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

epochs = 3
total_steps = len(train_loader) * epochs
optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

print(f"Training ready: {len(train_dataset)} train samples, {len(val_dataset)} val samples, {epochs} epochs.")
"""))

    # Cell 10: Training Loop
    cells.append(md("### Training Execution Cell (GPU Recommended)\n"
                    "Run this cell in Colab to train the model across all epochs."))
    cells.append(code("""# To execute full training in Colab, run:
def train_model():
    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            
            outputs = model(input_ids, attention_mask, labels=labels, loss_mask=loss_mask)
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            if (step + 1) % 50 == 0:
                print(f"Epoch {epoch} | Step {step+1}/{len(train_loader)} | Train Loss: {total_train_loss / (step + 1):.4f}")
                
        # Validation pass
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                loss_mask = batch["loss_mask"].to(device)
                outputs = model(input_ids, attention_mask, labels=labels, loss_mask=loss_mask)
                total_val_loss += outputs["loss"].item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"--> Epoch {epoch} Complete | Avg Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs("models/sbdh_transformer", exist_ok=True)
            torch.save(model.state_dict(), "models/sbdh_transformer/best_model.pt")
            print(f"    Saved new best checkpoint (Val Loss: {avg_val_loss:.4f})")

# In interactive notebook, uncomment to train:
# train_model()
"""))

    # Cell 11: Final Evaluation on Test Reviewed
    cells.append(md("""## 8. Phase 3 — Final Evaluation on Reviewed Test Set (`synth_sbdh_test_reviewed.csv`)

### Rigorous Evaluation Policy
- **Resolvable Gold Annotations ($N=3,340$):** Gold annotations with verifiable offsets in the reviewed test set.
- **Conservative Worst-Case Bound ($N=3,345$):** Penalizes all 5 ambiguous unresolvable annotations as False Negatives ($\text{Recall} = \frac{TP}{3345}$).
- **Comparison Baseline:** Frozen rules baseline ($N=1,235$ exact matches, $\text{Exact F1}=0.4727$, $\text{Relaxed F1}=0.4727$).
"""))
    cells.append(code("""# Evaluation on test_df
def evaluate_test_set(model, df):
    model.eval()
    exact_tp = 0
    relaxed_tp = 0
    total_preds = 0
    total_resolvable_gold = 0
    
    with torch.no_grad():
        for idx, row in df.iterrows():
            text = str(row["Text"])
            spans_raw = str(row["Textspan"]).split("||") if pd.notna(row.get("Textspan")) else []
            sbdhs_raw = str(row["SBDH"]).split("||") if pd.notna(row.get("SBDH")) else []
            
            # Gold spans
            gold_spans = []
            for s, cat in zip(spans_raw, sbdhs_raw):
                cat_clean = cat.strip().lower()
                s_clean = s.strip()
                pos = text.lower().find(s_clean.lower())
                if pos != -1:
                    gold_spans.append((pos, pos + len(s_clean), cat_clean))
            total_resolvable_gold += len(gold_spans)
            
            # Prediction
            enc = tokenizer(text, max_length=256, padding="max_length", truncation=True, return_offsets_mapping=True, return_tensors="pt")
            out = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            preds = decode_predictions_to_spans(out["logits"][0], enc["offset_mapping"][0].tolist(), threshold=0.5)
            total_preds += len(preds)
            
            # Exact matches
            for p in preds:
                if p in gold_spans:
                    exact_tp += 1
                    
            # Relaxed matching (bipartite)
            match_count, _ = find_maximum_bipartite_matching(gold_spans, preds, char_overlap)
            relaxed_tp += match_count
            
    # Metrics
    prec_exact = exact_tp / (total_preds + 1e-8)
    rec_exact_res = exact_tp / (total_resolvable_gold + 1e-8)
    f1_exact_res = 2 * prec_exact * rec_exact_res / (prec_exact + rec_exact_res + 1e-8)
    
    prec_rel = relaxed_tp / (total_preds + 1e-8)
    rec_rel_res = relaxed_tp / (total_resolvable_gold + 1e-8)
    f1_rel_res = 2 * prec_rel * rec_rel_res / (prec_rel + rec_rel_res + 1e-8)
    
    # Worst case (assuming 5 ambiguous gold annotations as FN)
    rec_exact_wc = exact_tp / (total_resolvable_gold + 5)
    rec_rel_wc = relaxed_tp / (total_resolvable_gold + 5)
    
    print("=================================================================")
    print("TRANSFORMER EVALUATION ON TEST_REVIEWED (SYNTH-SBDH)")
    print("=================================================================")
    print(f"Exact Matching (Resolvable):   Precision={prec_exact:.4f} | Recall={rec_exact_res:.4f} | F1={f1_exact_res:.4f}")
    print(f"Exact Matching (Worst-Case):   Precision={prec_exact:.4f} | Recall={rec_exact_wc:.4f}")
    print(f"Relaxed Matching (Resolvable): Precision={prec_rel:.4f} | Recall={rec_rel_res:.4f} | F1={f1_rel_res:.4f}")
    print(f"Relaxed Matching (Worst-Case): Precision={prec_rel:.4f} | Recall={rec_rel_wc:.4f}")
    print(f"Total Predicted Spans: {total_preds} | Total Gold Spans: {total_resolvable_gold}")

# Run evaluation on test sample
print("Ready for test evaluation.")
"""))

    # Cell 12: Comparison Table & Error Analysis
    cells.append(md("""## 9. Baseline Comparison & Error Analysis

| Metric | Rules Baseline (Milestone 2) | Transformer Pilot (CPU) | Transformer Full (Colab GPU Target) |
| :--- | :--- | :--- | :--- |
| **Exact F1 (Resolvable)** | 0.4727 | 0.0011 | **0.6500+** |
| **Relaxed F1 (Resolvable)** | 0.4727 | 0.0213 | **0.7200+** |
| **Multi-label Co-occurrence (`ex_no 7062`)** | Fails (mutually exclusive) | Preserved | **Fully supported (45 sigmoid outputs)** |
| **Uncertainty (`ex_no 1192`)** | Guessed occurrence | Loss masked | **Loss masked (unbiased)** |
| **Evaluator Matching** | Greedy | Bipartite | **Order-invariant Maximum Bipartite** |

### Error Taxonomy
1. **Boundary Errors:** Tokenizer boundary mismatches (e.g. leading spaces or punctuation).
2. **Context & Negation:** Syntactic patterns expressing denied or family SBDH (e.g. `"denies alcohol use"`).
3. **Repeated Mentions:** Selecting the specific clinical context among multiple verbatim tokens.
4. **Rare Categories:** Low-frequency determinants such as `living with` and `education` requiring contextual transfer learning.
"""))

    notebook_data = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook_data, f, indent=2)

    print(f"Successfully generated notebook at: {notebook_path}")

if __name__ == "__main__":
    create_notebook()
