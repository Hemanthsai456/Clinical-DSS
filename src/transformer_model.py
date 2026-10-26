"""
Milestone 3: Multi-Label SBDH Transformer Model Architecture.

Architecture:
- Encoder: Pretrained RoBERTa-family backbone (e.g. distilbert/distilroberta-base).
- Classifier: Multi-label token classification head with 45 outputs (15 categories * 3 BIO tags).
- Multi-label loss: Masked Binary Cross-Entropy with Logits (BCEWithLogitsLoss).
  Applies token-level active masks (masking padding & special tokens) and category-level loss masks
  (masking ambiguous annotations, such as ex_no 1192 'pain' candidates).
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, PreTrainedModel, AutoConfig
from typing import Dict, Any, Optional, Tuple


class MultiLabelSBDHTransformer(nn.Module):
    """
    Multi-label Token Classification Transformer for Social Determinants of Health (SBDH).
    Predicts 45 independent binary BIO logits per token (15 SBDH categories x 3 BIO states).
    """
    def __init__(
        self,
        model_name: str = "distilbert/distilroberta-base",
        num_labels: int = 45,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Args:
        - input_ids: (batch_size, seq_len)
        - attention_mask: (batch_size, seq_len)
        - active_mask: (batch_size, seq_len) 1.0 for real text tokens, 0.0 for special/padding
        - targets: (batch_size, seq_len, 45) multi-label binary target indicators
        - loss_mask: (batch_size, seq_len, 15) category-specific supervision masks

        Returns dict containing:
        - logits: (batch_size, seq_len, 45)
        - probabilities: (batch_size, seq_len, 45)
        - loss: scalar tensor if targets is provided, else None
        """
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        sequence_output = encoder_outputs[0]  # (batch_size, seq_len, hidden_size)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # (batch_size, seq_len, 45)
        probs = torch.sigmoid(logits)

        loss = None
        if targets is not None:
            # Per-element binary cross-entropy loss with logits
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # (batch, seq_len, 45)

            # Build supervision mask
            # active_mask is (batch, seq_len)
            # loss_mask is (batch, seq_len, 15)
            if active_mask is None:
                active_mask = attention_mask.float()
            if loss_mask is None:
                loss_mask = torch.ones((input_ids.size(0), input_ids.size(1), 15), device=input_ids.device)

            # Expand loss_mask to 45 dimensions (3 per category)
            # Repeat each category's mask for its B, I, and O labels
            expanded_cat_mask = torch.repeat_interleave(loss_mask, 3, dim=-1)  # (batch, seq_len, 45)
            token_active_mask = active_mask.unsqueeze(-1)                     # (batch, seq_len, 1)

            full_mask = token_active_mask * expanded_cat_mask                # (batch, seq_len, 45)
            masked_loss = bce * full_mask
            loss = masked_loss.sum() / (full_mask.sum() + 1e-8)

        return {
            "logits": logits,
            "probabilities": probs,
            "loss": loss,
        }

    def save_pretrained(self, save_directory: str):
        """Save model weights and configuration to directory."""
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)

    @classmethod
    def from_pretrained(cls, load_directory: str, model_name: str = "distilbert/distilroberta-base"):
        """Load model weights and configuration from directory."""
        model = cls(model_name=model_name)
        model_path = os.path.join(load_directory, "pytorch_model.bin")
        if not os.path.exists(model_path):
            model_path = os.path.join(load_directory, "best_model.pt")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict)
        return model

