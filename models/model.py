"""NACFN model for fasting blood glucose regression."""

from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def sparsemax(scores: Tensor) -> Tensor:
    """Project scores onto the probability simplex along the last axis."""
    logits = scores.float() if scores.dtype in (torch.float16, torch.bfloat16) else scores
    logits = logits - logits.amax(dim=-1, keepdim=True)
    ranked = logits.sort(dim=-1, descending=True).values
    cumulative = ranked.cumsum(dim=-1) - 1
    ranks = torch.arange(1, scores.size(-1) + 1, device=scores.device, dtype=logits.dtype)
    support = (ranks * ranked > cumulative).sum(dim=-1, keepdim=True)
    threshold = cumulative.gather(-1, support - 1) / support
    return (logits - threshold).clamp_min(0).to(scores.dtype)


class ScalarFeatureTokenizer(nn.Module):
    """Represent each scalar feature with its own learnable token."""

    def __init__(self, num_features: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_features, hidden_dim))
        self.bias = nn.Parameter(torch.empty(num_features, hidden_dim))
        self.missing = nn.Parameter(torch.empty(num_features, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.weight)
        nn.init.normal_(self.bias, std=0.02)
        nn.init.normal_(self.missing, std=0.02)

    def forward(self, values: Tensor) -> Tensor:
        missing = torch.isnan(values)
        values = torch.nan_to_num(values)
        tokens = values.unsqueeze(-1) * self.weight + self.bias
        tokens = torch.where(missing.unsqueeze(-1), self.missing, tokens)
        return self.dropout(self.norm(tokens))


class GatedResidualNetwork(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.content = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x + self.gate(x) * self.content(x))


class StaticEncoder(nn.Module):
    def __init__(self, num_features: int, dim: int, heads: int, layers: int, dropout: float):
        super().__init__()
        self.tokenizer = ScalarFeatureTokenizer(num_features, dim, dropout)
        block = nn.TransformerEncoderLayer(
            dim, heads, dim * 4, dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(block, layers, norm=nn.LayerNorm(dim))
        self.query = nn.Parameter(torch.randn(dim) * 0.02)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        tokens = self.encoder(self.tokenizer(x))
        scores = torch.einsum("bfd,d->bf", tokens, self.query) / self.query.numel() ** 0.5
        importance = sparsemax(scores)
        return torch.einsum("bf,bfd->bd", importance, tokens), importance


class IntraDayNutritionEncoder(nn.Module):
    """Contextualize nutrient tokens separately for each day and pool them."""

    def __init__(self, input_dim: int, dim: int, heads: int, layers: int, max_length: int, dropout: float):
        super().__init__()
        self.tokenizer = ScalarFeatureTokenizer(input_dim, dim, dropout)
        self.position = nn.Parameter(torch.randn(1, max_length, dim) * 0.02)
        block = nn.TransformerEncoderLayer(
            dim, heads, dim * 4, dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(block, layers, norm=nn.LayerNorm(dim))
        self.max_length = max_length

    def forward(self, x: Tensor, valid_mask: Tensor) -> Tensor:
        batch, days, nutrients = x.shape
        if not 1 <= days <= self.max_length:
            raise ValueError(f"expected sequence length between 1 and {self.max_length}, got {days}")
        values = x.masked_fill(~valid_mask.unsqueeze(-1), 0).reshape(batch * days, nutrients)
        # Each day is a separate attention batch item. The sequence axis contains only nutrients.
        tokens = self.encoder(self.tokenizer(values))
        daily = tokens.mean(dim=1).reshape(batch, days, -1) + self.position[:, :days]
        return daily.masked_fill(~valid_mask.unsqueeze(-1), 0)


class ProgressiveCrossFusion(nn.Module):
    def __init__(self, dim: int, heads: int, layers: int, dropout: float):
        super().__init__()
        self.cross_attention = nn.ModuleList(
            [nn.MultiheadAttention(dim, heads, dropout, batch_first=True) for _ in range(layers)]
        )
        self.residual = nn.ModuleList([GatedResidualNetwork(dim, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, static: Tensor, dynamic: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor]:
        query = static.unsqueeze(1)
        has_history = valid_mask.any(dim=1, keepdim=True)
        padding = ~valid_mask
        # Allow a zero placeholder for empty histories to keep attention finite.
        padding[:, -1:] &= has_history
        attention = None
        for cross, residual in zip(self.cross_attention, self.residual):
            update, attention = cross(query, dynamic, dynamic, key_padding_mask=padding, need_weights=True)
            update = update.masked_fill(~has_history.unsqueeze(-1), 0)
            query = residual(query + update)
        return self.norm(query.squeeze(1)), attention.squeeze(1).masked_fill(~valid_mask, 0)


@dataclass
class ModelOutput:
    prediction: Tensor
    loss: Tensor | None
    mse_loss: Tensor | None
    counterfactual_loss: Tensor
    static_attention: Tensor
    temporal_attention: Tensor
    history_gate: Tensor


class NACFN(nn.Module):
    """Dual-stream regression model with interpretable fusion and CF consistency."""

    def __init__(
        self,
        static_dim: int = 32,
        nutrition_dim: int = 78,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        sequence_length: int = 16,
        dropout: float = 0.15,
        cf_lambda: float = 0.05,
        perturb_scale: float = 0.05,
    ):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.static_dim, self.nutrition_dim = static_dim, nutrition_dim
        self.cf_lambda, self.perturb_scale = cf_lambda, perturb_scale
        self.static_encoder = StaticEncoder(static_dim, hidden_dim, num_heads, num_layers, dropout)
        self.dynamic_encoder = IntraDayNutritionEncoder(
            nutrition_dim, hidden_dim, num_heads, num_layers, sequence_length, dropout
        )
        self.fusion = ProgressiveCrossFusion(hidden_dim, num_heads, num_layers, dropout)
        self.history_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, hidden_dim)
        )
        self.history_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )

    @classmethod
    def from_args(cls, args) -> "NACFN":
        return cls(
            args.static_dim,
            args.nutrition_dim,
            args.hidden_dim,
            args.num_heads,
            args.num_layers,
            args.sequence_length,
            args.dropout,
            args.cf_lambda,
            args.perturb_scale,
        )

    def _predict(self, static: Tensor, nutrition: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if static.size(-1) != self.static_dim:
            raise ValueError(f"expected {self.static_dim} static features, got {static.size(-1)}")
        expected = self.nutrition_dim + 1
        if nutrition.size(-1) != expected:
            raise ValueError(
                f"expected {expected} longitudinal features (last is historical FBG), got {nutrition.size(-1)}"
            )
        nutrients, historical_fbg = nutrition[..., :-1], nutrition[..., -1:]
        static_summary, static_attention = self.static_encoder(static)
        timeline = self.dynamic_encoder(nutrients, mask)
        fused, temporal_attention = self.fusion(static_summary, timeline, mask)
        days = torch.arange(mask.size(1), device=mask.device)
        last_index = days.masked_fill(~mask, -1).amax(dim=1, keepdim=True)
        last_fbg = historical_fbg.gather(1, last_index.clamp_min(0).unsqueeze(-1)).squeeze(1)
        last_fbg = last_fbg.masked_fill(last_index < 0, 0)
        history = self.history_encoder(last_fbg)
        gate = self.history_gate(torch.cat([fused, history], dim=-1))
        gate = gate.masked_fill(last_index < 0, 0)
        personalized = gate * history + (1.0 - gate) * fused
        prediction = self.head(torch.cat([static_summary, personalized], dim=-1)).squeeze(-1)
        return prediction, static_attention, temporal_attention, gate.squeeze(-1)

    def forward(
        self,
        sample_id: Tensor,
        static: Tensor,
        nutrition: Tensor,
        mask: Tensor | None = None,
        label: Tensor | None = None,
        return_details: bool = False,
    ):
        # Backward compatibility: old calls used (id, static, nutrition, label).
        if label is None and mask is not None and mask.dtype != torch.bool:
            label, mask = mask, None
        if mask is None:
            mask = nutrition.abs().sum(-1).ne(0)
        prediction, static_attn, temporal_attn, gate = self._predict(static, nutrition, mask)
        mse = F.mse_loss(prediction, label.float()) if label is not None else None
        cf_loss = prediction.new_zeros(())
        if self.training and self.cf_lambda > 0:
            perturbed = static + torch.randn_like(static) * self.perturb_scale
            cf_prediction, *_ = self._predict(perturbed, nutrition, mask)
            cf_loss = F.smooth_l1_loss(cf_prediction, prediction.detach())
        loss = None if mse is None else mse + self.cf_lambda * cf_loss
        details = ModelOutput(prediction, loss, mse, cf_loss, static_attn, temporal_attn, gate)
        if return_details:
            return details
        return prediction, loss, mse, cf_loss


# Keep the original import path available for existing callers.
FBGPredictor = NACFN
