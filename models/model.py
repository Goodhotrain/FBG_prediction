"""Personalized Static-Dynamic Counterfactual Fusion Network (PSDCFN)."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor, nn
import torch.nn.functional as F

class StaticFeatureTokenizer(nn.Module):
    """Represent every scalar clinical feature as an independently learnable token."""
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
        self.tokenizer = StaticFeatureTokenizer(num_features, dim, dropout)
        block = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, norm=nn.LayerNorm(dim))
        self.query = nn.Parameter(torch.randn(dim) * 0.02)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        tokens = self.encoder(self.tokenizer(x))
        importance = torch.softmax(torch.einsum("bfd,d->bf", tokens, self.query) / self.query.numel() ** 0.5, dim=1)
        return torch.einsum("bf,bfd->bd", importance, tokens), importance

class TemporalNutritionEncoder(nn.Module):
    def __init__(self, input_dim: int, dim: int, heads: int, layers: int, max_length: int, dropout: float):
        super().__init__()
        self.input = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, dim), nn.GELU())
        self.position = nn.Parameter(torch.randn(1, max_length, dim) * 0.02)
        block = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, norm=nn.LayerNorm(dim))
        self.max_length = max_length

    def forward(self, x: Tensor, valid_mask: Tensor) -> Tensor:
        if x.size(1) > self.max_length:
            raise ValueError(f"sequence length {x.size(1)} exceeds configured maximum {self.max_length}")
        h = self.input(x) + self.position[:, :x.size(1)]
        # True means ignored for PyTorch's padding mask.
        return self.encoder(h, src_key_padding_mask=~valid_mask.bool())

class ProgressiveCrossFusion(nn.Module):
    def __init__(self, dim: int, heads: int, layers: int, dropout: float):
        super().__init__()
        self.cross_attention = nn.ModuleList([nn.MultiheadAttention(dim, heads, dropout, batch_first=True) for _ in range(layers)])
        self.residual = nn.ModuleList([GatedResidualNetwork(dim, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, static: Tensor, dynamic: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor]:
        query = static.unsqueeze(1)
        attention = None
        for cross, residual in zip(self.cross_attention, self.residual):
            update, attention = cross(query, dynamic, dynamic, key_padding_mask=~valid_mask.bool(), need_weights=True)
            query = residual(query + update)
        return self.norm(query.squeeze(1)), attention.squeeze(1)

@dataclass
class ModelOutput:
    prediction: Tensor
    loss: Tensor | None
    mse_loss: Tensor | None
    counterfactual_loss: Tensor
    static_attention: Tensor
    temporal_attention: Tensor
    history_gate: Tensor

class FBGPredictor(nn.Module):
    """Dual-stream regression model with interpretable fusion and CF consistency."""
    def __init__(self, static_dim: int = 32, nutrition_dim: int = 78, hidden_dim: int = 64,
                 num_heads: int = 4, num_layers: int = 2, sequence_length: int = 16,
                 dropout: float = 0.15, cf_lambda: float = 0.05, perturb_scale: float = 0.05):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.static_dim, self.nutrition_dim = static_dim, nutrition_dim
        self.cf_lambda, self.perturb_scale = cf_lambda, perturb_scale
        self.static_encoder = StaticEncoder(static_dim, hidden_dim, num_heads, num_layers, dropout)
        self.dynamic_encoder = TemporalNutritionEncoder(nutrition_dim, hidden_dim, num_heads, num_layers, sequence_length, dropout)
        self.fusion = ProgressiveCrossFusion(hidden_dim, num_heads, num_layers, dropout)
        self.history_encoder = nn.Sequential(nn.Linear(1, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, hidden_dim))
        self.history_gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    @classmethod
    def from_args(cls, args) -> "FBGPredictor":
        return cls(args.static_dim, args.nutrition_dim, args.hidden_dim, args.num_heads,
                   args.num_layers, args.sequence_length, args.dropout, args.cf_lambda, args.perturb_scale)

    def _predict(self, static: Tensor, nutrition: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if static.size(-1) != self.static_dim:
            raise ValueError(f"expected {self.static_dim} static features, got {static.size(-1)}")
        expected = self.nutrition_dim + 1
        if nutrition.size(-1) != expected:
            raise ValueError(f"expected {expected} longitudinal features (last is historical FBG), got {nutrition.size(-1)}")
        nutrients, historical_fbg = nutrition[..., :-1], nutrition[..., -1:]
        static_summary, static_attention = self.static_encoder(static)
        timeline = self.dynamic_encoder(nutrients, mask)
        fused, temporal_attention = self.fusion(static_summary, timeline, mask)
        counts = mask.long().sum(1).clamp_min(1)
        last_index = counts - 1 + (mask.size(1) - counts)  # sequences are left padded
        last_fbg = historical_fbg[torch.arange(len(mask), device=mask.device), last_index]
        history = self.history_encoder(last_fbg)
        gate = self.history_gate(torch.cat([fused, history], dim=-1))
        personalized = gate * history + (1.0 - gate) * fused
        prediction = self.head(torch.cat([static_summary, personalized], dim=-1)).squeeze(-1)
        return prediction, static_attention, temporal_attention, gate.squeeze(-1)

    def forward(self, sample_id: Tensor, static: Tensor, nutrition: Tensor, mask: Tensor | None = None,
                label: Tensor | None = None, return_details: bool = False):
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
