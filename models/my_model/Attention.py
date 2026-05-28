"""
Attention blocks with a simplified forward API.
- forward only needs satellite / timeseries, no longer passes pos_embedding or mask.
- Rotary positional encoding is dynamically generated internally by sequence length (can be disabled).

Classes
-------
- SatelliteMultiHeadAttentionSimple: self-attention (satellite -> satellite)
- FeatureFusionMultiHeadAttentionSimple: cross-attention (timeseries -> satellite)

Usage example
--------
>>> attn = SatelliteMultiHeadAttentionSimple(n_head=4, d_model=16, use_rotary=True)
>>> y, w = attn(torch.randn(2, 10, 16))
>>> cross = FeatureFusionMultiHeadAttentionSimple(n_head=4, d_model=16)
>>> out, w = cross(satellite=torch.randn(2, 12, 16), timeseries=torch.randn(2, 6, 16))
"""
from __future__ import annotations
from typing import Tuple
import math

import torch
import torch.nn as nn
from einops import rearrange


def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    x = rearrange(x, "... (d j) -> ... d j", j=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d j -> ... (d j)")


def build_rotary_sin_cos(T: int, D: int, device: torch.device, base: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate sin/cos of (1, T, D); D will be automatically truncated to even."""
    D = (D // 2) * 2  # ensure even
    pos = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1)  # (T,1)
    inv = torch.exp(torch.arange(0, D, 2, device=device, dtype=torch.float32) * (-math.log(base) / D))  # (D/2)
    angles = pos * inv  # (T, D/2)
    sin = torch.sin(torch.cat([angles, angles], dim=-1)).unsqueeze(0)[:, :, :D]  # (1,T,D)
    cos = torch.cos(torch.cat([angles, angles], dim=-1)).unsqueeze(0)[:, :, :D]
    return sin, cos


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # q,k,v: (B,H,L, Dh)
        Dh = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)  # (B,H,Lq,Lk)
        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, v)
        return out, attn


class SatelliteMultiHeadAttentionSimple(nn.Module):
    """Self-attention over satellite tokens (B, T, D).

    Args:
        n_head: number of heads.
        d_model: model dim (divisible by n_head).
        dropout: dropout prob.
        use_rotary: whether to apply rotary PE internally.
        rotary_frac: fraction of head dim to apply rotary on (0~1]; 1.0 = full Dh.
    """

    def __init__(self, n_head: int = 4, d_model: int = 16, dropout: float = 0.0, use_rotary: bool = True, rotary_frac: float = 1.0) -> None:
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        assert 0.0 < rotary_frac <= 1.0
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_model // n_head
        self.use_rotary = use_rotary
        self.rotary_dim = int(self.d_head * rotary_frac) // 2 * 2  # even

        self.norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.attn = ScaledDotProductAttention(dropout)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rotary_dim == 0:
            return q, k
        _, _, T, _ = q.shape
        sin, cos = build_rotary_sin_cos(T, self.rotary_dim, device=q.device)
        sin = sin.unsqueeze(1)  # (1,1,T,Dr)
        cos = cos.unsqueeze(1)
        Dr = self.rotary_dim
        (q_rot, q_pass), (k_rot, k_pass) = (q[..., :Dr], q[..., Dr:]), (k[..., :Dr], k[..., Dr:])
        q_rot = (q_rot * cos) + (rotate_every_two(q_rot) * sin)
        k_rot = (k_rot * cos) + (rotate_every_two(k_rot) * sin)
        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1)
        return q, k

    def forward(self, satellite: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = satellite.shape
        x = self.norm(satellite)
        q = self.q_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)

        if self.use_rotary:
            q, k = self._apply_rotary(q, k)

        out, attn = self.attn(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)
        out = self.drop(out)
        return out, attn


class FeatureFusionMultiHeadAttention(nn.Module):
    """Cross-attention: timeseries queries attend to satellite keys/values.

    Inputs:
        satellite: (B, Ts, D)
        timeseries: (B, Tt, D)
    Returns:
        outputs: (B, Tt, D), attn: (B, H, Tt, Ts)
    """

    def __init__(self, n_head: int = 4, d_model: int = 32, dropout: float = 0.0, use_rotary: bool = True, rotary_frac: float = 1.0) -> None:
        super().__init__()
        assert d_model % n_head == 0
        assert 0.0 < rotary_frac <= 1.0
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_model // n_head
        self.use_rotary = use_rotary
        self.rotary_dim = int(self.d_head * rotary_frac) // 2 * 2

        self.sat_norm = nn.LayerNorm(d_model)
        self.ts_norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.attn = ScaledDotProductAttention(dropout)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def _apply_rotary_dual(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rotary_dim == 0:
            return q, k
        _, _, Tt, _ = q.shape
        _, _, Ts, _ = k.shape
        sin_ts, cos_ts = build_rotary_sin_cos(Tt, self.rotary_dim, q.device)
        sin_sat, cos_sat = build_rotary_sin_cos(Ts, self.rotary_dim, k.device)
        sin_ts, cos_ts = sin_ts.unsqueeze(1), cos_ts.unsqueeze(1)  # (1,1,Tt,Dr)
        sin_sat, cos_sat = sin_sat.unsqueeze(1), cos_sat.unsqueeze(1)  # (1,1,Ts,Dr)
        Dr = self.rotary_dim
        q_rot, q_pass = q[..., :Dr], q[..., Dr:]
        k_rot, k_pass = k[..., :Dr], k[..., Dr:]
        q_rot = (q_rot * cos_ts) + (rotate_every_two(q_rot) * sin_ts)
        k_rot = (k_rot * cos_sat) + (rotate_every_two(k_rot) * sin_sat)
        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1)
        return q, k

    def forward(self, satellite: torch.Tensor, timeseries: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, Ts, D = satellite.shape
        B2, Tt, D2 = timeseries.shape
        assert B == B2 and D == D2

        sat = self.sat_norm(satellite)
        ts = self.ts_norm(timeseries)
        q = self.q_proj(ts).view(B, Tt, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(sat).view(B, Ts, self.n_head, self.d_head).transpose(1, 2)
        v = self.v_proj(sat).view(B, Ts, self.n_head, self.d_head).transpose(1, 2)

        if self.use_rotary:
            q, k = self._apply_rotary_dual(q, k)

        out, attn = self.attn(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, Tt, D)
        out = self.out_proj(out)
        out = self.drop(out)
        return out.mean(dim=1)



def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_feature_fusion_cross_attention_simple():
    torch.manual_seed(0)
    B, Ts, Tt, D, H = 2, 12, 6, 16, 4
    dev = _device()

    sat = torch.randn(B, Ts, D, device=dev, requires_grad=True)
    ts  = torch.randn(B, Tt, D, device=dev, requires_grad=True)

    # rotary full
    cross_full = FeatureFusionMultiHeadAttention(n_head=H, d_model=D, dropout=0.1, use_rotary=True, rotary_frac=1.0).to(dev)
    out1, w1 = cross_full(satellite=sat, timeseries=ts)
    assert out1.shape == (B, Tt, D)
    assert w1.shape == (B, H, Tt, Ts)

    # rotary half dims
    cross_half = FeatureFusionMultiHeadAttention(n_head=H, d_model=D, dropout=0.1, use_rotary=True, rotary_frac=0.5).to(dev)
    out2, w2 = cross_half(satellite=sat.detach().clone().requires_grad_(True), timeseries=ts.detach().clone().requires_grad_(True))
    assert out2.shape == (B, Tt, D)
    assert w2.shape == (B, H, Tt, Ts)

    # backward check
    loss = out1.pow(2).mean() + out2.pow(2).mean()
    loss.backward()
    assert sat.grad is not None and ts.grad is not None


if __name__ == "__main__":
    test_feature_fusion_cross_attention_simple()
    print("✓ feature-fusion cross-attention (simple) ok")
    print("All minimal simple-API tests passed.")
