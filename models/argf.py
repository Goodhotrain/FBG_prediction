import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal
from models.model import StaticFusion, DaynamicFusion, BCEWithLogitsLossWithLabelSmoothing

"""
Refactored multimodal fusion modules (cleaned, torch>=1.10 compatible).
- Removed Variable/cuda() direct calls; automatically use input tensor device/dtype.
- Unified softmax with dim parameter; replaced F.tanh/F.sigmoid with torch.tanh/torch.sigmoid (or nn modules).
- Fixed several potential bugs (e.g., fusion overwritten in multiplication model, non-integer Conv1d padding, etc.).
- Added type annotations and more readable structure/comments.
- Avoided using .cuda() in __init__; use xavier_normal_ for parameter initialization.

NOTE: These modules retain APIs mostly consistent with the original code (most return (y, weights or placeholder)).
"""
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Encoders
# -------------------------

class EncoderA(nn.Module):
    """MLP encoder used for audio/video (pre-fusion)."""

    def __init__(self, in_size: int, hidden_size: int, dropout: float = 0.5):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size * 5)
        self.linear_2 = nn.Linear(hidden_size * 5, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.norm(x))
        x = self.drop(F.relu(self.linear_1(x)))
        x = self.drop(F.relu(self.linear_2(x)))
        x = torch.tanh(self.linear_3(x))
        return x


class Encoder5(nn.Module):
    """Deeper encoder variant with intermediate norms."""

    def __init__(self, in_size: int, hidden_size: int, dropout: float = 0.5):
        super().__init__()
        self.norm_h = nn.BatchNorm1d(hidden_size)
        self.norm2 = nn.BatchNorm1d(in_size * 10)
        self.norm3 = nn.BatchNorm1d(hidden_size * 10)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, in_size * 10)
        self.linear_2 = nn.Linear(in_size * 10, hidden_size * 10)
        self.linear_3 = nn.Linear(hidden_size * 10, hidden_size)
        self.linear_4 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = F.leaky_relu(self.norm2(self.drop(self.linear_1(x))))
        y2 = F.leaky_relu(self.norm3(self.drop(self.linear_2(y1))))
        y2 = F.leaky_relu(self.norm_h(self.drop(self.linear_3(y2))))
        y3 = torch.tanh(self.linear_4(y2))
        return y3


class EncoderV(EncoderA):
    """Alias of EncoderA for clarity (video)."""
    pass


class EncoderL3(nn.Module):
    """Text encoder: 1D conv-gating + MLP."""

    def __init__(self, in_size: int, hidden_size: int, dropout: float = 0.5):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_size * 5)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size * 5, hidden_size * 5)
        self.linear_2 = nn.Linear(hidden_size * 5, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)
        kernel_size = 5
        padding = (kernel_size - 1) // 2
        self.gates = nn.Conv1d(1, 5, kernel_size, stride=1, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_size) -> (B, 1, in_size)
        x = x.unsqueeze(1)
        x = self.gates(x)               # (B, 5, in_size)
        x = x.view(x.shape[0], -1)      # (B, 5 * in_size)
        x = self.drop(self.norm(x))
        x = self.drop(F.relu(self.linear_1(x)))
        x = self.drop(F.relu(self.linear_2(x)))
        x = torch.tanh(self.linear_3(x))
        return x


class EncoderL(nn.Module):
    """LSTM-based text encoder."""

    def __init__(self, in_size: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.2, bidirectional: bool = False):
        super().__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, in_size)
        _, (h_n, _) = self.rnn(x)
        h = self.dropout(h_n[-1])  # (B, H)
        y = torch.tanh(self.linear_1(h))
        return y


# -------------------------
# Decoders / Discriminators / Classifiers
# -------------------------

class Decoder2(nn.Module):
    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_size, 512),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 64),
            nn.Dropout(0.5),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, out_size),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        img_flat = self.model(z)
        return img_flat.view(img_flat.shape[0], -1)


class Discriminator(nn.Module):
    def __init__(self, in_size: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_size, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)


class Classifier2(nn.Module):
    def __init__(self, in_size: int, output_dim: int, dropout: float = 0.5):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, output_dim * 10)
        self.linear_2 = nn.Linear(output_dim * 10, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.norm(x))
        x = F.relu(self.linear_1(x))
        x = F.softmax(self.linear_2(x), dim=1)
        return x


class Classifier3(nn.Module):
    def __init__(self, in_size: int, output_dim: int, dropout: float = 0.5):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, in_size)
        self.linear_2 = nn.Linear(in_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.norm(x))
        x = self.drop(torch.tanh(self.linear_1(x)))
        x = F.softmax(self.linear_2(x), dim=1)
        return x


# -------------------------
# Fusion variants
# -------------------------

class Graph11New(nn.Module):
    def __init__(self, in_size: int, output_dim: int, hidden: int = 50, dropout: float = 0.5):
        super().__init__()
        self.norm2 = nn.BatchNorm1d(in_size * 3)
        self.drop = nn.Dropout(p=dropout)

        def fusion_block():
            return nn.Sequential(
                nn.Linear(in_size * 2, 64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(64, in_size),
                nn.Tanh(),
            )

        self.graph_fusion = fusion_block()
        self.graph_fusion2 = fusion_block()
        self.attention = nn.Linear(in_size, 1)

        self.linear_1 = nn.Linear(in_size * 3, hidden)
        self.linear_2 = nn.Linear(hidden, hidden)
        self.linear_3 = nn.Linear(hidden, output_dim)

        self.in_size = in_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a1, v1, l1 = x[:, 0, :], x[:, 1, :], x[:, 2, :]

        # Unimodal attention
        sa = torch.sigmoid(self.attention(a1))  # (B,1)
        sv = torch.sigmoid(self.attention(v1))
        sl = torch.sigmoid(self.attention(l1))
        total_weights = torch.cat([sa, sv, sl], dim=1)

        unimodal_a = sa.expand_as(a1)
        unimodal_v = sv.expand_as(v1)
        unimodal_l = sl.expand_as(l1)
        unimodal = (unimodal_a * a1 + unimodal_v * v1 + unimodal_l * l1) / 3

        # Bimodal norms
        a = F.softmax(a1, dim=1)
        v = F.softmax(v1, dim=1)
        l = F.softmax(l1, dim=1)

        sav = (1 / (torch.bmm(a.unsqueeze(1), v.unsqueeze(2)).squeeze(-1).squeeze(-1) + 0.5) * (sa + sv)).squeeze(-1)
        sal = (1 / (torch.bmm(a.unsqueeze(1), l.unsqueeze(2)).squeeze(-1).squeeze(-1) + 0.5) * (sa + sl)).squeeze(-1)
        svl = (1 / (torch.bmm(v.unsqueeze(1), l.unsqueeze(2)).squeeze(-1).squeeze(-1) + 0.5) * (sl + sv)).squeeze(-1)

        normalize = torch.stack([sav, sal, svl], dim=1)  # (B,3)
        normalize = F.softmax(normalize, dim=1)
        total_weights = torch.cat([total_weights, normalize], dim=1)

        a_v = F.elu(normalize[:, 0].unsqueeze(1).expand_as(a1) * self.graph_fusion(torch.cat([a1, v1], dim=1)))
        a_l = F.elu(normalize[:, 1].unsqueeze(1).expand_as(a1) * self.graph_fusion(torch.cat([a1, l1], dim=1)))
        v_l = F.elu(normalize[:, 2].unsqueeze(1).expand_as(a1) * self.graph_fusion(torch.cat([v1, l1], dim=1)))
        bimodal = a_v + a_l + v_l

        # Trimodal
        a_v2 = F.softmax(self.graph_fusion(torch.cat([a1, v1], dim=1)), dim=1)
        a_l2 = F.softmax(self.graph_fusion(torch.cat([a1, l1], dim=1)), dim=1)
        v_l2 = F.softmax(self.graph_fusion(torch.cat([v1, l1], dim=1)), dim=1)

        def sp(a_: torch.Tensor, b_: torch.Tensor) -> torch.Tensor:
            return torch.bmm(a_.unsqueeze(1), b_.unsqueeze(2)).squeeze(-1).squeeze(-1)

        savvl = (1 / (sp(a_v2, v_l2) + 0.5) * (sav + svl))
        saavl = (1 / (sp(a_v2, a_l2) + 0.5) * (sav + sal))
        savll = (1 / (sp(a_l2, v_l2) + 0.5) * (sal + svl))
        savl = (1 / (sp(a_v2, l) + 0.5) * (sav + sl.squeeze(-1)))
        salv = (1 / (sp(a_l2, v) + 0.5) * (sal + sv.squeeze(-1)))
        svla = (1 / (sp(v_l2, a) + 0.5) * (sa.squeeze(-1) + svl))

        normalize2 = torch.stack([savvl, saavl, savll, savl, salv, svla], dim=1)
        normalize2 = F.softmax(normalize2, dim=1)
        total_weights = torch.cat([total_weights, normalize2], dim=1)

        avvl = F.elu(normalize2[:, 0].unsqueeze(1).expand_as(a1) * self.graph_fusion2(torch.cat([a_v, v_l], dim=1)))
        aavl = F.elu(normalize2[:, 1].unsqueeze(1).expand_as(a1) * self.graph_fusion2(torch.cat([a_v, a_l], dim=1)))
        avll = F.elu(normalize2[:, 2].unsqueeze(1).expand_as(a1) * self.graph_fusion2(torch.cat([v_l, a_l], dim=1)))
        avl = F.elu(normalize2[:, 3].unsqueeze(1).expand_as(a1) * self.graph_fusion2(torch.cat([a_v, l1], dim=1)))
        alv = F.elu(normalize2[:, 4].unsqueeze(1).expand_as(a1) * self.graph_fusion2(torch.cat([a_l, v1], dim=1)))
        vla = F.elu(normalize2[:, 5].unsqueeze(1).expand_as(a1) * self.graph_fusion2(torch.cat([v_l, a1], dim=1)))

        trimodal = avvl + aavl + avll + avl + alv + vla

        fusion = torch.cat([unimodal, bimodal, trimodal], dim=1)
        fusion = self.norm2(fusion)
        y = torch.tanh(self.linear_1(fusion))
        y = torch.tanh(self.linear_2(y))
        y = F.softmax(self.linear_3(y), dim=1)
        return y, total_weights


class Concat(nn.Module):
    def __init__(self, in_size: int, output_dim: int, hidden: int = 50, dropout: float = 0.5):
        super().__init__()
        self.norm2 = nn.BatchNorm1d(in_size * 3)
        self.linear_1 = nn.Linear(in_size * 3, hidden)
        self.linear_2 = nn.Linear(hidden, hidden)
        self.linear_3 = nn.Linear(hidden, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a1, v1, l1 = x[:, 0, :], x[:, 1, :], x[:, 2, :]
        fusion = torch.cat([a1, v1, l1], dim=1)
        fusion = self.norm2(fusion)
        y = torch.tanh(self.linear_1(fusion))
        y = torch.tanh(self.linear_2(y))
        y = F.softmax(self.linear_3(y), dim=1)
        return y, y  # keep placeholder for backward compatibility


class Multiplication(nn.Module):
    """Elementwise multiplicative fusion.
    In the original code, `fusion = a1*v1` was overwritten by `fusion = v1*l1`.
    Here we fix it to element-wise product of all three.
    """

    def __init__(self, in_size: int, output_dim: int, hidden: int = 50, dropout: float = 0.5):
        super().__init__()
        self.norm2 = nn.BatchNorm1d(in_size)
        self.linear_1 = nn.Linear(in_size, hidden)
        self.linear_2 = nn.Linear(hidden, hidden)
        self.linear_3 = nn.Linear(hidden, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a1, v1, l1 = x[:, 0, :], x[:, 1, :], x[:, 2, :]
        fusion = a1 * v1 * l1  # fix: multiply all three instead of overwriting
        fusion = self.norm2(fusion)
        y = torch.tanh(self.linear_1(fusion))
        y = torch.tanh(self.linear_2(y))
        y = F.softmax(self.linear_3(y), dim=1)
        return y, y


class TensorFusion(nn.Module):
    """Full outer-product (tensor) fusion."""

    def __init__(self, in_size: int, output_dim: int, hidden: int = 50, dropout: float = 0.5):
        super().__init__()
        self.post_fusion_dropout = nn.Dropout(p=dropout)
        self.post_fusion_layer_1 = nn.Linear((in_size + 1) * (in_size + 1) * (in_size + 1), hidden)
        self.post_fusion_layer_2 = nn.Linear(hidden, hidden)
        self.post_fusion_layer_3 = nn.Linear(hidden, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a1, v1, l1 = x[:, 0, :], x[:, 1, :], x[:, 2, :]
        B, D = a1.size(0), a1.size(1)
        device, dtype = a1.device, a1.dtype

        ones = torch.ones(B, 1, device=device, dtype=dtype)
        _a = torch.cat([ones, a1], dim=1)
        _v = torch.cat([ones, v1], dim=1)
        _l = torch.cat([ones, l1], dim=1)

        fusion_tensor = torch.bmm(_a.unsqueeze(2), _v.unsqueeze(1))  # (B, D+1, D+1)
        fusion_tensor = fusion_tensor.view(-1, (D + 1) * (D + 1), 1)
        fusion_tensor = torch.bmm(fusion_tensor, _l.unsqueeze(1)).view(B, -1)

        x = self.post_fusion_dropout(fusion_tensor)
        x = F.relu(self.post_fusion_layer_1(x))
        x = F.relu(self.post_fusion_layer_2(x))
        y = F.softmax(self.post_fusion_layer_3(x), dim=1)
        return y, y


class LowRankFusion(nn.Module):
    """Low-rank outer-product fusion (LMF)."""

    def __init__(self, in_size: int, output_dim: int, hidden: int = 50, dropout: float = 0.5, rank: int = 4):
        super().__init__()
        self.in_size = in_size
        self.output_dim = output_dim
        self.rank = rank

        # factors: (R, D+1, C)
        self.audio_factor = nn.Parameter(torch.empty(rank, in_size + 1, output_dim))
        self.video_factor = nn.Parameter(torch.empty(rank, in_size + 1, output_dim))
        self.text_factor = nn.Parameter(torch.empty(rank, in_size + 1, output_dim))
        self.fusion_weights = nn.Parameter(torch.empty(1, rank))
        self.fusion_bias = nn.Parameter(torch.zeros(1, output_dim))

        # init
        nn.init.xavier_normal_(self.audio_factor)
        nn.init.xavier_normal_(self.video_factor)
        nn.init.xavier_normal_(self.text_factor)
        nn.init.xavier_normal_(self.fusion_weights)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a1, v1, l1 = x[:, 0, :], x[:, 1, :], x[:, 2, :]
        B, D = a1.size(0), a1.size(1)
        device, dtype = a1.device, a1.dtype

        ones = torch.ones(B, 1, device=device, dtype=dtype)
        _a = torch.cat([ones, a1], dim=1)  # (B, D+1)
        _v = torch.cat([ones, v1], dim=1)
        _l = torch.cat([ones, l1], dim=1)

        # (B, D+1, C) after multiplying by factors (R, D+1, C) per rank -> broadcast via einsum
        fa = torch.einsum('bd,rdc->brc', _a, self.audio_factor)
        fv = torch.einsum('bd,rdc->brc', _v, self.video_factor)
        fl = torch.einsum('bd,rdc->brc', _l, self.text_factor)

        fzy = fa * fv * fl  # (B, R, C)
        out = torch.einsum('brc,1r->bc', fzy, self.fusion_weights) + self.fusion_bias  # (B, C)
        return out, out


class LateFusion(nn.Module):
    def __init__(self, in_size: int, output_dim: int, hidden: int = 50, dropout: float = 0.5):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.norm2 = nn.BatchNorm1d(in_size)
        self.attention = nn.Linear(in_size, 1)
        self.linear_1 = nn.Linear(in_size, hidden)
        self.linear_2 = nn.Linear(hidden, hidden)
        self.linear_3 = nn.Linear(hidden, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a1, v1, l1 = x[:, 0, :], x[:, 1, :], x[:, 2, :]

        a1 = self.norm2(a1)
        v1 = self.norm2(v1)
        l1 = self.norm2(l1)

        a = torch.tanh(self.attention(a1))  # (B,1)
        v = torch.tanh(self.attention(v1))
        l = torch.tanh(self.attention(l1))

        w = torch.cat([a, v, l], dim=1)
        w = F.softmax(w, dim=1)  # (B,3)

        fusion = (
            w[:, 0].unsqueeze(1).expand_as(a1) * a1
            + w[:, 1].unsqueeze(1).expand_as(v1) * v1
            + w[:, 2].unsqueeze(1).expand_as(l1) * l1
        )

        fusion = self.norm2(fusion)
        y = torch.tanh(self.linear_1(fusion))
        y = torch.tanh(self.linear_2(y))
        y = F.softmax(self.linear_3(y), dim=1)
        return y, y


class Graph12(nn.Module):
    def __init__(self, in_size: int, output_dim: int, hidden: int = 50, dropout: float = 0.5):
        super().__init__()
        self.norm2 = nn.BatchNorm1d(in_size * 3)
        self.drop = nn.Dropout(p=dropout)
        self.graph_fusion = nn.Sequential(
            nn.Linear(in_size * 2, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, in_size),
            nn.Tanh(),
        )
        self.attention = nn.Linear(in_size, 1)
        self.linear_1 = nn.Linear(in_size * 3, hidden)
        self.linear_2 = nn.Linear(hidden, hidden)
        self.linear_3 = nn.Linear(hidden, output_dim)
        self.in_size = in_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a1, v1, l1 = x[:, 0, :], x[:, 1, :], x[:, 2, :]
        sa = torch.tanh(self.attention(a1))  # (B,1)
        sv = torch.tanh(self.attention(v1))
        sl = torch.tanh(self.attention(l1))

        w = torch.cat([sa, sv, sl], dim=1)
        w = F.softmax(w, dim=1)
        sa, sv, sl = w[:, 0:1], w[:, 1:2], w[:, 2:3]

        total_weights = w

        unimodal = (
            sa.expand_as(a1) * a1 + sv.expand_as(v1) * v1 + sl.expand_as(l1) * l1
        ) / 3

        a = F.softmax(a1, dim=1).unsqueeze(1)
        v = F.softmax(v1, dim=1).unsqueeze(2)
        l = F.softmax(l1, dim=1).unsqueeze(2)

        sav = (1 / (torch.bmm(a, v).squeeze(-1).squeeze(-1) + 0.5) * (sa.squeeze(-1) + sv.squeeze(-1)))
        sal = (1 / (torch.bmm(a, l).squeeze(-1).squeeze(-1) + 0.5) * (sa.squeeze(-1) + sl.squeeze(-1)))
        svl = (1 / (torch.bmm(v.squeeze(2).unsqueeze(1), l).squeeze(-1).squeeze(-1) + 0.5) * (sl.squeeze(-1) + sv.squeeze(-1)))

        norm_bi = torch.stack([sav, sal, svl], dim=1)
        norm_bi = F.softmax(norm_bi, dim=1)
        total_weights = torch.cat([total_weights, norm_bi], dim=1)

        a_v = F.leaky_relu(norm_bi[:, 0].unsqueeze(1).expand_as(a1) * self.graph_fusion(torch.cat([a1, v1], dim=1)))
        a_l = F.leaky_relu(norm_bi[:, 1].unsqueeze(1).expand_as(a1) * self.graph_fusion(torch.cat([a1, l1], dim=1)))
        v_l = F.leaky_relu(norm_bi[:, 2].unsqueeze(1).expand_as(a1) * self.graph_fusion(torch.cat([v1, l1], dim=1)))
        bimodal = (a_v + a_l + v_l) / 3

        a_v2 = F.softmax(a_v, dim=1).unsqueeze(1)
        a_l2 = F.softmax(a_l, dim=1).unsqueeze(2)
        v_l2 = F.softmax(v_l, dim=1).unsqueeze(2)

        def sp(a_: torch.Tensor, b_: torch.Tensor) -> torch.Tensor:
            return torch.bmm(a_, b_).squeeze(-1).squeeze(-1)

        savvl = (1 / (sp(a_v2, v_l2) + 0.5) * (sav + svl))
        saavl = (1 / (sp(a_v2, a_l2) + 0.5) * (sav + sal))
        savll = (1 / (sp(a_l2.transpose(1, 2), v_l2) + 0.5) * (sal + svl))
        savl = (1 / (sp(a_v2, l) + 0.5) * (sav + sl.squeeze(-1)))
        salv = (1 / (sp(a_l2.transpose(1, 2), v) + 0.5) * (sal + sv.squeeze(-1)))
        svla = (1 / (sp(v_l2.transpose(1, 2), a) + 0.5) * (sa.squeeze(-1) + svl))

        norm_tri = torch.stack([savvl, saavl, savll, savl, salv, svla], dim=1)
        norm_tri = F.softmax(norm_tri, dim=1)
        total_weights = torch.cat([total_weights, norm_tri], dim=1)

        avvl = F.leaky_relu(norm_tri[:, 0].unsqueeze(1).expand_as(a1) * self.graph_fusion(torch.cat([a_v, v_l], dim=1)))
        aavl = F.leaky_relu(norm_tri[:, 1].unsqueeze(1).expand_as(a1) * self.graph_fusion(torch.cat([a_v, a_l], dim=1)))
        avll = F.leaky_relu(norm_tri[:, 2].unsqueeze(1).expand_as(a1) * self.graph_fusion(torch.cat([v_l, a_l], dim=1)))
        avl = F.leaky_relu(norm_tri[:, 3].unsqueeze(1).expand_as(a1) * self.graph_fusion(torch.cat([a_v, l1], dim=1)))
        alv = F.leaky_relu(norm_tri[:, 4].unsqueeze(1).expand_as(a1) * self.graph_fusion(torch.cat([a_l, v1], dim=1)))
        vla = F.leaky_relu(norm_tri[:, 5].unsqueeze(1).expand_as(a1) * self.graph_fusion(torch.cat([v_l, a1], dim=1)))

        trimodal = (avvl + aavl + avll + avl + alv + vla) / 6

        fusion = torch.cat([unimodal, bimodal, trimodal], dim=1)
        fusion = self.drop(self.norm2(fusion))
        y = torch.tanh(self.linear_1(fusion))
        y = torch.tanh(self.linear_2(y))
        y = F.softmax(self.linear_3(y), dim=1)
        return y, total_weights


class OuterProduct(nn.Module):
    """Full outer-product fusion with post-MLP."""

    def __init__(self, in_size: int=32, output_dim: int=32, hidden: int = 50, dropout: float = 0.5, use_softmax: bool = True):
        super().__init__()
        self.audio_in = in_size
        self.video_in = in_size
        self.text_in = in_size
        self.audio_hidden = hidden
        self.output_dim = output_dim
        self.use_softmax = use_softmax

        self.post_fusion_dropout = nn.Dropout(p=dropout)
        self.post_fusion_layer_1 = nn.Linear((in_size + 1) * (in_size + 1) * (in_size + 1), hidden)
        self.post_fusion_layer_2 = nn.Linear(hidden, hidden)
        self.post_fusion_layer_3 = nn.Linear(hidden, output_dim)

    def forward(self, a: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        t = t[:, 0, :]
        B, D = a.size(0), a.size(1)
        device, dtype = a.device, a.dtype

        ones = torch.ones(B, 1, device=device, dtype=dtype)
        _a = torch.cat([ones, a], dim=1)
        _v = torch.cat([ones, v], dim=1)
        _t = torch.cat([ones, t], dim=1)

        fusion = torch.bmm(_a.unsqueeze(2), _v.unsqueeze(1))
        fusion = fusion.view(-1, (D + 1) * (D + 1), 1)
        fusion = torch.bmm(fusion, _t.unsqueeze(1)).view(B, -1)

        y = self.post_fusion_dropout(fusion)
        y = F.relu(self.post_fusion_layer_1(y))
        y = F.relu(self.post_fusion_layer_2(y))
        y = torch.sigmoid(self.post_fusion_layer_3(y))
        if self.use_softmax:
            y = F.softmax(y, dim=1)
        return y


class SubNet(nn.Module):
    '''
    The subnetwork that is used in LMF for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, dropout):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(SubNet, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        normed = self.norm(x)
        dropped = self.drop(normed)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        y_3 = F.relu(self.linear_3(y_2))

        return y_3


class TextSubNet(nn.Module):
    '''
    The LSTM-based subnetwork that is used in LMF for text
    '''

    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(TextSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_len, in_size)
        '''
        _, final_states = self.rnn(x)
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1


class LMF(nn.Module):
    '''
    Low-rank Multimodal Fusion
    '''

    def __init__(self, input_dims=(32,32,32), hidden_dims=(32,32,32), text_out= 2, dropouts=(0.1,0.1,0.1,0.1), output_dim=32, rank=4, use_softmax=False):
        '''
        Args:
            input_dims - a length-3 tuple, contains (audio_dim, video_dim, text_dim)
            hidden_dims - another length-3 tuple, hidden dims of the sub-networks
            text_out - int, specifying the resulting dimensions of the text subnetwork
            dropouts - a length-4 tuple, contains (audio_dropout, video_dropout, text_dropout, post_fusion_dropout)
            output_dim - int, specifying the size of output
            rank - int, specifying the size of rank in LMF
        Output:
            (return value in forward) a scalar value between -3 and 3
        '''
        super(LMF, self).__init__()

        # dimensions are specified in the order of audio, video and text
        self.audio_in = input_dims[0]
        self.video_in = input_dims[1]
        self.text_in = input_dims[2]

        self.audio_hidden = hidden_dims[0]
        self.video_hidden = hidden_dims[1]
        self.text_hidden = hidden_dims[2]
        self.text_out= text_out
        self.output_dim = output_dim
        self.rank = rank
        self.use_softmax = use_softmax

        self.audio_prob = dropouts[0]
        self.video_prob = dropouts[1]
        self.text_prob = dropouts[2]
        self.post_fusion_prob = dropouts[3]

        # define the pre-fusion subnetworks
        self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, self.audio_prob)
        self.video_subnet = SubNet(self.video_in, self.video_hidden, self.video_prob)
        self.text_subnet = TextSubNet(self.text_in, self.text_hidden, self.text_out, dropout=self.text_prob)

        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        # self.post_fusion_layer_1 = nn.Linear((self.text_out + 1) * (self.video_hidden + 1) * (self.audio_hidden + 1), self.post_fusion_dim)
        self.audio_factor = Parameter(torch.Tensor(self.rank, self.audio_hidden + 1, self.output_dim))
        self.video_factor = Parameter(torch.Tensor(self.rank, self.video_hidden + 1, self.output_dim))
        self.text_factor = Parameter(torch.Tensor(self.rank, self.text_out + 1, self.output_dim))
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, self.output_dim))

        # init teh factors
        xavier_normal(self.audio_factor)
        xavier_normal(self.video_factor)
        xavier_normal(self.text_factor)
        xavier_normal(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, audio_x, video_x, text_x):
        '''
        Args:
            audio_x: tensor of shape (batch_size, audio_in)
            video_x: tensor of shape (batch_size, video_in)
            text_x: tensor of shape (batch_size, sequence_len, text_in)
        '''
        audio_h = self.audio_subnet(audio_x)
        video_h = self.video_subnet(video_x)
        text_h = self.text_subnet(text_x)
        batch_size = audio_h.data.shape[0]

        # next we perform low-rank multimodal fusion
        # here is a more efficient implementation than the one the paper describes
        # basically swapping the order of summation and elementwise product
        if audio_h.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        _audio_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), audio_h), dim=1)
        _video_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), video_h), dim=1)
        _text_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), text_h), dim=1)

        fusion_audio = torch.matmul(_audio_h, self.audio_factor)
        fusion_video = torch.matmul(_video_h, self.video_factor)
        fusion_text = torch.matmul(_text_h, self.text_factor)
        fusion_zy = fusion_audio * fusion_video * fusion_text

        # output = torch.sum(fusion_zy, dim=0).squeeze()
        # use linear transformation instead of simple summation, more flexibility
        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        if self.use_softmax:
            output = F.softmax(output)
        return output
    


class HealthPredictor(nn.Module):
    def __init__(self, args, hidden_size=32, hidden_size_d=32, num_layers=3, output_size=1):
        super(HealthPredictor, self).__init__()
        # ni 79 pe 16 pi 16 ls 10
        self.ni_dim = 79
        self.pe_dim = 16
        self.pi_dim = 16
        self.ls_dim = 10
        self.hidden_size = hidden_size
        self.hidden_size_d = hidden_size_d
        self.num_layers = num_layers
        # Static
        self.adj_matrix_s = torch.load('./data/adj_matrix2.pt')
        self.static_fusion = StaticFusion(self.pe_dim, self.pi_dim, self.ls_dim, out_size=hidden_size_d)
        # Dynamic
        self.adj_matrix = torch.load('./data/adj_matrix.pt').to(args.device)
        self.dynamic_fusion = DaynamicFusion(self.ni_dim, hidden_size, hidden_size_d, num_layers)
        # Classification
        self.fc = nn.Linear(hidden_size_d, output_size)
        self.sigmoid = nn.Sigmoid()
        self.loss = BCEWithLogitsLossWithLabelSmoothing(args)
        self.relu = nn.ReLU()
        self.lmf = OuterProduct()
        self._init_weights(self.fc)

    def forward(self, id, ls, pi, pe, ni, label):
        # ni 8,33,79
        # static
        b = id.shape[0]
        pe, pi, ls = self.static_fusion(pe, pi, ls, self.adj_matrix_s)
        sta_f  = torch.stack([pe, pi, ls], dim=1)
        # print('s_f', sta_f.shape)
        # dynamic
        ni_f = self.dynamic_fusion(ni, self.adj_matrix)
        ni_f = ni_f.unsqueeze(1)  # [b, 32]
        # fusion
        out = self.lmf(pe.squeeze(), pi.squeeze(), ni_f)
        # print('out', out.shape)
        out = self.fc(out)
        pre = self.sigmoid(out.squeeze(1))
        loss = self.loss(out.squeeze(1), label.float())
        # o = torch.concat(((pre-0.5).unsqueeze(1),(0.5-pre).unsqueeze(1)),1)
        # print('o', o.shape)
        return pre , loss
    
    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

if __name__ == "__main__":
    ni = torch.rand([8,33,79])
    # p = Pinjie()
    adj_matrix_s = torch.load('./data/adj_matrix.pt')
    pi = torch.randn([8,1,32])
    pe = torch.randn([8,1,32])
    ls = torch.randn([8,1,32])
    f = torch.randn([8,1,32])
    # dy = StaticFusion(16,16,10,48)
    # # x = x.reshape(32,144,-1)
    # out= dy(pi,pe,ls,adj_matrix_s)
    model = DaynamicFusion()
    out = model(ni,[pi,pe,ls,f],adj_matrix_s)
    print(out.shape)
    # print('r')
    # # r = r.reshape(32,72,134,-1)
    # w = WaveNet(134,0.2,7,1,144,32,32,128,64,4,72,2,2,32)
    # # r = r.permute(0,3,2,1)
    # rw = w(r)