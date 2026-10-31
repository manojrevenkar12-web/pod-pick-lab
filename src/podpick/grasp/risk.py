"""Pick risk: for one candidate and one extraction primitive, how likely is each outcome?

Three heads, trained on logged pick attempts:
- `success`: the target leaves the bin on the cup;
- `collateral`: at least one other item leaves the bin with it;
- `drop`: the target falls outside the bin.

The crop encoder can be pretrained without any pick outcomes, by masked reconstruction of crops
around candidates (in-domain self-supervised pretraining, as in Amazon's multi-suction pick
success work). `podpick.train.risk` measures whether that helps at small label budgets.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .. import config
from .candidates import FEATURES, Candidates

HEADS = ("success", "collateral", "drop")


def primitives() -> list[str]:
    return list(config.policy()["primitives"].keys())


class CropEncoder(nn.Module):
    """4 input channels: relative depth, target mask, valid, visible (1 unless masked in pretraining)."""

    def __init__(self, width: int = 16, out: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, width, 3, padding=1),
            nn.GroupNorm(8, width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.GroupNorm(8, width),
            nn.GELU(),  # 16
            nn.Conv2d(width, 2 * width, 3, padding=1),
            nn.GroupNorm(8, 2 * width),
            nn.GELU(),
            nn.Conv2d(2 * width, 2 * width, 3, stride=2, padding=1),
            nn.GroupNorm(8, 2 * width),
            nn.GELU(),  # 8
            nn.Conv2d(2 * width, out, 3, padding=1),
            nn.GroupNorm(8, out),
            nn.GELU(),
        )
        self.out = out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, out, 8, 8)


class MaskedDecoder(nn.Module):
    def __init__(self, cin: int = 64, width: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(cin, 2 * width, 2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(2 * width, width, 2, stride=2),
            nn.GELU(),
            nn.Conv2d(width, 3, 3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class RiskNet(nn.Module):
    def __init__(self, n_features: int = len(FEATURES), n_prims: int = 3):
        super().__init__()
        self.encoder = CropEncoder()
        self.register_buffer("f_mean", torch.zeros(n_features))
        self.register_buffer("f_std", torch.ones(n_features))
        self.feat = nn.Sequential(nn.Linear(n_features + n_prims, 64), nn.GELU())
        # the candidate (features + primitive) queries the crop tokens, as in pick-conditioned
        # cross-attention
        self.attn = nn.MultiheadAttention(64, 4, batch_first=True)
        self.head = nn.Sequential(nn.Linear(64 * 3, 96), nn.GELU(), nn.Dropout(0.1), nn.Linear(96, len(HEADS)))

    def forward(self, crops: torch.Tensor, feats: torch.Tensor, prim: torch.Tensor) -> torch.Tensor:
        vis = torch.ones_like(crops[:, :1])
        tok = self.encoder(torch.cat([crops, vis], 1)).flatten(2).transpose(1, 2)  # (B, 64 tokens, 64)
        f = (feats - self.f_mean) / self.f_std
        q = self.feat(torch.cat([f, prim], 1))
        a, _ = self.attn(q[:, None], tok, tok)
        return self.head(torch.cat([q, a[:, 0], tok.mean(1)], 1))  # logits per head


class RiskModel:
    """Deployed risk scoring: every (candidate, primitive) pair at once."""

    def __init__(self, net: RiskNet):
        self.net = net.eval()
        self.prims = primitives()

    @torch.no_grad()
    def __call__(self, c: Candidates) -> np.ndarray:
        """Probabilities, shape (K, P, 3) for heads success, collateral, drop."""
        K, P = len(c), len(self.prims)
        if K == 0:
            return np.zeros((0, P, len(HEADS)))
        crops = torch.from_numpy(np.repeat(c.crops, P, axis=0))
        feats = torch.from_numpy(np.repeat(c.features, P, axis=0))
        prim = torch.eye(P)[torch.arange(P).repeat(K)]
        return torch.sigmoid(self.net(crops, feats, prim)).numpy().reshape(K, P, len(HEADS))
