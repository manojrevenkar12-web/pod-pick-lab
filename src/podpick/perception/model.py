"""Find the ordered item in the bin image and say how obstructed it is.

A small U-Net over depth and intensity, conditioned on the order (catalogue dimensions, colour,
weight) by FiLM. Two outputs: the target's mask, and its occlusion class (unobstructed, below
other items, blocked) - the extra classification layer the Vulcan Pick write-up describes on
top of its segmentation model.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .. import config
from ..interface import Observation, TargetQuery

CLASSES = ("unobstructed", "below_others", "blocked")
DEPTH_CENTRE, DEPTH_SCALE = 0.45, 0.25


def encode_query(q: TargetQuery) -> np.ndarray:
    s = config.policy()["perception"]["query_scale"]
    return np.array([*(d / s["dims"] for d in q.dims), q.albedo, q.mass / s["mass"]], dtype=np.float32)


def encode_image(depth: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    valid = (depth > 0).astype(np.float32)
    d = np.where(depth > 0, (depth - DEPTH_CENTRE) / DEPTH_SCALE, 0.0).astype(np.float32)
    return np.stack([d, intensity.astype(np.float32), valid])


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class TargetNet(nn.Module):
    def __init__(self, widths: tuple[int, ...] = (16, 24, 32, 48), qdim: int = 5):
        super().__init__()
        self.downs = nn.ModuleList()
        cin = 3
        for w in widths:
            self.downs.append(_block(cin, w))
            cin = w
        self.pool = nn.MaxPool2d(2)
        self.film = nn.ModuleList([nn.Sequential(nn.Linear(qdim, 64), nn.ReLU(), nn.Linear(64, 2 * w)) for w in widths])
        self.ups = nn.ModuleList()
        self.upconv = nn.ModuleList()
        for i in range(len(widths) - 1, 0, -1):
            self.upconv.append(nn.ConvTranspose2d(widths[i], widths[i - 1], 2, stride=2))
            self.ups.append(_block(2 * widths[i - 1], widths[i - 1]))
        self.mask_head = nn.Conv2d(widths[0], 1, 1)
        self.cls_head = nn.Sequential(nn.Linear(widths[-1] + widths[0], 64), nn.ReLU(), nn.Linear(64, len(CLASSES)))

    def _mod(self, x: torch.Tensor, q: torch.Tensor, i: int) -> torch.Tensor:
        g, b = self.film[i](q).chunk(2, dim=1)
        return x * (1 + g[:, :, None, None]) + b[:, :, None, None]

    def forward(self, img: torch.Tensor, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skips = []
        x = img
        for i, down in enumerate(self.downs):
            x = self._mod(down(x), q, i)
            skips.append(x)
            if i < len(self.downs) - 1:
                x = self.pool(x)
        bottom = x
        for j, (up, conv) in enumerate(zip(self.upconv, self.ups, strict=True)):
            x = up(x)
            x = conv(torch.cat([x, skips[-2 - j]], dim=1))
        mask_logit = self.mask_head(x)[:, 0]
        att = torch.sigmoid(mask_logit)[:, None]
        pooled_low = (x * att).sum((2, 3)) / att.sum((2, 3)).clamp_min(1.0)
        pooled_high = bottom.mean((2, 3))
        cls = self.cls_head(torch.cat([pooled_high, pooled_low], dim=1))
        return mask_logit, cls


class Perception:
    """The deployed perception step: a trained TargetNet behind a numpy interface."""

    def __init__(self, net: TargetNet):
        self.net = net.eval()
        self.cfg = config.policy()["perception"]

    @torch.no_grad()
    def __call__(self, obs: Observation) -> dict[str, np.ndarray]:
        img = torch.from_numpy(encode_image(obs.depth, obs.intensity))[None]
        q = torch.from_numpy(encode_query(obs.query))[None]
        mlog, cls = self.net(img, q)
        prob = torch.sigmoid(mlog)[0].numpy()
        mask = prob > self.cfg["mask_threshold"]
        mask = _largest_component(mask)
        return {"mask": mask, "mask_prob": prob, "occlusion": torch.softmax(cls, 1)[0].numpy()}


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep the largest 4-connected component (the target is one item)."""
    H, W = mask.shape
    lab = np.zeros((H, W), dtype=np.int32)
    best, best_n, cur = 0, 0, 0
    for sv, su in zip(*np.nonzero(mask), strict=True):
        if lab[sv, su]:
            continue
        cur += 1
        stack = [(sv, su)]
        lab[sv, su] = cur
        n = 0
        while stack:
            v, u = stack.pop()
            n += 1
            for dv, du in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = v + dv, u + du
                if 0 <= a < H and 0 <= b < W and mask[a, b] and not lab[a, b]:
                    lab[a, b] = cur
                    stack.append((a, b))
        if n > best_n:
            best, best_n = cur, n
    return lab == best if best else np.zeros_like(mask)
