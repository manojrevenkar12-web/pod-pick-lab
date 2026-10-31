"""Train the target finder on labelled bin images (data/perception_*.npz)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from torch import nn

from ..perception.model import CLASSES, TargetNet


def _batches(n: int, bs: int, rng: np.random.Generator | None):
    idx = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, bs):
        yield idx[i : i + bs]


def _tensors(d: dict[str, np.ndarray], ix: np.ndarray, flip: np.ndarray | None = None) -> tuple[torch.Tensor, ...]:
    img = d["images"][d["scene"][ix]].astype(np.float32)
    mask = d["mask"][ix].astype(np.float32)
    if flip is not None:
        # a mirrored bin is still a bin: flip left-right
        img[flip] = img[flip][..., ::-1]
        mask[flip] = mask[flip][..., ::-1]
    return (
        torch.from_numpy(np.ascontiguousarray(img)),
        torch.from_numpy(d["query"][ix]),
        torch.from_numpy(np.ascontiguousarray(mask)),
        torch.from_numpy(d["occlusion"][ix].astype(np.int64)),
    )


def evaluate(net: TargetNet, d: dict[str, np.ndarray], threshold: float = 0.5) -> dict[str, Any]:
    net.eval()
    inter = union = 0.0
    ious, correct, n = [], 0, 0
    conf = np.zeros((len(CLASSES), len(CLASSES)), int)
    found = 0
    with torch.no_grad():
        for ix in _batches(len(d["query"]), 64, None):
            img, q, m, occ = _tensors(d, ix)
            ml, cl = net(img, q)
            pred = torch.sigmoid(ml) > threshold
            gt = m > 0.5
            i = (pred & gt).sum((1, 2)).float()
            u = (pred | gt).sum((1, 2)).float()
            inter += float(i.sum())
            union += float(u.sum())
            ious.extend((i / u.clamp_min(1)).tolist())
            found += int(((i / gt.sum((1, 2)).clamp_min(1)) > 0.5).sum())
            pc = cl.argmax(1)
            correct += int((pc == occ).sum())
            n += len(ix)
            for a, b in zip(occ.tolist(), pc.tolist(), strict=True):
                conf[a, b] += 1
    return {
        "mask_iou_mean": float(np.mean(ious)),
        "mask_iou_pooled": inter / max(union, 1),
        "target_found": found / max(n, 1),
        "occlusion_accuracy": correct / max(n, 1),
        "occlusion_confusion": conf.tolist(),
        "n": n,
    }


def train(tr: dict[str, np.ndarray], va: dict[str, np.ndarray], epochs: int, seed: int = 0, log=print) -> tuple[TargetNet, dict[str, Any]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = TargetNet()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(tr["query"]) / 32))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=max(steps, 1))
    counts = np.bincount(tr["occlusion"], minlength=len(CLASSES)).astype(np.float32)
    cw = torch.from_numpy(counts.sum() / np.maximum(counts, 1) / len(CLASSES))
    ce = nn.CrossEntropyLoss(weight=cw)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(4.0))
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        tot = 0.0
        for ix in _batches(len(tr["query"]), 32, rng):
            flip = rng.random(len(ix)) < 0.5
            img, q, m, occ = _tensors(tr, ix, flip)
            ml, cl = net(img, q)
            p = torch.sigmoid(ml)
            dice = 1 - (2 * (p * m).sum((1, 2)) + 1) / (p.sum((1, 2)) + m.sum((1, 2)) + 1)
            loss = bce(ml, m) + dice.mean() + 0.5 * ce(cl, occ)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(ix)
        log(f"perception epoch {ep + 1}/{epochs} loss {tot / len(tr['query']):.4f} ({time.time() - t0:.0f}s)")
    metrics = evaluate(net, va)
    metrics["train_samples"] = int(len(tr["query"]))
    metrics["parameters"] = int(sum(p.numel() for p in net.parameters()))
    return net, metrics
