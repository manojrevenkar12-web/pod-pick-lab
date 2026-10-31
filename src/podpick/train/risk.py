"""Train the pick-risk model, with and without self-supervised pretraining, at several label budgets.

Validation uses orders that are not in the training set (a separate seed range); the test
families are never touched here.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from torch import nn

from .. import config
from ..grasp.risk import HEADS, MaskedDecoder, RiskNet


def auc(y: np.ndarray, s: np.ndarray) -> float | None:
    y = np.asarray(y).astype(bool)
    if y.all() or (~y).all():
        return None
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s))
    sv = np.asarray(s)[order]
    i = 0
    while i < len(sv):  # average ranks over ties
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    npos, nneg = y.sum(), (~y).sum()
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            e += sel.mean() * abs(y[sel].mean() - p[sel].mean())
    return float(e)


def _f(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def _labels(d: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([d[h].astype(np.float32) for h in HEADS], 1)


def _prim_onehot(p: np.ndarray, n: int = 3) -> np.ndarray:
    return np.eye(n, dtype=np.float32)[p.astype(int)]


def _mask_patches(x: torch.Tensor, ratio: float, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    B, _, H, W = x.shape
    ps = 8
    keep = (torch.rand(B, 1, H // ps, W // ps, generator=g) > ratio).float()
    keep = keep.repeat_interleave(ps, 2).repeat_interleave(ps, 3)
    return torch.cat([x * keep, keep], 1), keep


def pretrain(crops: np.ndarray, epochs: int, seed: int = 0, log=print) -> dict[str, torch.Tensor]:
    """Masked reconstruction of candidate crops; returns the encoder weights."""
    cfg = config.policy()["risk"]
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    net = RiskNet()
    enc, dec = net.encoder, MaskedDecoder(net.encoder.out)
    opt = torch.optim.AdamW([*enc.parameters(), *dec.parameters()], lr=cfg["lr"], weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    n = len(crops)
    t0 = time.time()
    for ep in range(epochs):
        tot = 0.0
        for _ in range(max(1, n // 256)):
            ix = rng.integers(0, n, 256)
            x = torch.from_numpy(crops[ix].astype(np.float32))
            inp, keep = _mask_patches(x, cfg["mask_ratio"], g)
            rec = dec(enc(inp))
            hidden = 1 - keep
            loss = (((rec - x) ** 2) * hidden).sum() / hidden.sum().clamp_min(1) / 3
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
        log(f"pretrain epoch {ep + 1}/{epochs} masked MSE {tot / max(1, n // 256):.4f} ({time.time() - t0:.0f}s)")
    return {k: v.clone() for k, v in enc.state_dict().items()}


def fit(d: dict[str, np.ndarray], ix: np.ndarray, encoder: dict[str, torch.Tensor] | None, seed: int) -> RiskNet:
    cfg = config.policy()["risk"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = RiskNet()
    if encoder is not None:
        net.encoder.load_state_dict(encoder)
    f = d["features"][ix].astype(np.float32)
    net.f_mean.copy_(torch.from_numpy(f.mean(0)))
    net.f_std.copy_(torch.from_numpy(f.std(0) + 1e-3))
    crops = d["crops"][ix].astype(np.float32)
    prim = _prim_onehot(d["prim"][ix])
    y = _labels({k: v[ix] for k, v in d.items()})
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=1e-3)
    bs = cfg["batch"]
    steps = max(cfg["epochs"] * int(np.ceil(len(ix) / bs)), cfg["min_steps"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps)
    lossf = nn.BCEWithLogitsLoss()
    net.train()
    for _ in range(steps):
        b = rng.integers(0, len(ix), bs)
        c = crops[b]
        flip = rng.random(len(b)) < 0.5
        c[flip] = c[flip][..., ::-1]
        logits = net(torch.from_numpy(np.ascontiguousarray(c)), torch.from_numpy(f[b]), torch.from_numpy(prim[b]))
        loss = lossf(logits, torch.from_numpy(y[b]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return net.eval()


@torch.no_grad()
def predict(net: RiskNet, d: dict[str, np.ndarray]) -> np.ndarray:
    out = []
    for i in range(0, len(d["prim"]), 512):
        sl = slice(i, i + 512)
        out.append(
            torch.sigmoid(
                net(
                    torch.from_numpy(d["crops"][sl].astype(np.float32)),
                    torch.from_numpy(d["features"][sl].astype(np.float32)),
                    torch.from_numpy(_prim_onehot(d["prim"][sl])),
                )
            ).numpy()
        )
    return np.concatenate(out)


def metrics(net: RiskNet, d: dict[str, np.ndarray]) -> dict[str, Any]:
    p = predict(net, d)
    y = _labels(d)
    return {
        h: {"auc": auc(y[:, i], p[:, i]), "ece": ece(y[:, i], p[:, i]), "brier": float(np.mean((p[:, i] - y[:, i]) ** 2)), "rate": float(y[:, i].mean())}
        for i, h in enumerate(HEADS)
    }


def study(
    tr: dict[str, np.ndarray], va: dict[str, np.ndarray], unlabelled: np.ndarray, budgets: list[int], seeds: list[int], log=print
) -> tuple[dict[str, Any], RiskNet]:
    cfg = config.policy()["risk"]
    t0 = time.time()
    enc = pretrain(unlabelled, cfg["pretrain_epochs"], log=log)
    # budgets are counted in whole orders, so all counterfactual attempts of an order stay together
    orders = np.unique(tr["order"])
    per_order = len(tr["prim"]) / len(orders)
    rows = []
    models: dict[tuple[str, int, int], RiskNet] = {}
    for budget in budgets:
        n_orders = min(len(orders), int(round(budget / per_order)))
        for s in seeds:
            pick = np.random.default_rng(1000 + s).permutation(orders)[:n_orders]
            ix = np.nonzero(np.isin(tr["order"], pick))[0]
            for variant, e in (("scratch", None), ("pretrained", enc)):
                net = fit(tr, ix, e, s)
                m = metrics(net, va)
                rows.append({"budget": int(len(ix)), "seed": s, "variant": variant, **{f"{h}_{k}": v for h in HEADS for k, v in m[h].items()}})
                models[(variant, budget, s)] = net
                log(
                    f"risk {variant:10s} {len(ix):6d} labels seed {s}: success AUC {_f(m['success']['auc'])}, "
                    f"collateral AUC {_f(m['collateral']['auc'])} ({time.time() - t0:.0f}s)"
                )
    top = budgets[-1]

    def score(v: str) -> float:
        r = [x for x in rows if x["variant"] == v and x["budget"] == max(y["budget"] for y in rows)]
        return float(np.mean([(x["success_auc"] or 0) + (x["collateral_auc"] or 0) for x in r]))

    chosen = "pretrained" if score("pretrained") >= score("scratch") else "scratch"
    final = models[(chosen, top, seeds[0])]
    return {"rows": rows, "chosen": chosen, "chosen_on": "validation orders, largest budget", "final": metrics(final, va)}, final
