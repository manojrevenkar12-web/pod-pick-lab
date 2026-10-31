"""Monitor abort thresholds from passive logs on validation orders (never on a test family).

During pick-data collection the monitor runs on every extraction without acting. For every sealed
attempt we keep the largest signal held over two consecutive frames (what `abort_ticks: 2` sees)
and whether a neighbour came out.

How often an abort pays depends on how often the attempts a policy makes would bring a neighbour
out: the free-space ranking attempts only items it sees as unobstructed; the learned cell attempts
only spots whose expected cost is below a person's. So each closed-loop policy gets its own
threshold, chosen on the validation attempts that policy would have made, by expected cost in the
same human-equivalent seconds the pick decision uses:

    saving(t) = share of collateral attempts stopped * collateral_s
              - share of clean picks stopped * (human_pick_s + abort_s)

The saving is optimistic (it assumes the abort comes in time); whether it does is measured only
in the paired closed-loop evaluation. The chosen values are copied into config/policy.yaml by hand.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import config
from .pipeline import log, models_dir, run_dir, write_json

THRESHOLDS = (4, 6, 8, 10, 12, 14, 18, 24, 32, 48, 64, 96, 128, 192)


def _attempted(policy: str, d: dict[str, np.ndarray], profile: str) -> np.ndarray:
    from ..grasp.candidates import FEATURES

    if policy.startswith("heuristic"):
        occ = d["features"][:, [FEATURES.index(f"occ{k}") for k in range(3)]]
        eligible = config.policy()["heuristic"]["eligible_classes"]
        return np.isin(np.argmax(occ, axis=1), eligible) & (d["prim"] == 0)
    import torch

    from ..decide.planner import expected_cost
    from ..grasp.risk import RiskNet
    from ..train.risk import predict

    net = RiskNet()
    net.load_state_dict(torch.load(models_dir(profile) / "risk.pt", map_location="cpu", weights_only=True))
    p = predict(net.eval(), d)
    return expected_cost(p[:, 0], p[:, 1], p[:, 2]) < config.policy()["decision"]["human_pick_s"]


def tune(profile: str) -> dict[str, Any]:
    d = dict(np.load(run_dir(profile) / "picks_val.npz"))
    cost = config.policy()["decision"]
    out: dict[str, Any] = {"source": "validation orders (picks_val), passive monitor", "policies": {}}
    for policy in ("heuristic-closed", "learned-closed"):
        sel = _attempted(policy, d, profile) & d["sealed"].astype(bool)
        coll = sel & d["collateral"].astype(bool)
        clean = sel & d["success"].astype(bool) & ~d["collateral"].astype(bool)
        rows = []
        for t in THRESHOLDS:
            hit = d["sig_pair"] >= t
            cs = float(hit[coll].mean()) if coll.any() else 0.0
            ks = float(hit[clean].mean()) if clean.any() else 0.0
            saving = (cs * coll.sum() * cost["collateral_s"] - ks * clean.sum() * (cost["human_pick_s"] + cost["abort_s"])) / max(int(sel.sum()), 1)
            rows.append({"abort_px": t, "collateral_stopped": cs, "clean_stopped": ks, "saving_s_per_attempt": float(saving)})
        best = max(rows, key=lambda r: r["saving_s_per_attempt"])
        out["policies"][policy] = {
            "sealed_attempts": int(sel.sum()),
            "collateral_attempts": int(coll.sum()),
            "clean_attempts": int(clean.sum()),
            "table": rows,
            "recommended_abort_px": best["abort_px"],
            "configured_abort_px": config.policy()["policies"][policy].get("abort_px", config.policy()["monitor"]["abort_px"]),
        }
        log(f"{policy}: {int(sel.sum())} sealed validation attempts, {int(coll.sum())} with collateral, {int(clean.sum())} clean")
        for r in rows:
            log(
                f"  abort_px {r['abort_px']:3d}: stops {r['collateral_stopped']:.3f} of collateral, {r['clean_stopped']:.3f} of clean, "
                f"saves {r['saving_s_per_attempt']:+.2f} s per attempt"
            )
        log(f"  recommended abort_px {best['abort_px']}")
    write_json(run_dir(profile) / "monitor_tuning.json", out)
    return out
