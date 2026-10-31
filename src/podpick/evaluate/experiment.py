"""Paired evaluation: every policy picks the same orders from the same start state.

For each order (scene, target) the world is built once and snapshotted; each policy is run from
that snapshot, with the same sensor noise stream. The truth then says what happened: the target
out, a neighbour out (collateral), the target dropped outside the bin, the wrong item.
"""

from __future__ import annotations

import time
import zlib
from typing import Any

import numpy as np

from .. import config
from ..sim.datasets import _chunks, _pool_map, pick_order
from ..sim.scene import sample_scene
from ..sim.world import World

POLICIES = ("heuristic-open", "heuristic-closed", "learned-open", "learned-closed")
OCC = ("unobstructed", "below_others", "blocked")


def family_seeds(family: str, n: int) -> list[int]:
    base = config.load("experiment")["seed_base"]["test"]
    return [base + (zlib.crc32(family.encode()) % 997) * 10_000 + i for i in range(n)]


def run_order(w: World, cells: dict[str, Any], seed: int, family: str) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed + 17)
    t = pick_order(w, rng)
    if t is None:
        return []
    occ = w.occlusion_class(t)
    start = w.snapshot()
    rows = []
    settle = config.load("world")["outcome"]["settle_after_s"]
    for name, cell in cells.items():
        w.restore(start)
        r = cell.pick(w)
        w.settle(settle)
        o = w.outcome()
        first = r.plan["options"][0] if r.plan.get("options") else None
        rows.append(
            {
                "family": family,
                "seed": seed,
                "policy": name,
                "target": t,
                "occlusion_true": occ,
                "material": w.spec.items[t].material,
                "status": r.status,
                "reason": r.reason,
                "attempts": len(r.attempts),
                "robot_s": r.robot_s,
                "aborted": r.status == "aborted",
                "picked": bool(o["target_out"]),
                "collateral": int(o["collateral"]),
                "dropped": bool(o["target_dropped_out"]),
                "wrong_item": bool(o["wrong_item"]),
                "disturbed": int(o["disturbed"]),
                "occlusion_pred": int(np.argmax(r.plan["occlusion"])) if r.plan.get("occlusion") else -1,
                "first_p": first["p"] if first else None,
                "first_primitive": first["primitive"] if first else None,
                "monitor_signals": [s for a in r.attempts for s in a.signals],
                "monitor_actions": [s for a in r.attempts for s in a.actions],
            }
        )
    return rows


def _job(job: tuple[str, list[int], str, tuple[str, ...]]) -> list[dict[str, Any]]:
    import torch

    from ..cell import load_cell

    torch.set_num_threads(1)
    family, seeds, models_dir, policies = job
    from pathlib import Path

    cells = {p: load_cell(p, Path(models_dir)) for p in policies}
    rows: list[dict[str, Any]] = []
    for s in seeds:
        rows.extend(run_order(World(sample_scene(family, s)), cells, s, family))
    return rows


def run_family(family: str, n: int, models_dir: str, workers: int, policies: tuple[str, ...] = POLICIES) -> list[dict[str, Any]]:
    seeds = family_seeds(family, n)
    parts = _pool_map(_job, [(family, ch, models_dir, policies) for ch in _chunks(seeds, workers * 3)], workers)
    return [r for p in parts for r in p]


# ---------------------------------------------------------------------- metrics
def costs(rows: list[dict[str, Any]]) -> np.ndarray:
    d = config.policy()["decision"]
    out = []
    for r in rows:
        c = d["robot_second"] * r["robot_s"]
        c += 0.0 if r["picked"] else d["human_pick_s"]
        c += d["collateral_s"] * r["collateral"] + d["drop_s"] * r["dropped"] + d["abort_s"] * r["aborted"]
        out.append(c)
    return np.array(out)


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"orders": 0}
    d = config.policy()["decision"]
    pick = np.array([r["picked"] for r in rows])
    attempted = np.array([r["attempts"] > 0 for r in rows])
    robot = np.array([r["robot_s"] for r in rows])
    hours = (robot.sum() + d["overhead_s"] * attempted.sum()) / 3600
    return {
        "orders": n,
        "pick_rate": float(pick.mean()),
        "attempt_rate": float(attempted.mean()),
        "success_when_attempted": float(pick[attempted].mean()) if attempted.any() else None,
        "collateral_per_100": 100 * float(np.mean([r["collateral"] for r in rows])),
        "orders_with_collateral": float(np.mean([r["collateral"] > 0 for r in rows])),
        "drop_rate": float(np.mean([r["dropped"] for r in rows])),
        "wrong_item_rate": float(np.mean([r["wrong_item"] for r in rows])),
        "abort_rate": float(np.mean([r["aborted"] for r in rows])),
        "handoff_rate": float(1 - pick.mean()),
        "robot_s_per_order": float(robot.mean()),
        "uph": float(pick.sum() / hours) if hours > 0 else 0.0,
        "cost_per_order": float(costs(rows).mean()),
    }


def paired(rows: list[dict[str, Any]], a: str, b: str, key: str, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Mean difference b - a over the same orders, with a bootstrap 95% interval."""
    ra = {r["seed"]: r for r in rows if r["policy"] == a}
    rb = {r["seed"]: r for r in rows if r["policy"] == b}
    common = sorted(set(ra) & set(rb))
    if not common:
        return {}
    if key == "cost":
        va = costs([ra[s] for s in common])
        vb = costs([rb[s] for s in common])
    else:
        va = np.array([float(ra[s][key]) for s in common])
        vb = np.array([float(rb[s][key]) for s in common])
    diff = vb - va
    rng = np.random.default_rng(seed)
    boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)])
    return {"diff": float(diff.mean()), "lo": float(np.quantile(boots, 0.025)), "hi": float(np.quantile(boots, 0.975)), "n": len(common)}


def analyse(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pols = sorted({r["policy"] for r in rows}, key=lambda p: POLICIES.index(p) if p in POLICIES else 99)
    out: dict[str, Any] = {"policies": {p: summarise([r for r in rows if r["policy"] == p]) for p in pols}}
    out["by_occlusion"] = {p: {OCC[k]: summarise([r for r in rows if r["policy"] == p and r["occlusion_true"] == k]) for k in range(3)} for p in pols}
    base = "heuristic-open"
    if base in pols:
        out["vs_heuristic_open"] = {p: {k: paired(rows, base, p, k) for k in ("picked", "collateral", "dropped", "cost")} for p in pols if p != base}
    if "learned-open" in pols and "learned-closed" in pols:
        out["monitor_effect_learned"] = {k: paired(rows, "learned-open", "learned-closed", k) for k in ("picked", "collateral", "dropped", "cost")}
    if "heuristic-open" in pols and "heuristic-closed" in pols:
        out["monitor_effect_heuristic"] = {k: paired(rows, "heuristic-open", "heuristic-closed", k) for k in ("picked", "collateral", "dropped", "cost")}
    # what the learned model promised for its first choice, against what happened (open loop)
    lo = [r for r in rows if r["policy"] == "learned-open" and r["first_p"] is not None and r["attempts"] > 0]
    if lo:
        ps = np.array([r["first_p"][0] for r in lo])
        pc = np.array([r["first_p"][1] for r in lo])
        out["learned_promises"] = {
            "attempts": len(lo),
            "predicted_success": float(ps.mean()),
            "observed_success": float(np.mean([r["picked"] for r in lo])),
            "predicted_collateral": float(pc.mean()),
            "observed_collateral": float(np.mean([r["collateral"] > 0 for r in lo])),
        }
    occ_true = np.array([r["occlusion_true"] for r in rows if r["policy"] == pols[0]])
    occ_pred = np.array([r["occlusion_pred"] for r in rows if r["policy"] == pols[0]])
    ok = occ_pred >= 0
    out["occlusion_accuracy"] = float((occ_true[ok] == occ_pred[ok]).mean()) if ok.any() else None
    return out


def run_families(families: list[str], n: int, models_dir: str, workers: int, log=print) -> dict[str, Any]:
    res = {}
    for fam in families:
        t = time.time()
        rows = run_family(fam, n, models_dir, workers)
        res[fam] = {"analysis": analyse(rows), "rows": rows}
        s = res[fam]["analysis"]["policies"]
        log(
            f"{fam}: "
            + "; ".join(f"{p} pick {v['pick_rate']:.2f} coll/100 {v['collateral_per_100']:.1f} cost {v['cost_per_order']:.2f}" for p, v in s.items())
            + f" ({time.time() - t:.0f}s)"
        )
    return res
