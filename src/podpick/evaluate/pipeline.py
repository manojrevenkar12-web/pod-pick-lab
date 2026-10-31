"""The whole run, step by step: data, models, experiments, report, gate.

Each step writes into runs/<profile>/ and can be re-run on its own (`podpick <step>`).
`podpick all` runs them in order. The full profile's outputs are committed: models/, reports/,
docs/REPORT.md.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .. import config


def run_dir(profile: str) -> Path:
    d = config.ROOT / "runs" / profile
    d.mkdir(parents=True, exist_ok=True)
    return d


def models_dir(profile: str) -> Path:
    d = config.ROOT / "models" if profile == "full" else run_dir(profile) / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def exp(profile: str) -> dict[str, Any]:
    e = config.load("experiment")
    return {**e, **e["profiles"][profile]}


def seeds(profile: str, split: str, n: int) -> list[int]:
    base = config.load("experiment")["seed_base"][split]
    return list(range(base, base + n))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


# ---------------------------------------------------------------------- steps
def step_perception(profile: str) -> dict[str, Any]:
    import torch

    from ..sim.datasets import perception_data
    from ..train.perception import train

    e = exp(profile)
    rd = run_dir(profile)
    log("perception data")
    tr = perception_data(e["train_family"], seeds(profile, "perception_train", e["perception_scenes"]), e["max_targets_per_scene"], e["workers"])
    va = perception_data(e["train_family"], seeds(profile, "perception_val", e["perception_val_scenes"]), e["max_targets_per_scene"], e["workers"])
    log(f"perception: {len(tr['query'])} training targets in {len(tr['images'])} images")
    torch.set_num_threads(e["workers"])
    net, metrics = train(tr, va, e["perception_epochs"], log=log)
    torch.save(net.state_dict(), models_dir(profile) / "perception.pt")
    write_json(rd / "perception.json", metrics)
    log(f"perception: IoU {metrics['mask_iou_mean']:.3f}, occlusion accuracy {metrics['occlusion_accuracy']:.3f}")
    return metrics


def step_picks(profile: str) -> None:
    from ..sim.datasets import pick_data, unlabelled_crops

    e = exp(profile)
    rd = run_dir(profile)
    md = str(models_dir(profile))
    n_val = max(20, e["pick_orders"] // 6)
    log(f"pick data: {e['pick_orders']} orders x {e['attempts_per_order']} attempts")
    tr = pick_data(e["train_family"], seeds(profile, "picks", e["pick_orders"]), e["attempts_per_order"], md, e["workers"])
    va = pick_data(e["train_family"], seeds(profile, "picks_val", n_val), e["attempts_per_order"], md, e["workers"])
    np.savez_compressed(rd / "picks_train.npz", **tr)
    np.savez_compressed(rd / "picks_val.npz", **va)
    log("unlabelled crops")
    un = unlabelled_crops(e["train_family"], seeds(profile, "unlabelled", e["unlabelled_scenes"]), md, e["workers"])
    np.savez_compressed(rd / "unlabelled.npz", crops=un)
    stats = {k: float(np.mean(tr[k])) for k in ("success", "collateral", "drop", "sealed")}
    stats.update(attempts=int(len(tr["prim"])), val_attempts=int(len(va["prim"])), unlabelled_crops=int(len(un)))
    write_json(rd / "picks.json", stats)
    log(f"pick data: {stats}")


def step_risk(profile: str) -> dict[str, Any]:
    import torch

    from ..train.risk import study

    e = exp(profile)
    rd = run_dir(profile)
    torch.set_num_threads(e["workers"])
    tr = dict(np.load(rd / "picks_train.npz"))
    va = dict(np.load(rd / "picks_val.npz"))
    un = np.load(rd / "unlabelled.npz")["crops"]
    res, net = study(tr, va, un, e["label_budgets"], e["seeds"], log=log)
    torch.save(net.state_dict(), models_dir(profile) / "risk.pt")
    write_json(rd / "risk.json", res)
    return res


def step_evaluate(profile: str, families: list[str] | None = None) -> dict[str, Any]:
    from .experiment import run_families

    e = exp(profile)
    fams = families or e["test_families"]
    res = run_families(fams, e["test_orders"], str(models_dir(profile)), e["workers"], log=log)
    path = run_dir(profile) / "evaluation.json"
    old = json.loads(path.read_text()) if path.exists() and families else {}
    old.update(res)
    write_json(path, old)
    return res


def step_native(profile: str) -> dict[str, Any]:
    from .native_bench import bench

    res = bench(str(models_dir(profile)), n_orders=12 if profile == "quick" else 120)
    write_json(run_dir(profile) / "native.json", res)
    log(f"native monitor: {res}")
    return res


def step_tune(profile: str) -> dict[str, Any]:
    from .tuning import tune

    return tune(profile)


def step_report(profile: str) -> None:
    from .report import write_reports

    write_reports(profile)


def step_gate(profile: str) -> bool:
    from .gate import gate

    return gate(profile)


STEPS = {
    "perception": step_perception,
    "picks": step_picks,
    "risk": step_risk,
    "tune": step_tune,
    "evaluate": step_evaluate,
    "native": step_native,
    "report": step_report,
    "gate": step_gate,
}


def run_all(profile: str) -> bool:
    for name in ("perception", "picks", "risk", "tune", "evaluate", "native", "report"):
        t = time.time()
        STEPS[name](profile)
        log(f"step {name} done in {time.time() - t:.0f}s")
    return step_gate(profile)
