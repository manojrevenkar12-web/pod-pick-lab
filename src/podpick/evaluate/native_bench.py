"""The C++ monitor against the Python reference on real extraction frames: same answers, how fast."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .. import cell as cell_mod
from ..control import monitor as ref_mod
from ..control.native import NativeMonitor
from ..sim.datasets import pick_order
from ..sim.scene import sample_scene
from ..sim.world import World
from .experiment import family_seeds


class _Both:
    """Runs the reference and the native monitor on every frame and records both."""

    log: list[tuple[int, int, int, int, float, float]] = []

    def __init__(self, cfg, cam, ref, mask, tool0, extrude=0.0):
        self.py = ref_mod.Monitor(cfg, cam, ref, mask, tool0, extrude=extrude)
        self.cc = NativeMonitor(cfg, cam, ref, mask, tool0, extrude=extrude)
        self.cfg = cfg

    def step(self, frame, tool, pressure):
        t0 = time.perf_counter()
        a1, s1 = self.py.step(frame, tool, pressure)
        t1 = time.perf_counter()
        a2, s2 = self.cc.step(frame, tool, pressure)
        t2 = time.perf_counter()
        _Both.log.append((a1, s1, a2, s2, t1 - t0, t2 - t1))
        return a1, s1


def bench(models_dir: str, n_orders: int = 40, family: str = "mixed", min_frames: int = 100) -> dict[str, Any]:
    """Attempt the first candidate of each order (no eligibility check, so every seal is extracted)
    until at least `min_frames` monitor frames were compared, or `4 * n_orders` orders were tried."""
    from ..decide.planner import Option
    from ..grasp.candidates import generate

    _Both.log = []
    orig = cell_mod.Monitor
    cell_mod.Monitor = _Both  # type: ignore[assignment,misc]
    try:
        c = cell_mod.load_cell("heuristic-closed", Path(models_dir))
        tried = 0
        for s in family_seeds(family + "-native", 4 * n_orders):
            if tried >= n_orders and len(_Both.log) >= min_frames:
                break
            w = World(sample_scene(family, s))
            if pick_order(w, np.random.default_rng(s)) is None:
                continue
            tried += 1
            obs = w.observe()
            perc = c.perception(obs)
            cands = generate(obs, perc["mask"], perc["occlusion"])
            if len(cands) == 0:
                continue
            w.open_band()
            c.attempt(w, cands, Option(0, "lift", float("nan")), perc["mask"])
    finally:
        cell_mod.Monitor = orig  # type: ignore[misc]
    log = np.array(_Both.log, dtype=float)
    if len(log) == 0:
        return {"frames": 0}
    return {
        "frames": int(len(log)),
        "orders": tried,
        "signal_equal": float(np.mean(log[:, 1] == log[:, 3])),
        "action_equal": float(np.mean(log[:, 0] == log[:, 2])),
        "python_us_median": float(np.median(log[:, 4]) * 1e6),
        "native_us_median": float(np.median(log[:, 5]) * 1e6),
        "native_us_p99": float(np.quantile(log[:, 5], 0.99) * 1e6),
    }
