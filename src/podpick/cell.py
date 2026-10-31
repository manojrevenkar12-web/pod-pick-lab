"""The pick cell: one order in, the item out or handed to a person.

observe -> find the target -> suction candidates -> choose (heuristic or learned) -> open the band
-> approach until the cup touches -> seal -> extract along the primitive, with or without the
monitor -> picked, or put back and handed over.

The cell talks only to a `Driver`. It never sees item poses or outcomes; the evaluation judges
what it did from the simulator's truth afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import config
from .control.monitor import ABORT, ACTIONS, SLOW, Monitor, MonitorConfig, refine_mask
from .decide.planner import Option, Plan, plan_heuristic, plan_learned
from .geometry import monitor_camera
from .grasp.candidates import FEATURES, Candidates, generate
from .grasp.risk import RiskModel
from .interface import Attempt, Driver
from .perception.model import Perception


@dataclass
class PickResult:
    status: str  # picked | handoff | aborted | lost
    reason: str | None
    attempts: list[Attempt] = field(default_factory=list)
    robot_s: float = 0.0
    plan: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "robot_s": round(self.robot_s, 3),
            "attempts": [
                {
                    "candidate": a.candidate,
                    "primitive": a.primitive,
                    "sealed": a.sealed,
                    "aborted": a.aborted,
                    "actions": a.actions,
                    "signals": a.signals,
                    "pressures": [round(p, 3) for p in a.pressures],
                }
                for a in self.attempts
            ],
            "plan": self.plan,
        }


def downsample_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    H, W = mask.shape
    h, w = shape
    fy, fx = H // h, W // w
    return mask[: h * fy, : w * fx].reshape(h, fy, w, fx).any(axis=(1, 3))


class PickCell:
    def __init__(self, perception: Perception, risk: RiskModel | None, policy: str, *, native_monitor: bool = False):
        self.perception = perception
        self.risk = risk
        self.policy = policy
        self.p = config.policy()["policies"][policy]
        if self.p["scorer"] == "learned" and risk is None:
            raise ValueError(f"policy {policy} needs a risk model")
        self.robot = config.robot()
        self.cfg = config.policy()
        self.native = native_monitor

    # ------------------------------------------------------------------ planning
    def plan(self, driver: Driver) -> tuple[Any, dict[str, Any], Candidates, Plan]:
        obs = driver.observe()
        perc = self.perception(obs)
        if int(perc["mask"].sum()) < self.cfg["perception"]["min_mask_pixels"]:
            c = generate(obs, np.zeros_like(perc["mask"]), perc["occlusion"])
            return obs, perc, c, Plan([], "not_found", self.cfg["decision"]["human_pick_s"])
        c = generate(obs, perc["mask"], perc["occlusion"])
        if self.p["scorer"] == "heuristic":
            plan = plan_heuristic(c, perc["occlusion"])
        else:
            assert self.risk is not None
            plan = plan_learned(c, self.risk(c))
        return obs, perc, c, plan

    # ------------------------------------------------------------------ the pick
    def pick(self, driver: Driver) -> PickResult:
        t0 = driver.clock
        obs, perc, c, plan = self.plan(driver)
        summary = {
            "occlusion": [round(float(x), 3) for x in perc["occlusion"]],
            "mask_px": int(perc["mask"].sum()),
            "candidates": len(c),
            "options": [
                {
                    "candidate": o.candidate,
                    "primitive": o.primitive,
                    "cost": None if np.isnan(o.cost) else round(o.cost, 2),
                    "p": None if o.p is None else [round(x, 3) for x in o.p],
                }
                for o in plan.options[:5]
            ],
        }
        if plan.handoff:
            return PickResult("handoff", plan.handoff, robot_s=driver.clock - t0, plan=summary)
        driver.open_band()
        attempts: list[Attempt] = []
        tried: list[np.ndarray] = []
        status, reason = "handoff", "attempts"
        for opt in plan.options:
            if len(attempts) >= self.cfg["decision"]["max_attempts"]:
                break
            pt = c.points[opt.candidate]
            if any(np.linalg.norm(pt - t) < 0.015 for t in tried):
                continue
            tried.append(pt)
            a, outcome = self.attempt(driver, c, opt, perc["mask"])
            attempts.append(a)
            if outcome == "no_seal":
                continue
            status, reason = outcome, None if outcome == "picked" else outcome
            break
        driver.close_band()
        return PickResult(status, reason, attempts, robot_s=driver.clock - t0, plan=summary)

    def attempt(self, driver: Driver, c: Candidates, opt: Option, mask: np.ndarray, *, passive_monitor: bool = False) -> tuple[Attempt, str]:
        """One attempt. `passive_monitor` runs the monitor and logs its signal without acting on it
        (used to collect the logs the monitor thresholds are chosen from)."""
        mo, d = self.robot["motion"], self.cfg["decision"]
        a = Attempt(candidate=opt.candidate, primitive=opt.primitive)
        p, n = c.points[opt.candidate], c.normals[opt.candidate]
        approach = -n
        k = (-0.10 - p[1]) / n[1] if n[1] < -1e-3 else 0.12
        pre = p + n * k
        driver.place_tool(pre, approach)
        self._go(driver, p + n * mo["standoff"], mo["approach_speed"])
        touched = False
        for _ in range(20):
            tk = driver.move(p - n * mo["overtravel"], mo["approach_speed"] * 0.4, stop_on_contact=True)
            if tk.contact:
                touched = True
                break
            if tk.reached:
                break
        reading = driver.vacuum(True) if touched else 0.0
        a.pressures.append(reading)
        if reading < d["seal_ok"]:
            driver.vacuum(False)
            self._go(driver, pre, mo["approach_speed"])
            return a, "no_seal"
        a.sealed = True
        seal_pt = tk.tool
        prim = self.cfg["primitives"][opt.primitive]
        w1 = seal_pt + n * mo["retreat"]
        w2 = w1 + np.array([0.0, 0.0, min(prim["lift"], mo["max_lift"])])
        w3 = np.array([w2[0], mo["exit_y"], w2[2]])
        speed = mo["pull_speed"] * prim["speed"]
        monitor = None
        acting = bool(self.p["monitor"])
        if acting or passive_monitor:
            ref = driver.frame()
            mcam = monitor_camera()
            small = downsample_mask(mask, (mcam.height, mcam.width))
            length = float(c.features[opt.candidate, FEATURES.index("dim0")])  # longest catalogue dimension
            step = self.cfg["monitor"]["refine_step_m"]
            if step > 0:
                su, sv, _ = mcam.project(seal_pt[None])
                small = refine_mask(ref, small, (int(round(float(sv[0]))), int(round(float(su[0])))), mcam, length, step)
            mcfg = self.monitor_config()
            if self.native:
                from .control.native import NativeMonitor

                monitor = NativeMonitor(mcfg, mcam, ref, small, seal_pt, extrude=length)
            else:
                monitor = Monitor(mcfg, mcam, ref, small, seal_pt, extrude=length)
        path = [w1, w2, w3]
        tool = seal_pt
        last_pressure = reading
        for wp in path:
            for _ in range(60):
                tk = driver.move(wp, speed, frame=monitor is not None)
                tool, last_pressure = tk.tool, tk.pressure
                a.pressures.append(tk.pressure)
                if monitor is not None:
                    act, sig = monitor.step(tk.frame, tk.tool, tk.pressure)
                    a.actions.append(ACTIONS[act])
                    a.signals.append(sig)
                    if acting and act == SLOW:
                        speed = mo["pull_speed"] * prim["speed"] * mo["slow_factor"]
                    elif acting and act == ABORT:
                        a.aborted = "neighbour" if sig >= monitor.cfg.abort_px else "pressure"
                        self._put_back(driver, tool, seal_pt, w1, pre)
                        return a, "aborted"
                if tk.reached:
                    break
        held = last_pressure >= d["seal_ok"]
        return a, ("picked" if held else "lost")

    def monitor_config(self) -> MonitorConfig:
        """The monitor settings, with this policy's own abort threshold if it has one (docs/adr/0004)."""
        m = MonitorConfig.from_config()
        if "abort_px" in self.p:
            m.abort_px = int(self.p["abort_px"])
            m.warn_px = max(1, m.abort_px // 2)
        return m

    def _go(self, driver: Driver, target: np.ndarray, speed: float) -> None:
        for _ in range(60):
            if driver.move(target, speed).reached:
                return

    def _put_back(self, driver: Driver, tool: np.ndarray, seal_pt: np.ndarray, w1: np.ndarray, pre: np.ndarray) -> None:
        """Abort: take the item back to where it was sealed, let go, back off."""
        slow = self.robot["motion"]["pull_speed"] * self.robot["motion"]["slow_factor"]
        if tool[2] > w1[2] + 1e-4 or tool[1] < w1[1] - 1e-4:
            self._go(driver, np.array([tool[0], w1[1], tool[2]]), slow)
            self._go(driver, w1, slow)
        self._go(driver, seal_pt, slow)
        driver.vacuum(False)
        self._go(driver, pre, self.robot["motion"]["approach_speed"])


def load_cell(policy: str, models_dir: Any = None, *, native_monitor: bool = False) -> PickCell:
    """A cell with the committed models (models/*.pt)."""
    import torch

    from .grasp.risk import RiskNet
    from .perception.model import TargetNet

    root = config.ROOT / "models" if models_dir is None else models_dir
    tn = TargetNet()
    tn.load_state_dict(torch.load(root / "perception.pt", map_location="cpu", weights_only=True))
    risk = None
    if config.policy()["policies"][policy]["scorer"] == "learned":
        rn = RiskNet()
        rn.load_state_dict(torch.load(root / "risk.pt", map_location="cpu", weights_only=True))
        risk = RiskModel(rn)
    return PickCell(Perception(tn), risk, policy, native_monitor=native_monitor)
