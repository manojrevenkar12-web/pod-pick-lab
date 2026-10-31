"""The pick planner as a service: a bin image and an order in, a plan out; and a monitor session
that takes extraction frames at 10 Hz and answers continue / slow / abort.

Arrays travel as base64 of little-endian float32 (row-major) with an explicit shape.
The monitor sessions use the C++ implementation when it is built, the Python reference otherwise.
"""

from __future__ import annotations

import base64
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .. import config
from ..control.monitor import ACTIONS, Monitor, MonitorConfig
from ..geometry import monitor_camera, scene_camera
from ..interface import Observation, TargetQuery

app = FastAPI(title="pod-pick planner", version="0.1.0")
_state: dict[str, Any] = {
    "cells": {},
    "sessions": {},
    "lock": threading.Lock(),
    "metrics": {"plans": 0, "handoffs": 0, "frames": 0, "aborts": 0, "plan_ms_sum": 0.0},
}
SESSION_TTL_S = 120.0


class Array(BaseModel):
    shape: list[int]
    data: str = Field(description="base64 of little-endian float32, row-major")

    def array(self) -> np.ndarray:
        raw = base64.b64decode(self.data)
        n = int(np.prod(self.shape))
        if len(raw) != 4 * n:
            raise HTTPException(422, f"array data has {len(raw)} bytes, shape {self.shape} needs {4 * n}")
        return np.frombuffer(raw, dtype="<f4").reshape(self.shape).copy()


class Query(BaseModel):
    dims: tuple[float, float, float]
    albedo: float = Field(ge=0, le=1)
    mass: float = Field(gt=0)


class PlanRequest(BaseModel):
    depth: Array
    intensity: Array
    query: Query
    band_z: float
    policy: str = "learned-closed"


class MonitorStart(BaseModel):
    reference: Array
    mask: Array
    tool: tuple[float, float, float]
    extrude: float = 0.0


class MonitorFrame(BaseModel):
    frame: Array
    tool: tuple[float, float, float]
    pressure: float


def _cell(policy: str):
    from ..cell import load_cell

    if policy not in config.policy()["policies"]:
        raise HTTPException(404, f"unknown policy {policy}")
    with _state["lock"]:
        if policy not in _state["cells"]:
            models = os.environ.get("PODPICK_MODELS")
            _state["cells"][policy] = load_cell(policy, Path(models) if models else None)
        return _state["cells"][policy]


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict[str, Any]:
    try:
        _cell("learned-closed")
    except Exception as e:  # noqa: BLE001 - report why we are not ready
        raise HTTPException(503, f"models not loaded: {e}") from e
    native = False
    try:
        from ..control.native import lib

        lib()
        native = True
    except Exception:  # noqa: BLE001
        native = False
    return {"status": "ready", "native_monitor": native}


@app.post("/v1/plan")
def plan(req: PlanRequest) -> dict[str, Any]:
    from ..decide.planner import plan_heuristic, plan_learned
    from ..grasp.candidates import generate

    cam = scene_camera()
    depth, inten = req.depth.array(), req.intensity.array()
    if depth.shape != (cam.height, cam.width) or inten.shape != depth.shape:
        raise HTTPException(422, f"images must be {cam.height} x {cam.width}")
    t0 = time.perf_counter()
    cell = _cell(req.policy)
    obs = Observation(depth=depth, intensity=inten, query=TargetQuery(tuple(req.query.dims), req.query.albedo, req.query.mass), band_z=req.band_z)
    perc = cell.perception(obs)
    c = generate(obs, perc["mask"], perc["occlusion"])
    if int(perc["mask"].sum()) < config.policy()["perception"]["min_mask_pixels"]:
        reason = "not_found"
        options: list[Any] = []
    else:
        pl = plan_heuristic(c, perc["occlusion"]) if cell.p["scorer"] == "heuristic" else plan_learned(c, cell.risk(c))
        reason, options = pl.handoff, pl.options
    ms = (time.perf_counter() - t0) * 1000
    m = _state["metrics"]
    m["plans"] += 1
    m["handoffs"] += int(reason is not None)
    m["plan_ms_sum"] += ms
    return {
        "policy": req.policy,
        "decision": "handoff" if reason else "pick",
        "reason": reason,
        "occlusion": dict(zip(("unobstructed", "below_others", "blocked"), [round(float(x), 4) for x in perc["occlusion"]], strict=True)),
        "mask_pixels": int(perc["mask"].sum()),
        "options": [
            {
                "pixel": [int(v) for v in c.pixels[o.candidate]],
                "point": [round(float(v), 4) for v in c.points[o.candidate]],
                "normal": [round(float(v), 4) for v in c.normals[o.candidate]],
                "primitive": o.primitive,
                "expected_cost_s": None if np.isnan(o.cost) else round(o.cost, 3),
                "p": None if o.p is None else dict(zip(("success", "collateral", "drop"), [round(x, 4) for x in o.p], strict=True)),
            }
            for o in options[: config.policy()["decision"]["max_attempts"]]
        ],
        "plan_ms": round(ms, 2),
    }


def _gc() -> None:
    now = time.time()
    for k in [k for k, v in _state["sessions"].items() if now - v["seen"] > SESSION_TTL_S]:
        del _state["sessions"][k]


@app.post("/v1/monitor")
def monitor_start(req: MonitorStart) -> dict[str, Any]:
    cam = monitor_camera()
    ref, mask = req.reference.array(), req.mask.array() > 0.5
    if ref.shape != (cam.height, cam.width) or mask.shape != ref.shape:
        raise HTTPException(422, f"frames must be {cam.height} x {cam.width}")
    cfg = MonitorConfig.from_config()
    try:
        from ..control.native import NativeMonitor

        mon: Any = NativeMonitor(cfg, cam, ref, mask, np.array(req.tool), extrude=req.extrude)
        impl = "native"
    except Exception:  # noqa: BLE001 - the reference gives the same answers, only slower
        mon = Monitor(cfg, cam, ref, mask, np.array(req.tool), extrude=req.extrude)
        impl = "python"
    sid = uuid.uuid4().hex
    with _state["lock"]:
        _gc()
        _state["sessions"][sid] = {"monitor": mon, "seen": time.time()}
    return {"session": sid, "implementation": impl}


@app.post("/v1/monitor/{sid}/frame")
def monitor_frame(sid: str, req: MonitorFrame) -> dict[str, Any]:
    s = _state["sessions"].get(sid)
    if s is None:
        raise HTTPException(404, "no such monitor session (expired after 120 s without frames?)")
    f = req.frame.array()
    t0 = time.perf_counter()
    act, sig = s["monitor"].step(f, np.array(req.tool), req.pressure)
    s["seen"] = time.time()
    _state["metrics"]["frames"] += 1
    _state["metrics"]["aborts"] += int(act == 2)
    return {"action": ACTIONS[act], "signal": sig, "us": round((time.perf_counter() - t0) * 1e6, 1)}


@app.delete("/v1/monitor/{sid}")
def monitor_end(sid: str) -> dict[str, str]:
    _state["sessions"].pop(sid, None)
    return {"status": "closed"}


@app.get("/metrics")
def metrics() -> Any:
    from fastapi.responses import PlainTextResponse

    m = _state["metrics"]
    lines = [
        "# TYPE podpick_plans_total counter",
        f"podpick_plans_total {m['plans']}",
        "# TYPE podpick_handoffs_total counter",
        f"podpick_handoffs_total {m['handoffs']}",
        "# TYPE podpick_monitor_frames_total counter",
        f"podpick_monitor_frames_total {m['frames']}",
        "# TYPE podpick_monitor_aborts_total counter",
        f"podpick_monitor_aborts_total {m['aborts']}",
        "# TYPE podpick_plan_ms_sum counter",
        f"podpick_plan_ms_sum {m['plan_ms_sum']:.3f}",
        "# TYPE podpick_monitor_sessions gauge",
        f"podpick_monitor_sessions {len(_state['sessions'])}",
    ]
    return PlainTextResponse("\n".join(lines) + "\n")
