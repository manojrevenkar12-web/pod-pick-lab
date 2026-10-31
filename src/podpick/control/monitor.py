"""The extraction monitor: watch the bin at 10 Hz while the item comes out, and react.

Reference implementation. `cpp/monitor.cpp` is the real-time implementation and must return the
same signal and action on every frame (`tests/test_native.py`); keep the arithmetic in the same
order in both.

Each frame is compared with the frame taken just after the seal. Pixels where the target was,
and where the target is expected to be now (its points moved with the tool), are ignored; the
robot's own links are already filtered out of the frames. A neighbour pixel that comes closer to
the camera by more than `move_m` has moved toward the front: that is how an item riding on the
target, or dragged by it, shows up before it falls out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import config
from ..geometry import Camera

CONTINUE, SLOW, ABORT = 0, 1, 2
ACTIONS = ("continue", "slow", "abort")


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(r):
        m = out.copy()
        m[1:, :] |= out[:-1, :]
        m[:-1, :] |= out[1:, :]
        m[:, 1:] |= out[:, :-1]
        m[:, :-1] |= out[:, 1:]
        out = m
    return out


def refine_mask(ref: np.ndarray, mask: np.ndarray, seal_px: tuple[int, int], cam: Camera, max_extent: float, step_m: float) -> np.ndarray:
    """Grow the target mask over the sealed surface: pixels depth-connected to the cup's pixel.

    Perception under-segments: parts of the item it missed would look like a neighbour moving when
    the item comes out. Growing from the seal pixel across depth steps smaller than `step_m`, no
    further than the item's catalogue size from the seal point, recovers the rest of the face.
    """
    H, W = ref.shape
    v0, u0 = seal_px
    out = mask.copy()
    if not (0 <= v0 < H and 0 <= u0 < W) or ref[v0, u0] <= 0:
        return out
    pts = cam.points(ref.astype(np.float64))
    p0 = pts[v0, u0]
    seen = np.zeros((H, W), bool)
    seen[v0, u0] = True
    stack = [(v0, u0)]
    while stack:
        v, u = stack.pop()
        out[v, u] = True
        for dv, du in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = v + dv, u + du
            if 0 <= a < H and 0 <= b < W and not seen[a, b] and ref[a, b] > 0:
                if abs(float(ref[a, b]) - float(ref[v, u])) < step_m and np.linalg.norm(pts[a, b] - p0) <= max_extent:
                    seen[a, b] = True
                    stack.append((a, b))
    return out


@dataclass
class MonitorConfig:
    move_m: float
    dilate_px: int
    warn_px: int
    abort_px: int
    abort_ticks: int
    pressure_slow: float
    pressure_abort: float
    inside_y: float
    extrude_steps: int
    sag_m: float
    sag_steps: int
    follow: float

    @classmethod
    def from_config(cls) -> MonitorConfig:
        m = config.policy()["monitor"]
        return cls(**{k: m[k] for k in cls.__dataclass_fields__})


@dataclass
class Monitor:
    cfg: MonitorConfig
    cam: Camera
    ref: np.ndarray  # (H, W) float32 depth just after the seal
    mask: np.ndarray  # (H, W) bool, target at monitor resolution
    tool0: np.ndarray  # tool tip at the seal
    extrude: float = 0.0  # the item's catalogue length: its hidden body extends this far behind the face
    points: np.ndarray = field(init=False)
    static: np.ndarray = field(init=False)
    over: int = 0
    slowed: bool = False

    def __post_init__(self) -> None:
        ref = self.ref.astype(np.float64)
        vs, us = np.nonzero(self.mask & (self.ref > 0))
        d = ref[vs, us]
        c = self.cam
        x = (us.astype(np.float64) + 0.5 - c.cx) / c.f * d + c.position[0]
        z = -(vs.astype(np.float64) + 0.5 - c.cy) / c.f * d + c.position[2]
        y = d + c.position[1]
        # the visible face, and copies of it pushed back along +y: the parts of the item the camera
        # cannot see yet but will once it comes out
        # The held item also sags on the compliant cup: copies lowered by up to sag_m.
        n, ns = self.cfg.extrude_steps, self.cfg.sag_steps
        offs_y = [(self.extrude * k) / n for k in range(n + 1)] if n > 0 else [0.0]
        offs_z = [(self.cfg.sag_m * j) / ns for j in range(ns + 1)] if ns > 0 else [0.0]
        xs, ys, zs = [], [], []
        for oz in offs_z:
            for oy in offs_y:
                xs.append(x)
                ys.append(y + oy)
                zs.append(z - oz)
        self.points = np.stack([np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)], axis=1)
        self.static = dilate(self.mask, self.cfg.dilate_px)
        self.tool0 = np.asarray(self.tool0, dtype=np.float64)

    def signal(self, frame: np.ndarray, tool: np.ndarray) -> int:
        c = self.cam
        H, W = self.ref.shape
        disp = np.asarray(tool, dtype=np.float64) - self.tool0
        p = self.points + disp
        rel_x = p[:, 0] - c.position[0]
        rel_y = p[:, 1] - c.position[1]
        rel_z = p[:, 2] - c.position[2]
        pred = np.zeros((H, W), dtype=bool)
        ok = rel_y > 1e-6
        u = np.floor(rel_x[ok] / rel_y[ok] * c.f + c.cx).astype(np.int64)
        v = np.floor(-rel_z[ok] / rel_y[ok] * c.f + c.cy).astype(np.int64)
        inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        pred[v[inb], u[inb]] = True
        pred = dilate(pred, self.cfg.dilate_px)
        ref = self.ref.astype(np.float64)
        cur = frame.astype(np.float64)
        neighbour = ~(self.static | pred) & (self.ref > 0) & (frame > 0)
        # A neighbour that only gets nudged moves a centimetre or two and stops; one that rides out
        # on the target keeps pace with the tool. Count pixels that have come forward by at least
        # `follow` times the tool's own travel toward the camera (and never less than move_m).
        pulled = self.tool0[1] - float(tool[1])
        need = max(self.cfg.move_m, self.cfg.follow * pulled)
        moved = neighbour & ((ref - cur) > need)
        return int(moved.sum())

    def step(self, frame: np.ndarray, tool: np.ndarray, pressure: float) -> tuple[int, int]:
        """One monitor frame: returns (action, signal)."""
        s = self.signal(frame, tool)
        cfg = self.cfg
        self.over = self.over + 1 if s >= cfg.abort_px else 0
        if self.over >= cfg.abort_ticks:
            return ABORT, s
        if pressure < cfg.pressure_abort and float(tool[1]) > cfg.inside_y:
            return ABORT, s
        if s >= cfg.warn_px or pressure < cfg.pressure_slow:
            self.slowed = True
        return (SLOW if self.slowed else CONTINUE), s
