"""What passes between the pick cell and a robot (simulated or not).

The pick cell sees an `Observation` and drives a `Driver`. Neither carries any truth: no item
poses, masses, friction or seal quality. `tests/test_isolation.py` checks that the cell never
imports the simulator or the evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class TargetQuery:
    """What the order says about the item: catalogue dimensions (sorted, metres), colour, weight."""

    dims: tuple[float, float, float]
    albedo: float
    mass: float


@dataclass
class Observation:
    depth: np.ndarray  # (H, W) float32, metres along the optical axis, 0 = no return
    intensity: np.ndarray  # (H, W) float32, 0..1
    query: TargetQuery
    band_z: float  # band height, from the pod model


@dataclass
class Tick:
    """One control tick (1 / monitor_hz seconds) of motion."""

    tool: np.ndarray  # tool tip position (commanded = measured; the arm is stiff)
    contact: bool  # contact switch in the cup
    pressure: float  # lowest vacuum reading during the tick, 0..1
    frame: np.ndarray | None = None  # monitor depth frame (robot self-filtered), or None
    reached: bool = False


@dataclass
class Attempt:
    """What the cell did on one attempt; filled in by the cell, read by the evaluation."""

    candidate: int
    primitive: str
    sealed: bool = False
    aborted: str | None = None
    actions: list[str] = field(default_factory=list)
    pressures: list[float] = field(default_factory=list)
    signals: list[int] = field(default_factory=list)


class Driver(Protocol):
    def observe(self) -> Observation: ...

    def open_band(self) -> None: ...

    def close_band(self) -> None: ...

    def place_tool(self, pos: np.ndarray, direction: np.ndarray) -> None: ...

    def move(self, target: np.ndarray, speed: float, *, stop_on_contact: bool = False, frame: bool = False) -> Tick: ...

    def vacuum(self, on: bool) -> float: ...

    def frame(self) -> np.ndarray: ...

    @property
    def clock(self) -> float: ...
