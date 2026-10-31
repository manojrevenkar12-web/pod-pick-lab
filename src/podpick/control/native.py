"""ctypes binding to the C++ monitor (cpp/monitor.cpp), with the same interface as `Monitor`."""

from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

import numpy as np

from .. import config
from ..geometry import Camera
from .monitor import MonitorConfig

CPP = config.ROOT / "cpp"
LIB = CPP / "build" / "libpodmonitor.so"


class _Cfg(ctypes.Structure):
    _fields_ = [
        ("move_m", ctypes.c_double),
        ("dilate_px", ctypes.c_int32),
        ("warn_px", ctypes.c_int32),
        ("abort_px", ctypes.c_int32),
        ("abort_ticks", ctypes.c_int32),
        ("pressure_slow", ctypes.c_double),
        ("pressure_abort", ctypes.c_double),
        ("inside_y", ctypes.c_double),
        ("extrude_steps", ctypes.c_int32),
        ("extrude", ctypes.c_double),
        ("sag_m", ctypes.c_double),
        ("sag_steps", ctypes.c_int32),
        ("follow", ctypes.c_double),
        ("width", ctypes.c_int32),
        ("height", ctypes.c_int32),
        ("f", ctypes.c_double),
        ("cx", ctypes.c_double),
        ("cy", ctypes.c_double),
        ("cam", ctypes.c_double * 3),
    ]


def build() -> Path:
    """Compile the shared library with g++ if it is missing (CMake does the same in CI)."""
    if LIB.exists() and LIB.stat().st_mtime >= (CPP / "monitor.cpp").stat().st_mtime:
        return LIB
    LIB.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["g++", "-std=c++17", "-O2", "-ffp-contract=off", "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror", str(CPP / "monitor.cpp"), "-o", str(LIB)],
        check=True,
    )
    return LIB


_lib: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(str(build()))
        _lib.pm_create.restype = ctypes.c_void_p
        _lib.pm_create.argtypes = [ctypes.POINTER(_Cfg), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        _lib.pm_step.restype = ctypes.c_int32
        _lib.pm_step.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double, ctypes.POINTER(ctypes.c_int32)]
        _lib.pm_signal.restype = ctypes.c_int32
        _lib.pm_signal.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        _lib.pm_destroy.argtypes = [ctypes.c_void_p]
    return _lib


class NativeMonitor:
    def __init__(self, cfg: MonitorConfig, cam: Camera, ref: np.ndarray, mask: np.ndarray, tool0: np.ndarray, extrude: float = 0.0):
        self.cfg = cfg
        c = _Cfg(
            cfg.move_m,
            cfg.dilate_px,
            cfg.warn_px,
            cfg.abort_px,
            cfg.abort_ticks,
            cfg.pressure_slow,
            cfg.pressure_abort,
            cfg.inside_y,
            cfg.extrude_steps,
            float(extrude),
            cfg.sag_m,
            cfg.sag_steps,
            cfg.follow,
            cam.width,
            cam.height,
            cam.f,
            cam.cx,
            cam.cy,
            (ctypes.c_double * 3)(*cam.position),
        )
        self._ref = np.ascontiguousarray(ref, dtype=np.float32)
        self._mask = np.ascontiguousarray(mask, dtype=np.uint8)
        t0 = np.ascontiguousarray(tool0, dtype=np.float64)
        self._h = lib().pm_create(ctypes.byref(c), self._ref.ctypes.data, self._mask.ctypes.data, t0.ctypes.data)

    def signal(self, frame: np.ndarray, tool: np.ndarray) -> int:
        f = np.ascontiguousarray(frame, dtype=np.float32)
        t = np.ascontiguousarray(tool, dtype=np.float64)
        return int(lib().pm_signal(self._h, f.ctypes.data, t.ctypes.data))

    def step(self, frame: np.ndarray, tool: np.ndarray, pressure: float) -> tuple[int, int]:
        f = np.ascontiguousarray(frame, dtype=np.float32)
        t = np.ascontiguousarray(tool, dtype=np.float64)
        s = ctypes.c_int32(0)
        a = lib().pm_step(self._h, f.ctypes.data, t.ctypes.data, float(pressure), ctypes.byref(s))
        return int(a), int(s.value)

    def __del__(self) -> None:
        if getattr(self, "_h", None):
            lib().pm_destroy(self._h)
            self._h = None
