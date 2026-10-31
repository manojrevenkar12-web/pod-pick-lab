"""The monitor: its behaviour on a synthetic bin, and the C++ implementation against it."""

import shutil

import numpy as np
import pytest

from podpick.control.monitor import ABORT, CONTINUE, SLOW, Monitor, MonitorConfig
from podpick.geometry import Camera, monitor_camera

CFG = MonitorConfig(
    move_m=0.012,
    dilate_px=2,
    warn_px=6,
    abort_px=14,
    abort_ticks=2,
    pressure_slow=0.3,
    pressure_abort=0.16,
    inside_y=-0.02,
    extrude_steps=0,
    sag_m=0.0,
    sag_steps=0,
    follow=0.0,
)
CAM = Camera(64, 48, (0.0, -0.42, 0.12), 46.0)


def synthetic():
    ref = np.full((48, 64), 0.70, np.float32)
    mask = np.zeros((48, 64), bool)
    ref[20:30, 26:38] = 0.50
    mask[20:30, 26:38] = True
    ref[12:20, 26:38] = 0.52  # an item resting on the target
    return ref, mask


def test_quiet_bin_continues():
    ref, mask = synthetic()
    m = Monitor(CFG, CAM, ref, mask, np.array([0.0, 0.08, 0.12]))
    assert m.step(ref, np.array([0.0, 0.08, 0.12]), 0.8) == (CONTINUE, 0)


def test_the_target_itself_coming_out_is_not_a_signal():
    ref, mask = synthetic()
    m = Monitor(CFG, CAM, ref, mask, np.array([0.0, 0.08, 0.12]))
    f = ref.copy()
    f[20:30, 26:38] = 0.45
    assert m.step(f, np.array([0.0, 0.03, 0.12]), 0.8) == (CONTINUE, 0)


def test_a_neighbour_coming_forward_slows_then_aborts():
    ref, mask = synthetic()
    m = Monitor(CFG, CAM, ref, mask, np.array([0.0, 0.08, 0.12]))
    f = ref.copy()
    f[20:30, 26:38] = 0.45
    f[12:17, 26:38] = 0.49
    a1, s1 = m.step(f, np.array([0.0, 0.03, 0.12]), 0.8)
    assert a1 == SLOW and s1 >= CFG.abort_px
    assert m.step(f, np.array([0.0, 0.03, 0.12]), 0.8)[0] == ABORT


def test_vacuum_loss_aborts_only_inside_the_bin():
    ref, mask = synthetic()
    m = Monitor(CFG, CAM, ref, mask, np.array([0.0, 0.08, 0.12]))
    assert m.step(ref, np.array([0.0, 0.08, 0.12]), 0.1)[0] == ABORT
    m = Monitor(CFG, CAM, ref, mask, np.array([0.0, 0.08, 0.12]))
    assert m.step(ref, np.array([0.0, -0.10, 0.12]), 0.1)[0] == SLOW


def test_extrusion_covers_the_hidden_length_of_the_item():
    cfg = MonitorConfig(**{**vars(CFG), "extrude_steps": 4})
    ref, mask = synthetic()
    short = Monitor(CFG, CAM, ref, mask, np.zeros(3))
    long = Monitor(cfg, CAM, ref, mask, np.zeros(3), extrude=0.12)
    assert len(long.points) == 5 * len(short.points)
    assert long.points[:, 1].max() == pytest.approx(short.points[:, 1].max() + 0.12)


@pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++")
def test_native_matches_reference_bit_for_bit():
    from podpick.control.native import NativeMonitor

    rng = np.random.default_rng(0)
    cam = monitor_camera()
    cfg = MonitorConfig.from_config()
    for trial in range(30):
        ref = rng.uniform(0.3, 0.8, (cam.height, cam.width)).astype(np.float32)
        ref[rng.random(ref.shape) < 0.05] = 0
        mask = np.zeros(ref.shape, bool)
        v0, u0 = rng.integers(5, 30), rng.integers(5, 45)
        mask[v0 : v0 + rng.integers(4, 15), u0 : u0 + rng.integers(4, 15)] = True
        tool0 = np.array([rng.uniform(-0.1, 0.1), rng.uniform(0, 0.2), rng.uniform(0.02, 0.18)])
        ext = float(rng.uniform(0, 0.15))
        py = Monitor(cfg, cam, ref, mask, tool0, extrude=ext)
        cc = NativeMonitor(cfg, cam, ref, mask, tool0, extrude=ext)
        tool = tool0.copy()
        for _ in range(12):
            tool = tool + np.array([rng.normal(0, 0.002), -rng.uniform(0, 0.05), rng.normal(0, 0.003)])
            frame = (ref + rng.normal(0, 0.01, ref.shape) - (rng.random(ref.shape) < 0.05) * 0.05).astype(np.float32)
            frame[rng.random(ref.shape) < 0.03] = 0
            p = float(rng.uniform(0.05, 0.9))
            assert py.step(frame, tool, p) == cc.step(frame, tool, p), f"trial {trial}"
