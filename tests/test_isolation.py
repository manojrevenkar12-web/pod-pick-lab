"""The pick cell decides from what its sensors report. It never imports the simulator or the
evaluation, and it runs against a driver that refuses every question about the truth."""

import ast
from pathlib import Path

import numpy as np
import pytest

from podpick.cell import PickCell
from podpick.sim.scene import sample_scene
from podpick.sim.world import World

SRC = Path(__file__).resolve().parents[1] / "src" / "podpick"
SHIPPING = ["cell.py", "geometry.py", "interface.py", "perception", "grasp", "control", "decide", "service"]


def _imports(path: Path) -> set[str]:
    out = set()
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            out.add("." * node.level + mod)
        elif isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
    return out


def _files() -> list[Path]:
    files = []
    for s in SHIPPING:
        p = SRC / s
        files += [p] if p.is_file() else sorted(p.rglob("*.py"))
    return files


@pytest.mark.parametrize("path", _files(), ids=lambda p: str(p.relative_to(SRC)))
def test_shipping_code_never_imports_sim_or_evaluation(path):
    for mod in _imports(path):
        assert "sim" not in mod.split(".") and "evaluate" not in mod.split("."), f"{path.name} imports {mod}"
        assert not mod.startswith("podpick.sim") and not mod.startswith("podpick.evaluate")


def test_config_world_is_only_read_by_sim_and_evaluation():
    for path in _files():
        assert '"world"' not in path.read_text() and "'world'" not in path.read_text(), path


class TruthlessDriver:
    """Only the Driver protocol; any other attribute is an error."""

    ALLOWED = {"observe", "open_band", "close_band", "place_tool", "move", "vacuum", "frame", "clock"}

    def __init__(self, world):
        object.__setattr__(self, "_w", world)

    def __getattr__(self, name):
        if name not in TruthlessDriver.ALLOWED:
            raise AssertionError(f"the cell asked the driver for {name!r}")
        return getattr(object.__getattribute__(self, "_w"), name)


@pytest.mark.parametrize("policy", ["heuristic-open", "heuristic-closed"])
def test_cell_runs_on_sensors_only(policy):
    w = World(sample_scene("mixed", 1000017))
    w.set_target(5)
    seg = w.labels()["seg"]
    occ = np.array([1.0, 0.0, 0.0])
    mask = seg == 5

    cell = PickCell(lambda obs: {"mask": mask, "occlusion": occ}, None, policy)
    r = cell.pick(TruthlessDriver(w))
    assert r.status in {"picked", "lost", "aborted", "handoff"}
    assert r.attempts, "an unobstructed box should be attempted"
