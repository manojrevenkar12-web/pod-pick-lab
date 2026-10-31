"""Procedural fabric-pod bins: a scene is a list of items dropped into one bin, then settled.

A `SceneSpec` is the truth about a bin (sizes, materials, masses, seal quality). It is built from
a family in config/families.yaml (or a held-out file) and a seed, so every scene is reproducible.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .. import config


@dataclass
class ItemSpec:
    name: str
    material: str  # box | bag | book
    half: tuple[float, float, float]
    pos: tuple[float, float, float]
    yaw: float
    mass: float
    friction: float
    seal: float  # surface quality for suction, 0..1
    albedo: float
    solref: tuple[float, float]


@dataclass
class SceneSpec:
    family: str
    seed: int
    band_z: float
    items: list[ItemSpec] = field(default_factory=list)
    target: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _world() -> dict[str, Any]:
    return config.load("world")


def family(name: str) -> dict[str, Any]:
    fams = config.load("families")["families"]
    if name in fams:
        return fams[name]
    from pathlib import Path

    import yaml

    path = config.ROOT / "heldout" / f"{name}.yaml"
    if Path(path).exists():
        return yaml.safe_load(path.read_text())["family"]
    raise KeyError(f"unknown scene family {name!r}")


def sample_scene(fam_name: str, seed: int) -> SceneSpec:
    fam = family(fam_name)
    w = _world()
    b = w["bin"]
    rng = np.random.default_rng(seed)
    n = int(rng.integers(fam["items"][0], fam["items"][1] + 1))
    mats = list(fam["mix"].keys())
    probs = np.array([fam["mix"][m] for m in mats], dtype=float)
    probs /= probs.sum()
    half_w, depth, height = b["width"] / 2, b["depth"], b["height"]
    placed: list[ItemSpec] = []
    boxes: list[tuple[float, float, float, float, float]] = []  # x0 x1 y0 y1 ztop
    for i in range(n):
        mat = str(rng.choice(mats, p=probs))
        size = fam[f"{mat}_size"]
        full = [float(rng.uniform(*size[k])) for k in ("x", "y", "z")]
        if mat == "book" and rng.random() < 0.4:
            # some books lie flat
            full = [full[1] * 0.8, full[2] * 0.8, full[0]]
        hx, hy, hz = (v / 2 for v in full)
        yaw = math.radians(float(rng.uniform(-fam["yaw_deg"], fam["yaw_deg"])))
        c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
        ex, ey = hx * c + hy * s, hx * s + hy * c
        for _ in range(30):
            x = float(rng.uniform(-half_w + ex + 0.002, half_w - ex - 0.002)) if half_w > ex + 0.002 else 0.0
            u = float(rng.random()) ** (1.0 + 2.0 * fam["front_bias"])
            y = ey + 0.004 + (depth - 2 * ey - 0.008) * u
            base = 0.0
            for x0, x1, y0, y1, zt in boxes:
                if x - ex < x1 and x + ex > x0 and y - ey < y1 and y + ey > y0:
                    base = max(base, zt)
            z = base + hz + fam["drop_gap"]
            if z + hz < height - 0.012:
                break
        else:
            continue
        lo, hi = w["materials"][mat]["density"]
        mass = float(rng.uniform(lo, hi)) * 8 * hx * hy * hz
        mass = float(np.clip(mass, 0.04, 2.0))
        it = ItemSpec(
            name=f"item{i}",
            material=mat,
            half=(hx, hy, hz),
            pos=(x, y, z),
            yaw=yaw,
            mass=mass,
            friction=float(rng.uniform(*w["materials"][mat]["friction"])),
            seal=float(rng.uniform(*w["materials"][mat]["seal"])),
            albedo=float(rng.uniform(0.15, 0.95)),
            solref=tuple(w["materials"][mat]["solref"]),
        )
        placed.append(it)
        boxes.append((x - ex, x + ex, y - ey, y + ey, z + hz))
    band_z = float(rng.uniform(0.09, 0.15))
    return SceneSpec(family=fam_name, seed=seed, band_z=band_z, items=placed)


def mjcf(spec: SceneSpec) -> str:
    """The MJCF model of one bin, its band, its items and the suction tool."""
    w = _world()
    rb = config.robot()
    b, band, tool = w["bin"], w["band"], rb["tool"]
    hw, d, h, t = b["width"] / 2, b["depth"], b["height"], b["wall"]
    wsr = " ".join(str(v) for v in b["wall_solref"])
    fr = b["wall_friction"]
    walls = [
        ("floor", (0, d / 2, -t / 2), (hw + t, d / 2 + t, t / 2)),
        ("ceiling", (0, d / 2, h + t / 2), (hw + t, d / 2 + t, t / 2)),
        ("back", (0, d + t / 2, h / 2), (hw + t, t / 2, h / 2 + t)),
        ("left", (-hw - t / 2, d / 2, h / 2), (t / 2, d / 2 + t, h / 2 + t)),
        ("right", (hw + t / 2, d / 2, h / 2), (t / 2, d / 2 + t, h / 2 + t)),
        ("lip", (0, -0.004, b["lip_height"] / 2), (hw, 0.004, b["lip_height"] / 2)),
    ]
    wall_xml = "\n".join(
        f'<geom name="{n}" type="box" pos="{p[0]} {p[1]} {p[2]}" size="{s[0]} {s[1]} {s[2]}" '
        f'friction="{fr} 0.005 0.0001" solref="{wsr}" group="1" contype="3" conaffinity="3"/>'
        for n, p, s in walls
    )
    items_xml = []
    for it in spec.items:
        cz, sz = math.cos(it.yaw / 2), math.sin(it.yaw / 2)
        items_xml.append(
            f'<body name="{it.name}" pos="{it.pos[0]} {it.pos[1]} {it.pos[2]}" quat="{cz} 0 0 {sz}">'
            f'<freejoint/><geom name="{it.name}" type="box" size="{it.half[0]} {it.half[1]} {it.half[2]}" '
            f'mass="{it.mass}" friction="{it.friction} 0.005 0.0001" solref="{it.solref[0]} {it.solref[1]}" '
            f'condim="4" group="0"/></body>'
        )
    cl = tool["cup_length"]
    return f"""<mujoco model="pod_bin">
<option timestep="{w["timestep"]}" integrator="implicitfast" cone="elliptic" iterations="40"/>
<default><geom solimp="0.9 0.95 0.001" margin="0.0005"/></default>
<worldbody>
{wall_xml}
<geom name="station_floor" type="plane" pos="0 0 -0.6" size="2 2 0.1" group="1"/>
<body name="band" pos="0 -0.012 {spec.band_z}">
  <joint name="band_y" type="slide" axis="0 1 0" stiffness="{band["stiffness"]}" damping="{band["damping"]}" range="-0.08 0.01" limited="true"/>
  <joint name="band_z" type="slide" axis="0 0 1" damping="20" range="{band["open_z"] - spec.band_z} 0.0" limited="true"/>
  <geom name="band" type="capsule" fromto="{-hw - t} 0 0 {hw + t} 0 0" size="{band["radius"]}" mass="0.02" group="2" contype="3" conaffinity="3"/>
</body>
<body name="tool" mocap="true" pos="0 -0.5 0.1">
  <geom name="cup" type="cylinder" pos="0 0 {-cl / 2}" size="{tool["cup_radius"]} {cl / 2}" group="3" contype="1" conaffinity="1"/>
  <geom name="link" type="capsule" fromto="0 0 {-cl} 0 0 {-cl - tool["link_length"]}" size="{tool["link_radius"]}" group="3" contype="2" conaffinity="2"/>
</body>
{chr(10).join(items_xml)}
</worldbody>
<equality>
{chr(10).join(f'<weld name="suction_{it.name}" body1="tool" body2="{it.name}" active="false" solref="{w["suction"]["weld_solref"][0]} {w["suction"]["weld_solref"][1]}" torquescale="{w["suction"]["torquescale"]}"/>' for it in spec.items)}
</equality>
<actuator><position name="band_hold" joint="band_z" kp="400" ctrlrange="{band["open_z"] - spec.band_z} 0"/></actuator>
</mujoco>"""
