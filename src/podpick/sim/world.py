"""A simulated bin with a suction tool: physics in MuJoCo, sensors by ray casting.

`World` implements the `Driver` the pick cell uses, and in addition exposes the truth
(item poses, labels, outcomes) to the evaluation. The cell gets a `World` typed as a `Driver`
and never calls the truth methods; `tests/test_isolation.py` runs the cell against a driver
whose truth methods raise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from .. import config
from ..geometry import Camera, monitor_camera, quat_from_z, scene_camera
from ..interface import Observation, TargetQuery, Tick
from .scene import SceneSpec, mjcf

GROUP_ITEMS, GROUP_BIN, GROUP_BAND, GROUP_TOOL = 0, 1, 2, 3


def _quat_mat(q: np.ndarray) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


@dataclass
class Seal:
    eq: int  # the weld constraint (tool <-> item) that models the cup
    item: int  # index into spec.items
    quality: float
    over: float = 0.0  # seconds over capacity


class World:
    def __init__(self, spec: SceneSpec, *, settle: float = 1.0, noise_seed: int | None = None):
        self.spec = spec
        self.w = config.load("world")
        self.r = config.robot()
        self.model = mujoco.MjModel.from_xml_string(mjcf(spec))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(spec.seed * 7919 + 13 if noise_seed is None else noise_seed)
        m = self.model
        self.item_body = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, it.name) for it in spec.items]
        self.item_geom = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, it.name) for it in spec.items]
        self.geom_item = {g: i for i, g in enumerate(self.item_geom)}
        self.item_eq = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"suction_{it.name}") for it in spec.items]
        self.cup = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cup")
        self.tool_mocap = int(m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tool")])
        self.band_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "band")
        self.albedo = np.full(m.ngeom, 0.33)
        for i, g in enumerate(self.item_geom):
            self.albedo[g] = spec.items[i].albedo
        self.albedo[self.band_geom] = 0.85
        self.seal: Seal | None = None
        self._tool_vel = np.zeros(3)
        self._speed = 0.0
        self.load = 0.0
        self.pressure = 0.03
        self.t = 0.0
        self.scene_cam, self.mon_cam = scene_camera(), monitor_camera()
        self._rays = {id(c): c.rays() for c in (self.scene_cam, self.mon_cam)}
        self._band_mask = self.rng.random(self.scene_cam.width * self.scene_cam.height) < self.w["band"]["pass_through"]
        self._band_mask_mon = self.rng.random(self.mon_cam.width * self.mon_cam.height) < self.w["band"]["pass_through"]
        self.data.mocap_pos[self.tool_mocap] = [0.0, -0.6, 0.1]
        self.data.mocap_quat[self.tool_mocap] = quat_from_z(np.array([0.0, 1.0, 0.0]))
        mujoco.mj_forward(m, self.data)
        self._steps(settle)
        self._drop_escaped()
        self.start_pose = self.item_poses()
        self.t = 0.0

    # ------------------------------------------------------------------ physics
    def _steps(self, seconds: float) -> None:
        for _ in range(int(round(seconds / self.model.opt.timestep))):
            self._step()

    def _step(self) -> None:
        m, d = self.model, self.data
        mujoco.mj_step(m, d)
        self.t += m.opt.timestep
        if self.seal is None:
            return
        s = self.seal
        sc = self.w["suction"]
        rows = np.nonzero((d.efc_type[: d.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (d.efc_id[: d.nefc] == s.eq))[0]
        if len(rows) < 6:
            return
        f = d.efc_force[rows]
        cap = sc["max_force"] * s.quality
        # rotational rows are scaled by torquescale: the moment on the item is f * torquescale
        torque = float(np.linalg.norm(f[3:6])) * sc["torquescale"]
        load = max(float(np.linalg.norm(f[:3])) / cap, torque / (cap * sc["peel_lever"]))
        s.over = s.over + m.opt.timestep if load > 1.0 else max(0.0, s.over - m.opt.timestep)
        self.load = min(1.0, load)
        self.pressure = s.quality * (1.0 - 0.6 * self.load**2)
        if s.over > sc["peel_time"]:
            d.eq_active[s.eq] = 0
            self.seal = None
            self.pressure = 0.03

    def _drop_escaped(self) -> None:
        """Items that fell out while the bin was filled are not part of the scene."""
        b = self.w["bin"]
        keep = []
        for body in self.item_body:
            p = self.data.xpos[body]
            inside = -0.01 < p[1] < b["depth"] + 0.01 and -0.02 < p[2] < b["height"] and abs(p[0]) < b["width"] / 2 + 0.01
            keep.append(inside)
        self.present = np.array(keep, dtype=bool)

    def item_poses(self) -> np.ndarray:
        return np.array([np.concatenate([self.data.xpos[b], self.data.xquat[b]]) for b in self.item_body])

    # ------------------------------------------------------------------ sensors
    def _cast(self, cam: Camera, groups: tuple[int, ...], band_mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        m, d = self.model, self.data
        rays = self._rays[id(cam)]
        n = len(rays)

        def cast(gs: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            gg = np.zeros(6, dtype=np.uint8)
            gg[list(gs)] = 1
            gid = np.full(n, -1, dtype=np.int32)
            dist = np.full(n, -1.0)
            nrm = np.zeros(3 * n)
            mujoco.mj_multiRay(
                m,
                d,
                np.asarray(cam.position, dtype=np.float64),
                rays.reshape(-1).copy(),
                gg,
                True,
                -1,
                gid,
                dist,
                nrm,
                n,
                self.r["camera"]["max_range"] / rays[:, 1].min(),
            )
            return gid, dist, nrm.reshape(n, 3)

        gid, dist, nrm = cast(groups)
        if band_mask is not None and GROUP_BAND in groups:
            g2, d2, n2 = cast(tuple(g for g in groups if g != GROUP_BAND))
            through = band_mask & (gid == self.band_geom)
            gid[through], dist[through], nrm[through] = g2[through], d2[through], n2[through]
        depth = np.where(dist > 0, dist * rays[:, 1], 0.0)
        depth[depth > self.r["camera"]["max_range"]] = 0.0
        return gid, depth, nrm

    def _noisy(self, cam: Camera, gid: np.ndarray, depth: np.ndarray, nrm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rays = self._rays[id(cam)]
        dn = self.w["depth_noise"]
        noisy = depth + self.rng.normal(0, 1, depth.shape) * dn["sigma_at_1m"] * depth**2
        noisy[self.rng.random(depth.shape) < dn["dropout"]] = 0.0
        noisy[depth == 0] = 0.0
        shade = 0.35 + 0.65 * np.abs(np.sum(nrm * rays, axis=1))
        alb = np.where(gid >= 0, self.albedo[np.maximum(gid, 0)], 0.0)
        inten = np.clip(alb * shade + self.rng.normal(0, self.w["intensity_noise"], depth.shape), 0, 1)
        H, W = cam.height, cam.width
        return noisy.reshape(H, W).astype(np.float32), inten.reshape(H, W).astype(np.float32)

    # ------------------------------------------------------------------ Driver
    def observe(self) -> Observation:
        gid, depth, nrm = self._cast(self.scene_cam, (GROUP_ITEMS, GROUP_BIN, GROUP_BAND), self._band_mask)
        dep, inten = self._noisy(self.scene_cam, gid, depth, nrm)
        return Observation(depth=dep, intensity=inten, query=self.query(self.target_index), band_z=self.spec.band_z)

    def query(self, idx: int) -> TargetQuery:
        """The order for item `idx`: catalogue data, which is close to but not exactly the item."""
        tgt = self.spec.items[idx]
        dims = tuple(sorted((2 * h for h in tgt.half), reverse=True))
        rq = self.rng
        return TargetQuery(
            dims=tuple(float(v * (1 + rq.normal(0, 0.03))) for v in dims),
            albedo=float(np.clip(tgt.albedo + rq.normal(0, 0.04), 0, 1)),
            mass=float(tgt.mass * (1 + rq.normal(0, 0.05))),
        )

    def frame(self) -> np.ndarray:
        """A monitor frame, self-filtered: the robot's own geometry is excluded from the rays."""
        gid, depth, nrm = self._cast(self.mon_cam, (GROUP_ITEMS, GROUP_BIN, GROUP_BAND), self._band_mask_mon)
        dep, _ = self._noisy(self.mon_cam, gid, depth, nrm)
        return dep

    def open_band(self) -> None:
        self.data.ctrl[0] = self.w["band"]["open_z"] - self.spec.band_z
        self._steps(0.3)

    def close_band(self) -> None:
        self.data.ctrl[0] = 0.0
        self._steps(0.2)

    def place_tool(self, pos: np.ndarray, direction: np.ndarray) -> None:
        self.data.mocap_pos[self.tool_mocap] = pos
        self.data.mocap_quat[self.tool_mocap] = quat_from_z(direction)
        self._tool_vel = np.zeros(3)
        self._speed = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _cup_contact(self) -> int | None:
        d = self.data
        for k in range(d.ncon):
            c = d.contact[k]
            if c.geom1 == self.cup or c.geom2 == self.cup:
                other = c.geom2 if c.geom1 == self.cup else c.geom1
                return int(other)
        return None

    def move(self, target: np.ndarray, speed: float, *, stop_on_contact: bool = False, frame: bool = False) -> Tick:
        dt = self.model.opt.timestep
        tick = 1.0 / self.r["camera"]["monitor_hz"]
        pos = self.data.mocap_pos[self.tool_mocap].copy()
        target = np.asarray(target, dtype=np.float64)
        lowest = 1.0 if self.seal is not None else self.pressure
        contact = False
        reached = False
        accel = self.r["motion"]["accel"]
        for _ in range(int(round(tick / dt))):
            delta = target - pos
            dist = float(np.linalg.norm(delta))
            if dist < 1e-6 or (stop_on_contact and contact):
                self._tool_vel = np.zeros(3)
                self._speed = 0.0
                reached = dist < 1e-6
            else:
                # accelerate toward the commanded speed, and brake in time to stop at the target
                v = min(speed, self._speed + accel * dt, math.sqrt(2 * accel * dist) + accel * dt)
                self._speed = v
                step = min(dist, v * dt)
                pos = pos + delta / dist * step
                self._tool_vel = delta / dist * (step / dt)
            self.data.mocap_pos[self.tool_mocap] = pos
            self._step()
            if stop_on_contact and not contact and self._cup_contact() is not None:
                contact = True
            lowest = min(lowest, self.pressure if self.seal is not None else self.pressure)
        self._tool_vel = np.zeros(3)
        noise = float(self.rng.normal(0, self.w["suction"]["pressure_noise"]))
        return Tick(
            tool=pos.copy(),
            contact=contact or self._cup_contact() is not None,
            pressure=float(np.clip(lowest + noise, 0, 1)),
            frame=self.frame() if frame else None,
            reached=reached or float(np.linalg.norm(target - pos)) < 1e-6,
        )

    def vacuum(self, on: bool) -> float:
        if not on:
            if self.seal is not None:
                self.data.eq_active[self.seal.eq] = 0
            self.seal = None
            self.pressure = 0.03
            self._steps(0.05)
            return self.pressure
        g = self._cup_contact()
        sc = self.w["suction"]
        if g is None or g not in self.geom_item:
            self.pressure = 0.03
            self._steps(0.1)
            return float(np.clip(self.pressure + self.rng.normal(0, sc["pressure_noise"]), 0, 1))
        i = self.geom_item[g]
        body = self.item_body[i]
        it = self.spec.items[i]
        R = self.data.xmat[body].reshape(3, 3)
        tip = self.data.mocap_pos[self.tool_mocap].copy()
        approach = _quat_mat(self.data.mocap_quat[self.tool_mocap])[:, 2]
        local = R.T @ (tip - self.data.xpos[body])
        half = np.array(it.half)
        ax = int(np.argmax(np.abs(local) / half))
        normal_local = np.zeros(3)
        normal_local[ax] = math.copysign(1.0, local[ax])
        n_world = R @ normal_local
        angle = math.degrees(math.acos(float(np.clip(-approach @ n_world, -1, 1))))
        q_angle = max(0.0, 1.0 - (angle / sc["max_angle_deg"]) ** 2)
        others = [k for k in range(3) if k != ax]
        edge = min(half[k] - abs(local[k]) for k in others)
        r_cup = self.r["tool"]["cup_radius"]
        q_edge = float(np.clip((edge - sc["edge_margin"]) / r_cup, 0.0, 1.0)) ** 0.5
        # small faces: the cup cannot seal on a face narrower than itself
        width = min(2 * half[k] for k in others)
        q_size = float(np.clip((width - 0.6 * 2 * r_cup) / (0.8 * r_cup), 0.0, 1.0))
        q = it.seal * q_angle * q_edge * q_size
        if q < sc["min_seal"]:
            self.pressure = q * 0.5
            self._steps(0.1)
            return float(np.clip(self.pressure + self.rng.normal(0, sc["pressure_noise"]), 0, 1))
        point_local = local.copy()
        point_local[ax] = normal_local[ax] * half[ax]
        attach = self.data.xpos[body] + R @ point_local
        # a compliant weld between the tool and the item at the cup: the bellows
        eq = self.item_eq[i]
        qt = self.data.mocap_quat[self.tool_mocap]
        Rt = _quat_mat(qt)
        rel = np.zeros(4)
        mujoco.mju_mulQuat(rel, np.array([qt[0], -qt[1], -qt[2], -qt[3]]), self.data.xquat[body])
        self.model.eq_data[eq][:3] = point_local
        self.model.eq_data[eq][3:6] = Rt.T @ (attach - tip)
        self.model.eq_data[eq][6:10] = rel
        self.data.eq_active[eq] = 1
        self.seal = Seal(eq=eq, item=i, quality=q)
        self.pressure = q
        self._steps(0.1)
        return float(np.clip(self.pressure + self.rng.normal(0, sc["pressure_noise"]), 0, 1))

    @property
    def clock(self) -> float:
        return self.t

    # ------------------------------------------------------------------ counterfactuals
    def snapshot(self) -> dict[str, Any]:
        """Everything needed to replay the bin from this instant (physics, sensors' noise, tool)."""
        sp = mujoco.mjtState.mjSTATE_INTEGRATION
        mujoco.mj_forward(self.model, self.data)  # derived quantities as restore() will see them
        st = np.zeros(mujoco.mj_stateSize(self.model, sp))
        mujoco.mj_getState(self.model, self.data, st, sp)
        return {"state": st, "t": self.t, "rng": self.rng.bit_generator.state, "pressure": self.pressure, "seal": self.seal}

    def restore(self, snap: dict[str, Any]) -> None:
        sp = mujoco.mjtState.mjSTATE_INTEGRATION
        mujoco.mj_setState(self.model, self.data, snap["state"], sp)
        mujoco.mj_forward(self.model, self.data)
        self.t = snap["t"]
        self.rng.bit_generator.state = snap["rng"]
        self.pressure = snap["pressure"]
        self.seal = snap["seal"]
        self._tool_vel = np.zeros(3)
        self._speed = 0.0

    # ------------------------------------------------------------------ truth (evaluation only)
    target_index: int = 0

    def set_target(self, index: int) -> None:
        self.target_index = index

    def held_item(self) -> int | None:
        return None if self.seal is None else self.seal.item

    def settle(self, seconds: float) -> None:
        self._steps(seconds)

    def outcome(self) -> dict[str, Any]:
        """What happened, from the truth: target out, neighbours out, displaced, dropped."""
        o = self.w["outcome"]
        poses = self.item_poses()
        out = [bool(p[1] < o["out_y"] or p[2] < -0.05) for p in poses]
        moved = np.linalg.norm(poses[:, :3] - self.start_pose[:, :3], axis=1)
        held = self.held_item()
        tgt = self.target_index
        neighbours_out = [i for i in range(len(poses)) if i != tgt and out[i] and self.present[i] and i != held]
        return {
            "target_out": bool(out[tgt] and held == tgt),
            "target_dropped_out": bool(out[tgt] and held != tgt),
            "wrong_item": held is not None and held != tgt and out[held],
            "collateral": len(neighbours_out),
            "disturbed": int(sum(1 for i in range(len(poses)) if i != tgt and self.present[i] and moved[i] > o["disturbed_m"])),
            "held": held,
        }

    def labels(self) -> dict[str, Any]:
        """Training labels for perception: item id per pixel and the target's occlusion class."""
        gid, depth, _ = self._cast(self.scene_cam, (GROUP_ITEMS, GROUP_BIN, GROUP_BAND), self._band_mask)
        H, W = self.scene_cam.height, self.scene_cam.width
        item_px = np.full(len(gid), -1, dtype=np.int32)
        for g, i in self.geom_item.items():
            item_px[gid == g] = i
        seg = item_px.reshape(H, W)
        return {"seg": seg, "occlusion": self.occlusion_class(self.target_index, seg)}

    def occlusion_class(self, idx: int, seg: np.ndarray | None = None) -> int:
        """0 unobstructed, 1 below others (something rests on it), 2 blocked (mostly hidden in front).

        Mirrors the classes the Vulcan Pick write-up describes for its segmentation model.
        """
        d = self.data
        body = self.item_body[idx]
        for k in range(d.ncon):
            c = d.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            if self.item_geom[idx] not in (g1, g2):
                continue
            other = g2 if g1 == self.item_geom[idx] else g1
            if other not in self.geom_item:
                continue
            j = self.geom_item[other]
            # contact normal points from geom1 to geom2; is the other item on top of this one?
            n = np.array(c.frame[:3])
            if other == g2:
                up = n[2]
            else:
                up = -n[2]
            if up > 0.6 and d.xpos[self.item_body[j]][2] > d.xpos[body][2]:
                return 1
        if seg is None:
            seg = self.labels()["seg"]
        vis = float((seg == idx).sum())
        full = self._unoccluded_pixels(idx)
        if full > 0 and vis / full < 0.5:
            return 2
        return 0

    def _unoccluded_pixels(self, idx: int) -> float:
        cam = self.scene_cam
        rays = self._rays[id(cam)]
        n = len(rays)
        gid = np.full(n, -1, dtype=np.int32)
        dist = np.full(n, -1.0)
        gg = np.zeros(6, dtype=np.uint8)
        gg[GROUP_ITEMS] = 1
        # only this item: exclude all other bodies by casting against the item group and
        # keeping hits that land on the item even when something else is in front
        m, d = self.model, self.data
        saved = m.geom_group.copy()
        for g in self.item_geom:
            if g != self.item_geom[idx]:
                m.geom_group[g] = 5
        mujoco.mj_multiRay(m, d, np.asarray(cam.position, dtype=np.float64), rays.reshape(-1).copy(), gg, False, -1, gid, dist, None, n, 5.0)
        m.geom_group[:] = saved
        return float((gid == self.item_geom[idx]).sum())

    def visible_items(self, min_pixels: int = 30) -> list[int]:
        seg = self.labels()["seg"]
        return [i for i in range(len(self.item_body)) if self.present[i] and int((seg == i).sum()) >= min_pixels]
