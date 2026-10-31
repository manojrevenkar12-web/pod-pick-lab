"""Suction candidates on the target's visible surface, and what is around each of them.

Everything here is computed from the depth image, the target mask and the known camera and
tool geometry. The features describe the free space the item needs on its way out: what is in
front of the candidate (the extraction corridor), what touches the target on each side (an item
resting on top is the usual way a neighbour comes out with it), and how good a seal looks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import config
from ..geometry import Camera, depth_normals, scene_camera
from ..interface import Observation

FEATURES = (
    "angle_deg",
    "normal_z",
    "planarity_mm",
    "edge_m",
    "face_w_m",
    "face_h_m",
    "clearance_m",
    "front_frac",
    "top_touch",
    "top_front",
    "left_touch",
    "right_touch",
    "bottom_touch",
    "height_m",
    "depth_m",
    "mass_kg",
    "dim0",
    "dim1",
    "dim2",
    "occ0",
    "occ1",
    "occ2",
)


@dataclass
class Candidates:
    pixels: np.ndarray  # (K, 2) v, u
    points: np.ndarray  # (K, 3) world
    normals: np.ndarray  # (K, 3) world, toward the camera
    features: np.ndarray  # (K, len(FEATURES)) float32
    crops: np.ndarray  # (K, 3, C, C) float32: relative depth, target mask, valid
    clearance: np.ndarray  # (K,) signed distance to the nearest neighbour in the corridor

    def __len__(self) -> int:
        return len(self.points)


def _erode(mask: np.ndarray, r: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(r):
        m = out.copy()
        m[1:, :] &= out[:-1, :]
        m[:-1, :] &= out[1:, :]
        m[:, 1:] &= out[:, :-1]
        m[:, :-1] &= out[:, 1:]
        out = m
    return out


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(r):
        m = out.copy()
        m[1:, :] |= out[:-1, :]
        m[:-1, :] |= out[1:, :]
        m[:, 1:] |= out[:, :-1]
        m[:, :-1] |= out[:, 1:]
        out = m
    return out


def _fps(pts: np.ndarray, k: int) -> np.ndarray:
    """Farthest-point sampling, deterministic (starts at the point nearest the centroid)."""
    if len(pts) <= k:
        return np.arange(len(pts))
    chosen = [int(np.argmin(np.linalg.norm(pts - pts.mean(0), axis=1)))]
    d = np.linalg.norm(pts - pts[chosen[0]], axis=1)
    for _ in range(k - 1):
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(pts - pts[i], axis=1))
    return np.array(chosen)


def generate(obs: Observation, mask: np.ndarray, occlusion: np.ndarray, cam: Camera | None = None) -> Candidates:
    cfg = config.policy()["candidates"]
    tool = config.robot()["tool"]
    cam = cam or scene_camera()
    depth = obs.depth.astype(np.float64)
    valid = depth > 0
    pts = cam.points(depth)
    # the band is opened by the band arm before the pick: its pixels are not obstacles
    band = (pts[..., 1] < 0.0) & (np.abs(pts[..., 2] - obs.band_z) < 0.012)
    valid &= ~band
    normals = depth_normals(pts, valid)
    H, W = depth.shape
    m = mask & valid
    # the cup needs the patch around it: candidates sit at least one pixel inside the mask
    inner = _erode(m, 1) & (np.linalg.norm(normals, axis=-1) > 0)
    vs, us = np.nonzero(inner)
    keep = (vs % cfg["stride"] == 0) & (us % cfg["stride"] == 0)
    vs, us = vs[keep], us[keep]
    empty = Candidates(
        np.zeros((0, 2), int),
        np.zeros((0, 3)),
        np.zeros((0, 3)),
        np.zeros((0, len(FEATURES)), np.float32),
        np.zeros((0, 3, cfg["crop"], cfg["crop"]), np.float32),
        np.zeros(0),
    )
    if len(vs) == 0:
        return empty
    # smoothed normals and planarity over a patch inside the mask
    p = cfg["patch"]
    n_s, plan = [], []
    for v, u in zip(vs, us, strict=True):
        sl = (slice(max(0, v - p), v + p + 1), slice(max(0, u - p), u + p + 1))
        sel = m[sl] & (np.linalg.norm(normals[sl], axis=-1) > 0)
        nn_ = normals[sl][sel].mean(0)
        nn_ /= max(np.linalg.norm(nn_), 1e-9)
        q = pts[sl][sel]
        plan.append(float(np.abs((q - q.mean(0)) @ nn_).mean()) * 1000 if len(q) >= 3 else 10.0)
        n_s.append(nn_)
    n_s = np.array(n_s)
    plan = np.array(plan)
    front = np.array([0.0, -1.0, 0.0])
    ang = np.degrees(np.arccos(np.clip(n_s @ front, -1, 1)))
    ok = ang <= cfg["max_angle_deg"]
    vs, us, n_s, plan, ang = vs[ok], us[ok], n_s[ok], plan[ok], ang[ok]
    if len(vs) == 0:
        return empty
    cand_pts = pts[vs, us]
    sel = _fps(cand_pts, cfg["max"])
    vs, us, n_s, plan, ang, cand_pts = vs[sel], us[sel], n_s[sel], plan[sel], ang[sel], cand_pts[sel]

    # distance to the mask boundary, in metres on the surface
    boundary = m & ~_erode(m, 1)
    bv, bu = np.nonzero(boundary)
    bpts = pts[bv, bu] if len(bv) else cand_pts[:1]
    edge = np.min(np.linalg.norm(cand_pts[:, None, :] - bpts[None, :, :], axis=-1), axis=1)
    mv, mu = np.nonzero(m)
    tpts = pts[mv, mu]
    face_w = float(tpts[:, 0].max() - tpts[:, 0].min())
    face_h = float(tpts[:, 2].max() - tpts[:, 2].min())

    # neighbours: everything valid that is not the target
    other = valid & ~_dilate(m, 1)
    ov, ou = np.nonzero(other)
    opts = pts[ov, ou]
    # the extraction corridor: the tool and the item come straight out toward -y
    rad = tool["cup_radius"] + cfg["corridor_margin"]
    item_r = 0.5 * max(face_w, face_h)
    clearance = np.zeros(len(cand_pts))
    front_frac = np.zeros(len(cand_pts))
    for k, c in enumerate(cand_pts):
        ahead = opts[:, 1] < c[1] - 0.005
        if not ahead.any():
            clearance[k] = 0.1
            continue
        r = np.hypot(opts[ahead, 0] - c[0], opts[ahead, 2] - c[2])
        clearance[k] = min(0.1, float(r.min()) - rad)
        front_frac[k] = float((r < item_r).mean()) if item_r > 0 else 0.0

    # what touches the target on each side, from the ring of pixels around the mask
    ring = _dilate(m, cfg["ring_px"]) & ~m & valid
    tdepth = np.where(m, depth, np.nan)
    sides = {}
    rows = np.nonzero(m.any(1))[0]
    cols = np.nonzero(m.any(0))[0]
    vmin, vmax, umin, umax = rows.min(), rows.max(), cols.min(), cols.max()
    vv, uu = np.mgrid[0:H, 0:W]
    target_d = float(np.nanmedian(tdepth))
    for name, region in (
        ("top", ring & (vv < vmin + 2)),
        ("bottom", ring & (vv > vmax - 2)),
        ("left", ring & (uu < umin + 2)),
        ("right", ring & (uu > umax - 2)),
    ):
        dd = depth[region] - target_d
        touch = float((np.abs(dd) < cfg["touch_m"]).mean()) if dd.size else 0.0
        infront = float((dd < -cfg["touch_m"]).mean()) if dd.size else 0.0
        sides[name] = (touch, infront)

    q = obs.query
    feats = []
    for k in range(len(cand_pts)):
        feats.append(
            [
                ang[k],
                n_s[k][2],
                min(plan[k], 10.0),
                edge[k],
                face_w,
                face_h,
                clearance[k],
                front_frac[k],
                sides["top"][0],
                sides["top"][1],
                sides["left"][0],
                sides["right"][0],
                sides["bottom"][0],
                cand_pts[k][2],
                cand_pts[k][1],
                q.mass,
                *q.dims,
                *occlusion,
            ]
        )
    crops = _crops(depth, m, valid, vs, us, cand_pts[:, 1], cfg["crop"])
    return Candidates(np.stack([vs, us], 1), cand_pts, n_s, np.array(feats, np.float32), crops, clearance)


def _crops(depth: np.ndarray, mask: np.ndarray, valid: np.ndarray, vs: np.ndarray, us: np.ndarray, ref: np.ndarray, c: int) -> np.ndarray:
    H, W = depth.shape
    h = c // 2
    pad = np.pad(depth, h)
    pm = np.pad(mask, h)
    pv = np.pad(valid, h)
    out = np.zeros((len(vs), 3, c, c), np.float32)
    for k, (v, u) in enumerate(zip(vs, us, strict=True)):
        d = pad[v : v + c, u : u + c]
        ok = pv[v : v + c, u : u + c]
        rel = np.where(ok, np.clip((d - ref[k]) / 0.1, -2, 2), 0.0)
        out[k, 0] = rel
        out[k, 1] = pm[v : v + c, u : u + c]
        out[k, 2] = ok
    return out


def heuristic_scores(c: Candidates) -> np.ndarray:
    """Rank by signed clearance in the extraction corridor, then edge distance and angle."""
    h = config.policy()["heuristic"]
    f = {n: c.features[:, i] for i, n in enumerate(FEATURES)}
    return c.clearance + h["w_edge"] * np.minimum(f["edge_m"], 0.02) - h["w_angle"] * f["angle_deg"]
