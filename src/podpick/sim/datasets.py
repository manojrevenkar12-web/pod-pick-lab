"""Labelled data from the simulator: what a fleet's logs and an annotation team would provide.

- perception: bin images with the target's mask and occlusion class;
- picks: logged attempts (candidate, primitive) with their outcome, several counterfactual
  attempts per order from the same start state;
- unlabelled: candidate crops with no outcome, for self-supervised pretraining.

Pick attempts are made by the deployed perception with a uniformly random behaviour policy over
candidates and primitives, open loop, so the outcome labels are not shaped by any scorer.
"""

from __future__ import annotations

import math
from multiprocessing import get_context
from typing import Any

import numpy as np

from .. import config
from .scene import sample_scene
from .world import World


def _chunks(seeds: list[int], n: int) -> list[list[int]]:
    k = max(1, math.ceil(len(seeds) / n))
    return [seeds[i : i + k] for i in range(0, len(seeds), k)]


def _pool_map(fn: Any, jobs: list[Any], workers: int) -> list[Any]:
    if workers <= 1 or len(jobs) <= 1:
        return [fn(j) for j in jobs]
    with get_context("spawn").Pool(workers) as pool:
        return pool.map(fn, jobs)


# ---------------------------------------------------------------------- perception
def _perception_job(job: tuple[str, list[int], int]) -> dict[str, Any]:
    from ..perception.model import encode_image, encode_query

    family, seeds, max_targets = job
    imgs, scene_idx, queries, masks, occ = [], [], [], [], []
    for seed in seeds:
        w = World(sample_scene(family, seed))
        vis = w.visible_items()
        if not vis:
            continue
        rng = np.random.default_rng(seed)
        targets = list(rng.permutation(vis)[:max_targets])
        w.set_target(int(targets[0]))
        obs = w.observe()
        lab = w.labels()
        imgs.append(encode_image(obs.depth, obs.intensity).astype(np.float16))
        for t in targets:
            w.set_target(int(t))
            scene_idx.append(len(imgs) - 1)
            queries.append(encode_query(w.query(int(t))))
            masks.append(lab["seg"] == int(t))
            occ.append(w.occlusion_class(int(t), lab["seg"]))
    return {
        "images": np.array(imgs),
        "scene": np.array(scene_idx, np.int32),
        "query": np.array(queries, np.float32),
        "mask": np.array(masks, bool),
        "occlusion": np.array(occ, np.int8),
    }


def perception_data(family: str, seeds: list[int], max_targets: int, workers: int = 1) -> dict[str, np.ndarray]:
    parts = _pool_map(_perception_job, [(family, ch, max_targets) for ch in _chunks(seeds, workers * 4)], workers)
    out: dict[str, list[np.ndarray]] = {k: [] for k in ("images", "scene", "query", "mask", "occlusion")}
    offset = 0
    for p in parts:
        if len(p["images"]) == 0:
            continue
        out["images"].append(p["images"])
        out["scene"].append(p["scene"] + offset)
        offset += len(p["images"])
        for k in ("query", "mask", "occlusion"):
            out[k].append(p[k])
    return {k: np.concatenate(v) for k, v in out.items()}


# ---------------------------------------------------------------------- picks
def pick_order(w: World, rng: np.random.Generator) -> int | None:
    vis = w.visible_items()
    if not vis:
        return None
    t = int(rng.choice(vis))
    w.set_target(t)
    return t


def _picks_job(job: tuple[str, list[int], int, str]) -> dict[str, Any]:
    import torch

    from ..cell import load_cell
    from ..decide.planner import Option
    from ..grasp.candidates import generate
    from ..grasp.risk import primitives

    torch.set_num_threads(1)
    family, seeds, attempts, models_dir = job
    from pathlib import Path

    cell = load_cell("heuristic-open", Path(models_dir))
    prims = primitives()
    rows: dict[str, list[Any]] = {
        k: [] for k in ("crops", "features", "prim", "success", "collateral", "drop", "sealed", "order", "occ_true", "sig_max", "sig_pair", "p_min_inside")
    }
    for seed in seeds:
        rng = np.random.default_rng(seed + 17)
        w = World(sample_scene(family, seed))
        t = pick_order(w, rng)
        if t is None:
            continue
        occ_true = w.occlusion_class(t)
        start = w.snapshot()
        obs = w.observe()
        perc = cell.perception(obs)
        if perc["mask"].sum() < config.policy()["perception"]["min_mask_pixels"]:
            continue
        c = generate(obs, perc["mask"], perc["occlusion"])
        if len(c) == 0:
            continue
        pairs = [(int(k), int(p)) for k in range(len(c)) for p in range(len(prims))]
        chosen = rng.permutation(len(pairs))[:attempts]
        for j in chosen:
            k, p = pairs[j]
            w.restore(start)
            w.observe()  # keep the sensor noise stream identical to the deployed sequence
            w.open_band()
            a, _ = cell.attempt(w, c, Option(k, prims[p], float("nan")), perc["mask"], passive_monitor=True)
            w.settle(config.load("world")["outcome"]["settle_after_s"])
            o = w.outcome()
            rows["crops"].append(c.crops[k].astype(np.float16))
            rows["features"].append(c.features[k])
            rows["prim"].append(p)
            rows["success"].append(o["target_out"])
            rows["collateral"].append(o["collateral"] > 0)
            rows["drop"].append(o["target_dropped_out"])
            rows["sealed"].append(a.sealed)
            rows["order"].append(seed)
            rows["occ_true"].append(occ_true)
            sg = a.signals
            rows["sig_max"].append(max(sg) if sg else 0)
            rows["sig_pair"].append(max((min(x, y) for x, y in zip(sg, sg[1:], strict=False)), default=0))
            rows["p_min_inside"].append(min(a.pressures[1:4]) if len(a.pressures) > 1 else a.pressures[0] if a.pressures else 0.0)
    return {k: np.array(v) for k, v in rows.items()}


def pick_data(family: str, seeds: list[int], attempts: int, models_dir: str, workers: int = 1) -> dict[str, np.ndarray]:
    parts = _pool_map(_picks_job, [(family, ch, attempts, models_dir) for ch in _chunks(seeds, workers * 4)], workers)
    parts = [p for p in parts if len(p["prim"])]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


# ---------------------------------------------------------------------- unlabelled crops
def _crops_job(job: tuple[str, list[int], str]) -> np.ndarray:
    import torch

    from ..cell import load_cell
    from ..grasp.candidates import generate

    torch.set_num_threads(1)
    family, seeds, models_dir = job
    from pathlib import Path

    cell = load_cell("heuristic-open", Path(models_dir))
    out = []
    for seed in seeds:
        rng = np.random.default_rng(seed + 29)
        w = World(sample_scene(family, seed))
        if pick_order(w, rng) is None:
            continue
        obs = w.observe()
        perc = cell.perception(obs)
        if perc["mask"].sum() < config.policy()["perception"]["min_mask_pixels"]:
            continue
        c = generate(obs, perc["mask"], perc["occlusion"])
        out.extend(c.crops.astype(np.float16))
    return np.array(out) if out else np.zeros((0, 3, 32, 32), np.float16)


def unlabelled_crops(family: str, seeds: list[int], models_dir: str, workers: int = 1) -> np.ndarray:
    parts = _pool_map(_crops_job, [(family, ch, models_dir) for ch in _chunks(seeds, workers * 4)], workers)
    return np.concatenate([p for p in parts if len(p)])
