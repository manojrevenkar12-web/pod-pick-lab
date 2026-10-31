"""Smoke test against a running planner: python scripts/smoke.py http://localhost:8080

1. Renders bins in the simulator and asks the service for a plan; the plan must equal the one the
   local pick cell makes with the same models.
2. Opens a monitor session and streams a synthetic extraction in which a neighbour comes forward;
   the service must slow down and then abort.
"""

from __future__ import annotations

import base64
import sys

import httpx
import numpy as np

from podpick.cell import load_cell
from podpick.geometry import monitor_camera
from podpick.sim.datasets import pick_order
from podpick.sim.scene import sample_scene
from podpick.sim.world import World


def arr(a: np.ndarray) -> dict:
    a = np.ascontiguousarray(a, dtype="<f4")
    return {"shape": list(a.shape), "data": base64.b64encode(a.tobytes()).decode()}


def main(url: str) -> int:
    c = httpx.Client(base_url=url, timeout=30)
    ready = c.get("/readyz").json()
    print("ready:", ready)
    cell = load_cell("learned-closed")
    checked = agree = 0
    for seed in range(4_000_000, 4_000_012):
        w = World(sample_scene("mixed", seed))
        if pick_order(w, np.random.default_rng(seed)) is None:
            continue
        obs = w.observe()
        q = obs.query
        body = {
            "depth": arr(obs.depth),
            "intensity": arr(obs.intensity),
            "band_z": obs.band_z,
            "query": {"dims": list(q.dims), "albedo": q.albedo, "mass": q.mass},
            "policy": "learned-closed",
        }
        r = c.post("/v1/plan", json=body)
        r.raise_for_status()
        got = r.json()
        perc = cell.perception(obs)
        from podpick.decide.planner import plan_learned
        from podpick.grasp.candidates import generate

        cands = generate(obs, perc["mask"], perc["occlusion"])
        local = plan_learned(cands, cell.risk(cands)) if perc["mask"].sum() >= 25 else None
        want = [] if local is None else [[int(v) for v in cands.pixels[o.candidate]] for o in local.options[:2]]
        occ_local = np.array([float(x) for x in perc["occlusion"]])
        occ_got = np.array(list(got["occlusion"].values()))
        same_plan = [o["pixel"] for o in got["options"]] == want
        print(
            f"seed {seed}: {got['decision']} ({got['reason']}), {len(got['options'])} options, {got['plan_ms']} ms; "
            f"mask {got['mask_pixels']} vs {int(perc['mask'].sum())} px locally; same plan: {same_plan}"
        )
        # The image and the platform can differ (CPU kernels, library builds), so a pixel can cross
        # the mask threshold: compare what should be stable, and count exact agreement.
        assert np.abs(occ_local - occ_got).max() < 0.02, (seed, occ_local, occ_got)
        assert abs(got["mask_pixels"] - int(perc["mask"].sum())) <= max(3, 0.03 * got["mask_pixels"]), seed
        agree += int(same_plan)
        checked += 1
    print(f"identical plans: {agree} of {checked}")
    assert agree >= checked - 1, "plans from the service differ from the local cell"
    assert checked >= 5

    cam = monitor_camera()
    ref = np.full((cam.height, cam.width), 0.70, np.float32)
    mask = np.zeros_like(ref)
    ref[20:30, 26:38] = 0.50
    mask[20:30, 26:38] = 1
    ref[12:20, 26:38] = 0.52
    s = c.post("/v1/monitor", json={"reference": arr(ref), "mask": arr(mask), "tool": [0.0, 0.08, 0.12]}).json()
    print("monitor session:", s)
    f = ref.copy()
    f[20:30, 26:38] = 0.45
    f[12:17, 26:38] = 0.49
    actions = [
        c.post(f"/v1/monitor/{s['session']}/frame", json={"frame": arr(f), "tool": [0.0, 0.03, 0.12], "pressure": 0.8}).json()["action"] for _ in range(2)
    ]
    print("actions:", actions)
    assert actions == ["slow", "abort"], actions
    assert "podpick_plans_total" in c.get("/metrics").text
    print("smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"))
