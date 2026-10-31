"""The planner service with the committed models: plans match the local cell; monitor sessions work."""

import base64

import numpy as np
import pytest
from fastapi.testclient import TestClient

from podpick import config

pytestmark = pytest.mark.skipif(not (config.ROOT / "models" / "risk.pt").exists(), reason="needs the committed models")


def arr(a):
    a = np.ascontiguousarray(a, dtype="<f4")
    return {"shape": list(a.shape), "data": base64.b64encode(a.tobytes()).decode()}


@pytest.fixture(scope="module")
def client():
    from podpick.service.app import app

    return TestClient(app)


def test_health_and_ready(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json()["status"] == "ready"


def test_plan_matches_the_local_cell(client, world):
    from podpick.cell import load_cell
    from podpick.decide.planner import plan_learned
    from podpick.grasp.candidates import generate

    obs = world.observe()
    q = obs.query
    r = client.post(
        "/v1/plan",
        json={
            "depth": arr(obs.depth),
            "intensity": arr(obs.intensity),
            "band_z": obs.band_z,
            "query": {"dims": list(q.dims), "albedo": q.albedo, "mass": q.mass},
        },
    )
    assert r.status_code == 200
    got = r.json()
    cell = load_cell("learned-closed")
    perc = cell.perception(obs)
    c = generate(obs, perc["mask"], perc["occlusion"])
    plan = plan_learned(c, cell.risk(c))
    assert [o["pixel"] for o in got["options"]] == [[int(v) for v in c.pixels[o.candidate]] for o in plan.options[:2]]
    assert set(got["occlusion"]) == {"unobstructed", "below_others", "blocked"}


def test_bad_shapes_are_rejected(client):
    body = {
        "depth": arr(np.zeros((10, 10))),
        "intensity": arr(np.zeros((10, 10))),
        "band_z": 0.1,
        "query": {"dims": [0.1, 0.1, 0.1], "albedo": 0.5, "mass": 0.2},
    }
    assert client.post("/v1/plan", json=body).status_code == 422
    body["depth"]["data"] = body["depth"]["data"][:8]
    assert client.post("/v1/plan", json=body).status_code == 422


def test_monitor_session_slows_then_aborts(client):
    ref = np.full((48, 64), 0.70, np.float32)
    mask = np.zeros_like(ref)
    ref[20:30, 26:38] = 0.50
    mask[20:30, 26:38] = 1
    ref[12:20, 26:38] = 0.52
    sid = client.post("/v1/monitor", json={"reference": arr(ref), "mask": arr(mask), "tool": [0.0, 0.08, 0.12]}).json()["session"]
    f = ref.copy()
    f[20:30, 26:38] = 0.45
    f[12:17, 26:38] = 0.49
    acts = [client.post(f"/v1/monitor/{sid}/frame", json={"frame": arr(f), "tool": [0.0, 0.03, 0.12], "pressure": 0.8}).json()["action"] for _ in range(2)]
    assert acts == ["slow", "abort"]
    assert client.delete(f"/v1/monitor/{sid}").json() == {"status": "closed"}
    assert client.post(f"/v1/monitor/{sid}/frame", json={"frame": arr(f), "tool": [0, 0, 0], "pressure": 0.8}).status_code == 404
    assert "podpick_monitor_frames_total" in client.get("/metrics").text
