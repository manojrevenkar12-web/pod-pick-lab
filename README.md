# pod-pick-lab

**Picking one ordered item out of a crowded fabric bin, without bringing its neighbours with it.**

A suction robot reaches into a storage-pod bin behind an elastic band, finds the item the order
asks for, puts its cup on it and pulls it out. The failure that costs most is not a missed pick:
it is the item resting on top, or wedged beside, coming out with the target and landing on the
floor. This repository builds and measures a pick cell around that problem:

- **perception**: a U-Net, told what the order is (catalogue size, colour, weight), finds the item
  in the depth and intensity image and says whether it is free, below other items or mostly hidden;
- **where to put the cup**: suction candidates on the item, described by the free space in their
  extraction corridor and what touches the item on each side;
- **whether to try at all**: a learned risk model predicts, for every candidate and every way of
  pulling, the chance of a clean pick, of a neighbour coming out, and of dropping the item; the cell
  attempts only when the expected cost is below a person's (in human-equivalent seconds);
- **watching while pulling**: an extraction monitor compares 10 Hz depth frames with the frame at
  the seal and puts the item back when a neighbour keeps pace with it. It is a dependency-free C++
  library, held equal to a Python reference on every frame;
- **a planner service** (FastAPI) with monitor sessions, a Docker image and Kubernetes manifests,
  tested end to end on a kind cluster in CI; a **dashboard** over the generated results.

Bins are simulated in MuJoCo with real contact (friction, stacking, a sewn lip, an elastic band, a
suction cup that peels under load) and a ray-cast camera ([ADR 1](docs/adr/0001-simulated-bins.md)),
so every pick's outcome is known exactly and every policy can be run on the same orders.

RESULTS_PLACEHOLDER

## How it is built

```
sim/          bins (MuJoCo): items dropped and settled, band, lip, suction as a compliant weld that
              peels over capacity; ray-cast depth + intensity with noise and a see-through band;
              snapshots for counterfactuals; datasets (perception labels, logged attempts, crops)
perception/   TargetNet: U-Net on depth + intensity, FiLM-conditioned on the order; mask + occlusion class
grasp/        candidates (normals, planarity, edge distance, corridor clearance, side contacts, 32x32
              crops); RiskNet: crop encoder (masked-reconstruction pretraining) + candidate-conditioned
              cross-attention, three heads
decide/       free-space ranking with eligibility; expected-cost decision over candidates x primitives
control/      the extraction monitor (reference) and the ctypes binding to cpp/
cell.py       observe, perceive, choose, open the band, approach to contact, seal, extract, abort
service/      POST /v1/plan, monitor sessions (/v1/monitor), /metrics, /readyz
evaluate/     paired experiments, tuning, held-out run, C++ parity bench, report, release gate
cpp/          monitor.cpp (C++17, no allocation per frame), unit tests, CMake
web/          React 19 + Motion dashboard; reads reports/*.json, computes nothing
deploy/       Dockerfile (C++ built and tested in a build stage), kustomize (restricted pods, no egress)
```

Decisions are in [`docs/adr/`](docs/adr). The full report, generated, is
[`docs/REPORT.md`](docs/REPORT.md).

## Run it

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[train,service,dev]"
pytest -m "not slow"              # seconds
podpick all --profile quick       # the whole pipeline on a few bins, a few minutes
podpick all                       # the committed run: about two hours on two cores
podpick heldout                   # the blind family (its first run is already recorded)
podpick serve                     # the planner on :8080 with the committed models
python scripts/smoke.py http://localhost:8080
cd web && npm ci && npm run dev   # the dashboard
```

| Endpoint | |
|---|---|
| `POST /v1/plan` | Depth and intensity (base64 float32), the order, the band height. Returns pick or hand over, the candidates in order, their outcome probabilities and expected cost. |
| `POST /v1/monitor` | Opens a monitor session with the frame and mask at the seal. |
| `POST /v1/monitor/{id}/frame` | One 10 Hz frame, tool position and vacuum: `continue`, `slow` or `abort`. |
| `GET /healthz` `/readyz` `/metrics` | Liveness, models and C++ monitor loaded, Prometheus counters. |

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)): ruff and the tests (including C++
parity and truth isolation) and the release gate on the committed run; the C++ monitor with CMake,
`-Werror`, ctest and AddressSanitizer/UBSan; the quick pipeline end to end; the planner image
deployed to kind and checked against plans made locally; the dashboard with Biome, Vitest, a build,
and Playwright with axe on desktop and phone.

## Limits

- **Simulated, not real.** Items are boxes; polybags are soft boxes; the arm link does not touch
  items. The rates are rates in this simulator. What carries over is the method: paired
  counterfactual evaluation, risk labels from random attempts, thresholds chosen on validation
  logs, and a real-time monitor proven equal to its reference.
- **Perception is the weakest part.** LIMIT_PERCEPTION
- **The test families were seen once before the final run** (aggregate only); see
  [ADR 7](docs/adr/0007-first-test-run.md). The held-out family was not.
- **CPU only.** Models are small; nothing here was trained on a GPU.
