# Working in pod-pick-lab

## Fast feedback

```bash
pytest -q -m "not slow"                  # seconds: simulator, candidates, planner, monitor, C++ parity, isolation, service
cmake -S cpp -B cpp/build && cmake --build cpp/build && ctest --test-dir cpp/build   # the C++ monitor on its own
podpick all --profile quick              # the whole pipeline on a handful of bins, a few minutes
pytest -q -m slow                        # the same, as a test
podpick gate                             # the release gate on the committed full run (reports/results.json)
cd web && npm test && npm run lint       # the dashboard
```

`podpick all` (full profile) rebuilds everything that is committed: `models/`, `reports/`,
`docs/REPORT.md`. It takes a couple of hours on two cores.

## Invariants (each has a test)

1. **The pick cell never sees the truth.** `cell.py`, `perception/`, `grasp/`, `control/`,
   `decide/`, `service/` never import `podpick.sim` or `podpick.evaluate` and never read
   `config/world.yaml`. The cell talks to a `Driver` (`interface.py`) and nothing else.
   `tests/test_isolation.py` scans the imports and runs the cell against a driver that raises on
   any question about the truth.
2. **The C++ monitor equals the Python reference**, signal and action, on every frame
   (`tests/test_monitor.py`, and `podpick native` on real extraction frames, gated at 100%). Keep
   the arithmetic in the same order in both, and build with `-ffp-contract=off`.
3. **Settings live in `config/`**, not in code: the world (simulator only), the robot, the policy
   (thresholds, costs, primitives), the experiment, the release gate.
4. **Nothing is tuned on a test family.** Monitor thresholds come from passive monitor logs on
   training orders; the risk model's variant is chosen on validation orders. Test families and the
   held-out family are only ever evaluated.
5. **Reports are generated.** `reports/results.json`, `reports/orders.json` and `docs/REPORT.md` are
   written by `podpick report`; do not edit them.
6. **Release floors are never lowered to pass.** Missed targets are reported as missed.

## The held-out rule

`heldout/heldout.yaml` was written from `heldout/WRITING_BINS.md` alone by someone who had not seen
the code, the configuration or any result. Its sha256 is in `heldout/heldout.sha256`. Its first run
is `reports/heldout/first_run.json` and is never rewritten; `podpick heldout` refuses to run if the
file has changed. If you study its failures to change the cell, it has become a development set:
move it to `config/families.yaml`, and have someone who has not seen the code write a new
`heldout/heldout.yaml` from `heldout/WRITING_BINS.md` only.

## Where things are

```
src/podpick/sim/        MuJoCo bins, suction, ray-cast sensors, datasets (truth lives here)
src/podpick/perception/ target finder (U-Net + FiLM on the order) and occlusion class
src/podpick/grasp/      suction candidates, clearance features, risk model
src/podpick/decide/     heuristic ranking and expected-cost decision
src/podpick/control/    extraction monitor (reference) and the ctypes binding to cpp/
src/podpick/cell.py     the pick cell: observe, choose, approach, seal, extract, abort
src/podpick/service/    FastAPI planner and monitor sessions
src/podpick/evaluate/   paired experiments, held-out run, report, gate, pipeline steps
src/podpick/train/      perception and risk training (reads data files only)
cpp/                    real-time monitor, its tests and CMake build
web/                    dashboard over reports/results.json and reports/orders.json
deploy/                 Docker image and kustomize manifests
```
