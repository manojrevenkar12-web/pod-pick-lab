"""The blind held-out family: written from heldout/WRITING_BINS.md only, run, first run kept forever."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from .. import config
from .experiment import analyse, run_family
from .pipeline import exp, log, models_dir, run_dir, write_json

HELD = config.ROOT / "reports" / "heldout"


def run_heldout(profile: str) -> dict[str, Any]:
    e = exp(profile)
    spec = config.ROOT / "heldout" / "heldout.yaml"
    sha = hashlib.sha256(spec.read_bytes()).hexdigest()
    rows = run_family("heldout", e["test_orders"], str(models_dir(profile)), e["workers"])
    res = {"analysis": analyse(rows), "rows": rows, "spec_sha256": sha}
    write_json(run_dir(profile) / "heldout.json", res)
    first = HELD / "first_run.json"
    if profile == "full" and not first.exists():
        HELD.mkdir(parents=True, exist_ok=True)
        write_json(first, {"recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "spec_sha256": sha, "orders": e["test_orders"], "analysis": res["analysis"]})
        log("held-out first run recorded")
    elif first.exists():
        rec = json.loads(first.read_text())
        if rec["spec_sha256"] != sha:
            raise SystemExit("heldout/heldout.yaml changed after its first run; a changed set is a new set (see AGENTS.md)")
    s = res["analysis"]["policies"]
    log("held-out: " + "; ".join(f"{p} pick {v['pick_rate']:.2f} coll/100 {v['collateral_per_100']:.1f}" for p, v in s.items()))
    return res
