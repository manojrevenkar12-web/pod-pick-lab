"""The release gate over reports/results.json (or runs/<profile>/results.json)."""

from __future__ import annotations

import json
import operator
from typing import Any

from .. import config
from .pipeline import log

OPS = {">=": operator.ge, "<=": operator.le, "<": operator.lt, ">": operator.gt, "==": operator.eq}


def lookup(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def check(results: dict[str, Any], items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for it in items:
        v = lookup(results, it["path"])
        ok = v is not None and bool(OPS[it["op"]](v, it["value"]))
        out.append({**it, "actual": v, "ok": ok})
    return out


def results_path(profile: str):
    return config.ROOT / "reports" / "results.json" if profile == "full" else config.ROOT / "runs" / profile / "results.json"


def gate(profile: str = "full") -> bool:
    r = json.loads(results_path(profile).read_text())
    rel = config.load("release")
    floors = check(r, rel["floors"])
    targets = check(r, rel["targets"])
    for f in floors:
        log(f"{'PASS' if f['ok'] else 'FAIL'}  {f['name']}: {f['actual']} {f['op']} {f['value']}")
    for t in targets:
        log(f"{'met ' if t['ok'] else 'miss'}  target {t['name']}: {t['actual']} {t['op']} {t['value']}")
    ok = all(f["ok"] for f in floors)
    if profile != "full":
        # the quick profile trains on a handful of scenes: only the engineering floors apply
        ok = all(f["ok"] for f in floors if f["path"].startswith("native."))
    log("release gate: " + ("PASS" if ok else "FAIL"))
    return ok
