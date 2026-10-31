import numpy as np
import pytest

from podpick.evaluate.experiment import paired, summarise
from podpick.evaluate.gate import check, lookup
from podpick.train.risk import auc, ece


def test_auc_matches_the_pairwise_definition():
    rng = np.random.default_rng(0)
    y = rng.random(300) < 0.3
    s = rng.normal(y * 1.0, 1.0).round(1)  # ties on purpose
    pos, neg = s[y], s[~y]
    ref = ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg))
    assert auc(y, s) == pytest.approx(ref)
    assert auc(np.ones(5, bool), np.arange(5)) is None


def test_ece_is_zero_when_calibrated_in_every_bin():
    p = np.repeat([0.05, 0.25, 0.75], 400)
    y = np.concatenate([np.arange(400) < 20, np.arange(400) < 100, np.arange(400) < 300])
    assert ece(y, p) == pytest.approx(0.0, abs=1e-9)


def _row(seed, policy, picked, coll=0):
    return {
        "seed": seed,
        "policy": policy,
        "picked": picked,
        "collateral": coll,
        "dropped": False,
        "wrong_item": False,
        "aborted": False,
        "attempts": 1,
        "robot_s": 3.0,
    }


def test_paired_difference_and_summary():
    rows = [_row(i, "a", i % 2 == 0) for i in range(100)] + [_row(i, "b", True, coll=int(i < 5)) for i in range(100)]
    d = paired(rows, "a", "b", "picked")
    assert d["diff"] == pytest.approx(0.5) and d["lo"] < 0.5 < d["hi"]
    s = summarise([r for r in rows if r["policy"] == "b"])
    assert s["pick_rate"] == 1.0 and s["collateral_per_100"] == pytest.approx(5.0)


def test_gate_lookup_and_check():
    r = {"a": {"b-c": {"d": 0.8}}}
    assert lookup(r, "a.b-c.d") == 0.8 and lookup(r, "a.x") is None
    out = check(r, [{"name": "x", "path": "a.b-c.d", "op": ">=", "value": 0.7}, {"name": "y", "path": "a.zz", "op": ">=", "value": 0}])
    assert out[0]["ok"] and not out[1]["ok"]
