"""The whole pipeline on the quick profile: data, models, paired evaluation, C++ parity, report, gate."""

import json

import pytest

from podpick import config
from podpick.evaluate.pipeline import run_all


@pytest.mark.slow
def test_quick_pipeline_end_to_end():
    assert run_all("quick")
    r = json.loads((config.ROOT / "runs" / "quick" / "results.json").read_text())
    assert r["native"]["signal_equal"] == 1.0 and r["native"]["action_equal"] == 1.0
    an = r["evaluation"]["mixed"]["analysis"]
    assert set(an["policies"]) == {"heuristic-open", "heuristic-closed", "learned-open", "learned-closed"}
    n = {p: v["orders"] for p, v in an["policies"].items()}
    assert len(set(n.values())) == 1, "every policy picks the same orders"
    assert (config.ROOT / "runs" / "quick" / "REPORT.md").read_text().startswith("# Report")
