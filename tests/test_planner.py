import numpy as np
import pytest

from podpick import config
from podpick.decide.planner import expected_cost, plan_heuristic, plan_learned
from podpick.grasp.candidates import FEATURES, Candidates


def cands(k: int) -> Candidates:
    f = np.zeros((k, len(FEATURES)), np.float32)
    f[:, FEATURES.index("edge_m")] = 0.02
    return Candidates(
        np.zeros((k, 2), int), np.zeros((k, 3)), np.tile([0, -1.0, 0], (k, 1)), f, np.zeros((k, 3, 32, 32), np.float32), np.linspace(-0.01, 0.03, k)
    )


def test_expected_cost_orders_by_risk():
    d = config.policy()["decision"]
    safe = expected_cost(np.array(0.9), np.array(0.0), np.array(0.0))
    risky = expected_cost(np.array(0.9), np.array(0.2), np.array(0.0))
    assert risky - safe == pytest.approx(0.2 * d["collateral_s"])


def test_learned_hands_over_when_every_option_costs_more_than_a_person():
    c = cands(3)
    probs = np.zeros((3, 3, 3))
    probs[..., 0] = 0.3
    probs[..., 1] = 0.4
    plan = plan_learned(c, probs)
    assert plan.handoff == "risk" and plan.options == []


def test_learned_picks_the_cheapest_candidate_and_primitive():
    c = cands(3)
    probs = np.zeros((3, 3, 3))
    probs[..., 0] = 0.8
    probs[..., 1] = 0.05
    probs[1, 2, 1] = 0.0
    plan = plan_learned(c, probs)
    assert plan.handoff is None
    first = plan.options[0]
    assert first.candidate == 1 and first.primitive == list(config.policy()["primitives"])[2]


def test_heuristic_ranks_by_clearance_and_checks_eligibility():
    c = cands(4)
    plan = plan_heuristic(c, np.array([0.9, 0.05, 0.05]))
    assert [o.candidate for o in plan.options][0] == 3
    assert plan_heuristic(c, np.array([0.1, 0.8, 0.1])).handoff == "ineligible"
