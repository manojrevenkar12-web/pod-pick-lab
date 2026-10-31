"""Pick or hand over, and with which candidate and primitive.

Two scorers:
- `heuristic`: the free-space ranking (signed clearance in the extraction corridor) behind an
  eligibility check on the predicted occlusion class, as described for Vulcan Pick;
- `learned`: the risk model's outcome probabilities, turned into an expected cost in
  human-equivalent seconds (config/policy.yaml: decision), compared with handing the item to a
  person straight away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import config
from ..grasp.candidates import Candidates, heuristic_scores
from ..grasp.risk import primitives


@dataclass
class Option:
    candidate: int
    primitive: str
    cost: float  # expected human-equivalent seconds if attempted
    p: tuple[float, float, float] | None = None  # success, collateral, drop (learned only)


@dataclass
class Plan:
    options: list[Option]  # in the order they will be tried
    handoff: str | None  # reason to hand over now, or None
    handoff_cost: float


def expected_cost(p_success: np.ndarray, p_collateral: np.ndarray, p_drop: np.ndarray) -> np.ndarray:
    d = config.policy()["decision"]
    robot = d["robot_second"] * d["attempt_s"]
    return robot + p_collateral * d["collateral_s"] + p_drop * d["drop_s"] + (1.0 - p_success) * d["human_pick_s"]


def plan_heuristic(c: Candidates, occlusion: np.ndarray) -> Plan:
    h = config.policy()["heuristic"]
    human = config.policy()["decision"]["human_pick_s"]
    if len(c) == 0:
        return Plan([], "no_candidate", human)
    if int(np.argmax(occlusion)) not in h["eligible_classes"]:
        return Plan([], "ineligible", human)
    order = np.argsort(-heuristic_scores(c), kind="stable")
    return Plan([Option(int(k), h["primitive"], float("nan")) for k in order], None, human)


def plan_learned(c: Candidates, probs: np.ndarray) -> Plan:
    human = config.policy()["decision"]["human_pick_s"]
    if len(c) == 0:
        return Plan([], "no_candidate", human)
    prims = primitives()
    cost = expected_cost(probs[..., 0], probs[..., 1], probs[..., 2])  # (K, P)
    best_prim = np.argmin(cost, axis=1)
    per_cand = cost[np.arange(len(c)), best_prim]
    order = np.argsort(per_cand, kind="stable")
    options = [
        Option(int(k), prims[int(best_prim[k])], float(per_cand[k]), tuple(float(x) for x in probs[k, best_prim[k]])) for k in order if per_cand[k] < human
    ]
    return Plan(options, None if options else "risk", human)
