"""reports/results.json and docs/REPORT.md, generated from a run. Do not edit them by hand."""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

from .. import config
from .experiment import POLICIES
from .gate import check
from .pipeline import run_dir, write_json

FAMILY_LABEL = {"mixed": "mixed (training distribution)", "dense": "dense bins", "soft": "mostly polybags", "heldout": "held-out (blind)"}


def _load(path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def collect(profile: str) -> dict[str, Any]:
    rd = run_dir(profile)
    ev = _load(rd / "evaluation.json") or {}
    r: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "profile": profile,
        "perception": _load(rd / "perception.json"),
        "picks": _load(rd / "picks.json"),
        "risk": _load(rd / "risk.json"),
        "native": _load(rd / "native.json"),
        "monitor_tuning": _load(rd / "monitor_tuning.json"),
        "evaluation": {f: {"analysis": v["analysis"]} for f, v in ev.items()},
        "config": {"decision": config.policy()["decision"], "monitor": config.policy()["monitor"]},
    }
    first = config.ROOT / "reports" / "heldout" / "first_run.json"
    if profile == "full" and first.exists():
        r["heldout_first_run"] = _load(first)
    return r


def _compact_rows(profile: str) -> list[dict[str, Any]]:
    ev = _load(run_dir(profile) / "evaluation.json") or {}
    keep = (
        "family",
        "seed",
        "policy",
        "occlusion_true",
        "material",
        "status",
        "reason",
        "attempts",
        "robot_s",
        "picked",
        "collateral",
        "dropped",
        "wrong_item",
        "first_primitive",
    )
    out = []
    for v in ev.values():
        for row in v["rows"]:
            c = {k: row[k] for k in keep}
            c["robot_s"] = round(c["robot_s"], 2)
            c["first_p"] = [round(x, 3) for x in row["first_p"]] if row["first_p"] else None
            c["max_signal"] = max(row["monitor_signals"]) if row["monitor_signals"] else 0
            out.append(c)
    return out


def write_reports(profile: str) -> dict[str, Any]:
    r = collect(profile)
    rel = config.load("release")
    r["gate"] = {"floors": check(r, rel["floors"]), "targets": check(r, rel["targets"])}
    out = config.ROOT / "reports" if profile == "full" else run_dir(profile)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "results.json", r)
    write_json(out / "orders.json", _compact_rows(profile))
    target = config.ROOT / "docs" / "REPORT.md" if profile == "full" else out / "REPORT.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(markdown(r))
    return r


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{100 * v:.1f}%"


def _ci(d: dict[str, Any], scale: float = 100.0, unit: str = " pts") -> str:
    if not d:
        return "n/a"
    return f"{scale * d['diff']:+.1f}{unit} [{scale * d['lo']:+.1f}, {scale * d['hi']:+.1f}]"


def _policy_table(an: dict[str, Any]) -> str:
    lines = [
        "| Policy | Picked | Attempted | Success when attempted | Collateral / 100 orders | Target dropped | Aborted | Robot s / order | Cost / order (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in POLICIES:
        v = an["policies"].get(p)
        if not v:
            continue
        lines.append(
            f"| {p} | {_pct(v['pick_rate'])} | {_pct(v['attempt_rate'])} | {_pct(v['success_when_attempted'])} | "
            f"{v['collateral_per_100']:.1f} | {_pct(v['drop_rate'])} | {_pct(v['abort_rate'])} | {v['robot_s_per_order']:.2f} | "
            f"{v['cost_per_order']:.2f} |"
        )
    return "\n".join(lines)


def _diff_table(an: dict[str, Any]) -> str:
    vs = an.get("vs_heuristic_open", {})
    lines = ["| Against heuristic-open | Picked | Collateral items / order | Target dropped | Cost / order (s) |", "|---|---:|---:|---:|---:|"]
    for p in POLICIES[1:]:
        if p in vs:
            d = vs[p]
            lines.append(f"| {p} | {_ci(d['picked'])} | {_ci(d['collateral'], 100, ' / 100')} | {_ci(d['dropped'])} | {_ci(d['cost'], 1.0, ' s')} |")
    return "\n".join(lines)


def _occ_table(an: dict[str, Any]) -> str:
    lines = [
        "| Policy | Unobstructed: picked | collateral / 100 | Below others: picked | collateral / 100 | Blocked: picked | collateral / 100 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for p in POLICIES:
        b = an["by_occlusion"].get(p)
        if not b:
            continue
        cells = []
        for k in ("unobstructed", "below_others", "blocked"):
            v = b[k]
            cells += ["n/a", "n/a"] if not v.get("orders") else [_pct(v["pick_rate"]), f"{v['collateral_per_100']:.1f}"]
        lines.append(f"| {p} | " + " | ".join(cells) + " |")
    n = {k: an["by_occlusion"][POLICIES[0]][k].get("orders", 0) for k in ("unobstructed", "below_others", "blocked")}
    return "\n".join(lines) + f"\n\nOrders per class: unobstructed {n['unobstructed']}, below others {n['below_others']}, blocked {n['blocked']}."


def _risk_table(risk: dict[str, Any]) -> str:
    rows = risk["rows"]
    budgets = sorted({x["budget"] for x in rows})
    lines = [
        "| Labelled attempts | Success AUC, scratch | Success AUC, pretrained | Collateral AUC, scratch | Collateral AUC, pretrained |",
        "|---:|---:|---:|---:|---:|",
    ]

    def ms(b: int, v: str, k: str) -> str:
        xs = [x[k] for x in rows if x["budget"] == b and x["variant"] == v and x[k] is not None]
        if not xs:
            return "n/a"
        return f"{np.mean(xs):.3f}" + (f" ± {np.std(xs):.3f}" if len(xs) > 1 else "")

    for b in budgets:
        lines.append(
            f"| {b:,} | {ms(b, 'scratch', 'success_auc')} | {ms(b, 'pretrained', 'success_auc')} | "
            f"{ms(b, 'scratch', 'collateral_auc')} | {ms(b, 'pretrained', 'collateral_auc')} |"
        )
    return "\n".join(lines)


def markdown(r: dict[str, Any]) -> str:
    p, pk, risk, nat, ev = r["perception"], r["picks"], r["risk"], r["native"], r["evaluation"]
    parts = [f"# Report\n\nGenerated by `podpick report` ({r['generated_at']}, profile `{r['profile']}`). Do not edit.\n"]
    if pk:
        parts.append(
            f"Training data: {p['train_samples']:,} labelled targets for perception; {pk['attempts']:,} logged pick attempts "
            f"(random candidate and primitive, open loop; success {_pct(pk['success'])}, collateral {_pct(pk['collateral'])}, "
            f"target dropped {_pct(pk['drop'])}); {pk['unlabelled_crops']:,} unlabelled crops for pretraining. "
            "Costs are human-equivalent seconds from `config/policy.yaml: decision`.\n"
        )
    if p:
        conf = p["occlusion_confusion"]
        parts.append(
            "## Q1. Finding the ordered item\n\n"
            f"On validation bins the target finder reaches a mean mask IoU of {p['mask_iou_mean']:.3f} and finds the item "
            f"(more than half its pixels) in {_pct(p['target_found'])} of orders. Occlusion class accuracy: "
            f"{_pct(p['occlusion_accuracy'])}.\n\n| True \\ predicted | unobstructed | below others | blocked |\n|---|---:|---:|---:|\n"
            + "\n".join(f"| {n} | " + " | ".join(str(x) for x in conf[i]) + " |" for i, n in enumerate(("unobstructed", "below others", "blocked")))
            + "\n"
        )
    if "mixed" in ev:
        an = ev["mixed"]["analysis"]
        parts.append(
            "## Q2. Which spot, and whether to try at all\n\nPaired: every policy picks the same orders from the same start state.\n\n"
            + _policy_table(an)
            + "\n\nDifferences with 95% bootstrap intervals over orders:\n\n"
            + _diff_table(an)
            + "\n\nBy the true occlusion class of the target:\n\n"
            + _occ_table(an)
            + "\n"
        )
        if an.get("learned_promises"):
            lp = an["learned_promises"]
            parts.append(
                f"What the learned model promised for its first choice against what happened (learned-open, {lp['attempts']} attempts): "
                f"success {_pct(lp['predicted_success'])} predicted, {_pct(lp['observed_success'])} observed; collateral "
                f"{_pct(lp['predicted_collateral'])} predicted, {_pct(lp['observed_collateral'])} observed.\n"
            )
    if risk:
        parts.append(
            f"## Q3. Does pretraining without labels help?\n\nValidation orders, never test orders. Final model: `{risk['chosen']}`, "
            f"chosen on {risk['chosen_on']}.\n\n" + _risk_table(risk) + "\n"
        )
    if "mixed" in ev:
        an = ev["mixed"]["analysis"]
        parts.append(
            "## Q4. Watching while pulling\n\nThe monitor's effect, paired, on the mixed bins:\n\n| | Picked | Collateral / 100 | Target dropped | Cost / order (s) |\n|---|---:|---:|---:|---:|\n"
            + f"| heuristic: closed minus open | {_ci(an['monitor_effect_heuristic']['picked'])} | {_ci(an['monitor_effect_heuristic']['collateral'], 100, '')} | "
            f"{_ci(an['monitor_effect_heuristic']['dropped'])} | {_ci(an['monitor_effect_heuristic']['cost'], 1.0, '')} |\n"
            + f"| learned: closed minus open | {_ci(an['monitor_effect_learned']['picked'])} | {_ci(an['monitor_effect_learned']['collateral'], 100, '')} | "
            f"{_ci(an['monitor_effect_learned']['dropped'])} | {_ci(an['monitor_effect_learned']['cost'], 1.0, '')} |\n"
        )
    mt = r.get("monitor_tuning")
    if mt:
        parts.append(
            "How the abort thresholds were chosen: from passive monitor logs on validation orders, for the attempts each policy "
            "would have made, by expected cost (docs/adr/0004). Share of attempts a threshold stops (signal held two frames):\n"
        )
        for pol, v in mt["policies"].items():
            parts.append(
                f"{pol}: {v['sealed_attempts']} sealed attempts, {v['collateral_attempts']} with a neighbour out.\n\n"
                "| Neighbour pixels | Collateral stopped | Clean picks stopped | Saving per attempt (s) |\n|---:|---:|---:|---:|\n"
                + "\n".join(f"| {x['abort_px']} | {_pct(x['collateral_stopped'])} | {_pct(x['clean_stopped'])} | {x['saving_s_per_attempt']:+.2f} |" for x in v["table"])
                + f"\n\nRecommended {v['recommended_abort_px']}, configured {v['configured_abort_px']}.\n"
            )
    for fam in ("dense", "soft"):
        if fam in ev:
            an = ev[fam]["analysis"]
            parts.append(f"## Q5. Shift: {FAMILY_LABEL[fam]}\n\n" + _policy_table(an) + "\n\n" + _diff_table(an) + "\n")
    if nat:
        parts.append(
            "## Q6. Real time\n\nThe C++ monitor against the Python reference on every extraction frame of "
            f"{nat['frames']} frames: signal identical on {_pct(nat['signal_equal'])}, action identical on {_pct(nat['action_equal'])}. "
            f"Median time per 64 × 48 frame: C++ {nat['native_us_median']:.0f} µs (p99 {nat['native_us_p99']:.0f} µs), Python "
            f"{nat['python_us_median']:.0f} µs. The monitor runs at 10 Hz, so it uses well under 1% of a 100 ms tick.\n"
        )
    if r.get("heldout_first_run"):
        f = r["heldout_first_run"]
        parts.append(
            f"## Held-out bins (blind)\n\nWritten from `heldout/WRITING_BINS.md` by someone who had not seen the code; run once "
            f"({f['recorded_at']}, spec sha256 `{f['spec_sha256'][:12]}…`), never rewritten.\n\n"
            + _policy_table(f["analysis"])
            + "\n\n"
            + _diff_table(f["analysis"])
            + "\n"
        )
    g = r.get("gate")
    if g:
        parts.append(
            "## Release gate\n\n| | Check | Actual | Required |\n|---|---|---:|---:|\n"
            + "\n".join(f"| {'pass' if x['ok'] else '**fail**'} | {x['name']} | {_fmt(x['actual'])} | {x['op']} {x['value']} |" for x in g["floors"])
            + "\n\nTargets (reported, not gated):\n\n| | Target | Actual | Aim |\n|---|---|---:|---:|\n"
            + "\n".join(f"| {'met' if x['ok'] else 'missed'} | {x['name']} | {_fmt(x['actual'])} | {x['op']} {x['value']} |" for x in g["targets"])
            + "\n"
        )
    return "\n".join(parts)


def _fmt(v: Any) -> str:
    return "n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
