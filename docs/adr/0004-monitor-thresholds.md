# 4. Monitor thresholds from passive logs, per policy, by expected cost

**Context.** The monitor aborts a pick when enough neighbour pixels come toward the camera on two
frames in a row. Too low a threshold puts back clean picks for nothing; too high lets neighbours
fall out. The threshold must not be chosen by looking at a test family.

**What changed during the build (all before any test run).**

1. The first monitor counted every neighbour pixel that moved more than 1.2 cm. On development
   bins it fired on the item's own sides and top as they came into view, and on the item sagging on
   the cup. The expected item region now covers the catalogue length behind the visible face and up
   to 3 cm of sag.
2. With the deployed perception, passive logs showed the signal on a third of clean picks. Two
   causes: perception misses part of the face (the missed part then "moves"), and neighbours get
   nudged a centimetre or two without coming out. The target mask is now grown over the sealed face
   by depth connectivity from the cup's pixel (bounded by the catalogue size), and a neighbour pixel
   only counts once it has come forward by half the tool's own travel: a rider keeps pace, a nudged
   neighbour does not.
3. The first threshold rule (the lowest threshold that stops at most 5% of clean picks) was met by
   no threshold on the training logs. A rule on cost replaced it, and because the share of attempts
   that end in collateral differs by policy (the random logs have far more than the learned cell
   will accept), each closed-loop policy gets its own threshold.

**Decision.** Passive monitor logs are recorded on every extraction during pick-data collection.
`podpick tune` selects, from the validation orders, the attempts each policy would have made
(ranking: predicted unobstructed, straight pull; learned: expected cost below a person's), and
picks the threshold with the largest expected saving in human-equivalent seconds:
collateral attempts stopped times `collateral_s`, minus clean picks stopped times
`human_pick_s + abort_s`. The values are copied into `config/policy.yaml` by hand, so the choice is
visible in review. The table is in `docs/REPORT.md`.

**Consequences.** The saving assumes an abort comes in time to keep the neighbour in. Whether it
does is measured only by the paired closed-loop evaluation and reported whatever it shows.
