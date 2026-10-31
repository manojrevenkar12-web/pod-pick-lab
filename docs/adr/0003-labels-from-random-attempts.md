# 3. Risk labels from random, counterfactual attempts

**Context.** The risk model needs outcomes for candidates the deployed policy would not choose,
or it can only confirm its own habits. Fleet logs have this problem; a simulator does not have to.

**Decision.** Pick data is collected by the deployed perception with a uniformly random behaviour
policy over (candidate, primitive), open loop. Each order gets several attempts, every one from the
same snapshot of the bin, so the labels for one order are counterfactuals of each other. Budgets are
counted in whole orders, and validation orders come from a separate seed range.

**Consequences.** The labels describe open-loop outcomes. A closed-loop policy changes outcomes (it
aborts some attempts), so its expected cost is computed from open-loop probabilities: the monitor
can only make an attempt cheaper than predicted, never riskier, as long as an abort costs less than
what it prevents.
