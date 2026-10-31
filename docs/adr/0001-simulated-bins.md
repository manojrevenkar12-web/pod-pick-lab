# 1. Simulated bins, real contact physics

**Context.** The problem worth solving in targeted picking from fabric pods is the one the Vulcan
Pick write-up names: pulling the ordered item out without bringing its neighbours with it. That
is a contact problem: an item resting on the target rides out on it, a wedged item resists, a
poor seal peels under load. A study of it needs friction, stacking and a seal that can fail, and
it needs to know exactly what happened on every pick.

**Decision.** Bins are simulated in MuJoCo: items are boxes with material-specific friction,
softness and seal quality, dropped into a fabric-walled bin with a sewn lip and an elastic band,
then settled. The camera is ray-cast (depth and intensity, noise, dropouts, a band that is partly
see-through), so no GPU or renderer is needed. Every scene is reproducible from its family and seed,
and any instant can be snapshotted, so every policy is tested on exactly the same orders.

**Consequences.** Every rate in the report is a rate in this simulator. What transfers is the
method (the paired protocol, the counterfactual labels, the monitor and its parity tests, the
decision rule) and the plumbing, not the numbers. Deformable items are approximated by soft
contacts on boxes; the arm link does not collide with items (only the cup does), which understates
disturbance on approach.
