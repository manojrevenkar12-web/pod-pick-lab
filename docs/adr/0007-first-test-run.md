# 7. The first test run, and what changed after it

**What happened.** The first evaluation on the test families was started with the models and
thresholds of that day. It finished the `mixed` family before it was stopped: every policy picked
about one order in ten (ranking 10%, learned 11%), with 6.0 and 1.2 neighbours out per 100 orders.
Only those aggregate numbers were seen. The `dense` and `soft` families and the held-out family were
not run.

**Why it was stopped.** A robot that hands nine orders in ten to a person is not a useful baseline
for anything. The causes were studied on validation orders only (seed range `picks_val`), never on
test orders:

- about a third of approaches touched the wrong item first: the target finder, trained for 8 epochs,
  reached a mask IoU of 0.61 and its masks bled into neighbours;
- the free-space ranking pulled straight out, dragging items over the sewn lip; the training logs
  already showed that lifting first works better (38% of sealed attempts succeeded against 30%), so the ranking
  now lifts;
- light boxes peeled off the cup while being dragged over a 12 mm lip on a high-friction floor.

**What changed.** Perception is trained for 16 epochs (validation mask IoU 0.61 to 0.69, occlusion class 67% to 74%). The ranking uses the `lift` primitive. The
lip is 8 mm, fabric friction 0.45 and the cup's capacity 55 N at a perfect seal (about what a 35 mm
cup gives at 60 kPa). Every dataset and model was rebuilt from scratch after these changes.

**Consequences.** The test families have been seen once, in aggregate, and the changes above were
made after that. They were not tuned on (no number from a test family was used to choose a value),
but a reader should know they were seen. The held-out family was not run before the final models
existed; its first run is the only one.
