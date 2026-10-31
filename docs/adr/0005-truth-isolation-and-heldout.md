# 5. The cell never sees the truth; one blind held-out family

**Decision.** The pick cell (`cell.py`, perception, grasp, decide, control, service) talks to a
`Driver` that exposes sensors and motion only. The simulator implements it and, separately, the
truth the evaluation reads. `tests/test_isolation.py` checks imports and runs the cell against a
driver that raises on anything else.

A held-out bin family was written from `heldout/WRITING_BINS.md` alone by someone who had not seen
the code or the other families, frozen by its sha256 and run once. Its first run is kept as recorded.

**Consequences.** Training labels come from the simulator's truth (as annotation or fleet outcomes
would); decisions at pick time never do.
