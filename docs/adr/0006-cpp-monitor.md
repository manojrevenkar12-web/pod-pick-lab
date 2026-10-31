# 6. The monitor in C++, held equal to a Python reference

**Context.** The extraction monitor runs on every frame at 10 Hz while the arm is moving; in a
real cell it sits in the motion-control loop, where Python is the wrong tool. A second
implementation is only useful if it is provably the same algorithm.

**Decision.** `cpp/monitor.cpp` (C++17, no dependencies, no allocation per frame, C ABI) implements
the monitor; `control/monitor.py` is the reference. Both use the same arithmetic in the same order
on doubles, and the library is built with `-ffp-contract=off` so the compiler does not fuse
multiply-adds. Tests compare signal and action on random sequences; `podpick native` compares them
on every frame of real extractions, and the release gate requires 100% agreement.

**Consequences.** Any change to the monitor is made in both places, or the gate fails. The service
uses the C++ monitor when the library is present and the reference otherwise; the answers are the same.
