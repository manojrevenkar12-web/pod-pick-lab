# 2. Suction as a compliant weld with a capacity

**Context.** A first version applied the cup as an explicit spring force at the contact point.
Light items (40-80 g) went unstable at a 2 ms step, and a point force let items swing freely, which
real bellows cups do not allow.

**Decision.** The cup is a soft weld equality constraint between the tool and the item, anchored at
the contact point (MuJoCo solves it implicitly, so it is stable for any item). Seal quality is set
when the vacuum is switched on: material, the angle between the cup and the face, how far the cup is
from a face edge, and whether the face is wider than the cup. The constraint force and moment are
read every step; above capacity (55 N times seal quality, and that force times a 22 mm lever for the
moment) for more than 30 ms, the seal peels. Vacuum pressure, the only signal the cell gets, falls as
the load approaches capacity.

**Consequences.** Failures come from physics rather than from dice: heavy items held off-centre,
items wedged by their neighbours or pinned by what rests on them, polybags with poor seals. The
tool accelerates at 2.5 m/s², so a pull does not jerk the item.
