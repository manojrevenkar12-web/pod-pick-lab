"""Write the dashboard's unit-test fixtures from the committed results (a few orders per set)."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
r = json.loads((ROOT / "reports" / "results.json").read_text())
orders = json.loads((ROOT / "reports" / "orders.json").read_text())
keep = {}
for o in orders:
    keep.setdefault((o["family"], o["seed"]), []).append(o)
small = [o for i, (_, os) in enumerate(sorted(keep.items())) if i % 25 == 0 for o in os]
out = ROOT / "web" / "src" / "app"
(out / "fixture.json").write_text(json.dumps(r))
(out / "orders.fixture.json").write_text(json.dumps(small))
print(f"fixture: {len(small)} order rows")
