"""podpick <step> [--profile quick|full]"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    from .evaluate.pipeline import STEPS, run_all

    ap = argparse.ArgumentParser(prog="podpick", description=__doc__)
    ap.add_argument("step", choices=[*STEPS, "all", "heldout", "serve", "build-native"])
    ap.add_argument("--profile", default="full", choices=["quick", "full"])
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args(argv)
    if a.step == "all":
        return 0 if run_all(a.profile) else 1
    if a.step == "heldout":
        from .evaluate.heldout import run_heldout

        run_heldout(a.profile)
        return 0
    if a.step == "serve":
        import uvicorn

        uvicorn.run("podpick.service.app:app", host="0.0.0.0", port=a.port)
        return 0
    if a.step == "build-native":
        from .control.native import build

        print(build())
        return 0
    out = STEPS[a.step](a.profile)
    if a.step == "gate":
        return 0 if out else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
