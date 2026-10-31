"""Paths and YAML settings. Business and research settings live in config/, never in code."""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(os.environ.get("PODPICK_ROOT", Path(__file__).resolve().parents[2]))
CONFIG = ROOT / "config"


@cache
def load(name: str) -> dict[str, Any]:
    """config/<name>.yaml as a dict (cached; treat as read-only)."""
    with open(CONFIG / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def robot() -> dict[str, Any]:
    return load("robot")


def policy() -> dict[str, Any]:
    return load("policy")
