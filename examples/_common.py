"""Shared helpers for the example scripts (argument parsing, output, quick mode)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

QUICK = bool(os.environ.get("SOLVEPHASE_QUICK"))


def setup(description: str) -> Path:
    """Parse ``--output`` and return the output directory (created)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output", default="example-output", help="directory for figures")
    parser.add_argument("--device", default="cpu", help="cpu, gpu or auto")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    setup.device = args.device  # type: ignore[attr-defined]
    return out


def device() -> str:
    return getattr(setup, "device", "cpu")
