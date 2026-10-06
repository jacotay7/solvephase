"""Render Markdown tables from benchmark artifacts.

Usage::

    python benchmarks/render_table.py benchmarks/artifacts/<suite>.json [--compare <compare>.json]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any


def _fmt(value: float, unit: str) -> str:
    if unit == "ms":
        ms = value * 1e3
        return f"{ms:.1f} ms" if ms < 100 else f"{ms:.0f} ms"
    if unit == "s":
        return f"{value:.2f} s"
    if unit == "rate":
        return f"{value:,.0f}"
    return f"{value:.3g}"


def _key(row: dict[str, Any], fields: list[str]) -> tuple[Any, ...]:
    return tuple(row.get(f) for f in fields)


SPECS: list[tuple[str, str, list[str], str, str]] = [
    ("focal_lm", "Focal-plane LM time to solution (36 modes, 2 images)", ["size"], "seconds", "s"),
    ("focal_gradient", "Objective + gradient evaluation (zonal, 2 images)", ["size", "wavelengths"], "seconds", "ms"),
    ("gerchberg_saxton", "Gerchberg-Saxton/Misell iterations per second (3 images)", ["size"], "iterations_per_s", "rate"),
    ("retrieve_auto", "`retrieve()` end to end (robust default)", ["size"], "seconds", "s"),
    ("cdi", "CDI iterations per second (total over starts)", ["size", "algorithm", "starts"], "iterations_per_s", "rate"),
    ("tie", "TIE solve (3 planes, non-uniform intensity)", ["size", "method"], "seconds", "ms"),
    ("lift", "LIFT estimate (10 modes, one image)", ["size"], "seconds", "ms"),
    ("fast_furious", "Fast & Furious step latency", ["size"], "seconds", "ms"),
    ("phase_diversity", "Extended-object phase diversity solve (20 modes)", ["size"], "seconds", "s"),
    ("wirtinger", "Coded-diffraction retrieval (6 masks) to 1e-6", ["size", "method"], "seconds", "s"),
    ("unwrap", "Least-squares phase unwrapping (spidered pupil)", ["size"], "seconds", "ms"),
]


def render_suite(artifact: dict[str, Any]) -> str:
    env = artifact["environment"]
    lines = [
        f"Hardware: {env.get('cpu', env.get('processor'))} ({env.get('cpu_count')} threads)"
        + (f"; {env['gpu']} (CuPy {env.get('cupy')})" if "gpu" in env else "")
        + f"; NumPy {env.get('numpy')}, SciPy {env.get('scipy')}; solvephase {env.get('solvephase')}.",
        "",
    ]
    rows = artifact["results"]
    for case, title, fields, metric, unit in SPECS:
        selected = [r for r in rows if r.get("case") == case and metric in r]
        if not selected:
            continue
        table: dict[tuple[Any, ...], dict[str, str]] = defaultdict(dict)
        devices = []
        for r in selected:
            dev = f"{r['device'].upper()} ({r['precision']})"
            if dev not in devices:
                devices.append(dev)
            table[_key(r, fields)][dev] = _fmt(r[metric], unit)
        lines.append(f"**{title}**")
        lines.append("")
        lines.append("| " + " | ".join(fields + devices) + " |")
        lines.append("|" + "---|" * (len(fields) + len(devices)))
        for key, values in table.items():
            cells = [str(k) for k in key] + [values.get(d, "–") for d in devices]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def render_compare(artifact: dict[str, Any]) -> str:
    lines = []
    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in artifact["rows"]:
        by_problem[r["problem"]].append(r)
    for problem, rows in by_problem.items():
        lines.append(f"**{problem}**")
        lines.append("")
        if "error_nm" in rows[0]:
            lines.append("| method | time | wavefront error |")
            lines.append("|---|---|---|")
            for r in rows:
                lines.append(f"| {r['method']} | {r['seconds']:.2f} s | {r['error_nm']:.2f} nm |")
        else:
            base = rows[0]["iterations_per_s"]
            lines.append("| method | iterations/s | speed-up |")
            lines.append("|---|---|---|")
            for r in rows:
                lines.append(f"| {r['method']} | {r['iterations_per_s']:,.0f} | {r['iterations_per_s'] / base:.1f}x |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", nargs="?")
    parser.add_argument("--compare", default=None)
    args = parser.parse_args()
    if args.suite:
        with open(args.suite) as handle:
            print(render_suite(json.load(handle)))
    if args.compare:
        with open(args.compare) as handle:
            print(render_compare(json.load(handle)))


if __name__ == "__main__":
    main()
