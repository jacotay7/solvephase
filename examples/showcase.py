"""Render the README showcase (examples/solvephase_showcase.webp).

A race between every focal-plane method on the same wavefront (a VLT-like
pupil, 0.1 waves RMS of aberration, 10^6 photons per image), each fed the data
it needs, plus a gallery of the other measurement models. All panels run on
one wall clock (logarithmic), so the animation shows both how fast each
method is and how accurate it ends up. Timings are rescaled from an
un-instrumented run on the CPU; see benchmarks/methods.py.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "benchmarks"))
import methods  # noqa: E402

OUT = HERE / "solvephase_showcase.webp"

SHORT = {
    "misell": ("Gerchberg-Saxton / Misell", "2 images"),
    "lm": ("Max. likelihood, modal (LM)", "2 images"),
    "lbfgs": ("Max. likelihood, zonal (L-BFGS)", "2 images, flat start"),
    "retrieve": ("retrieve(): robust default", "2 images"),
    "pd": ("Phase diversity", "unknown extended scene"),
    "lift": ("LIFT", "1 astigmatic image, 10 modes"),
    "ff": ("Fast & Furious", "1 image per step + DM"),
}
COLORS = {
    "misell": "#1f77b4",
    "lm": "#d62728",
    "lbfgs": "#9467bd",
    "retrieve": "#2ca02c",
    "pd": "#ff7f0e",
    "lift": "#8c564b",
    "ff": "#e377c2",
}


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    name = line.split(":", 1)[1].strip()
                    return name.replace("(R)", "").replace("(TM)", "").split(" CPU")[0]
    except OSError:
        pass
    return platform.processor() or "CPU"


def blank(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    traces, gallery = methods.run_all(quick=True, record=True, gpu=False)
    focal = [t for t in traces if t.family == "focal-plane"]
    others = {t.key: t for t in traces if t.family != "focal-plane"}
    t_max = max(max(t.times) if t.times else t.seconds for t in traces) * 1.15
    clock = np.concatenate([[0.0], np.geomspace(1e-3, t_max, 46), [t_max] * 8])

    truth = gallery["truth"] * 1e9
    vmax = float(np.nanmax(np.abs(truth)))
    fig = plt.figure(figsize=(16, 9.2))
    fig.patch.set_facecolor("white")
    gs = GridSpec(
        3, 6, figure=fig, hspace=0.45, wspace=0.12, left=0.01, right=0.95, top=0.855, bottom=0.04
    )
    fig.suptitle(
        "solvephase: one wavefront, every focal-plane method, one clock", fontsize=17, weight="bold"
    )
    clock_text = fig.text(
        0.5, 0.918, "", ha="center", fontsize=13, color="#444", family="monospace"
    )

    ax_truth = fig.add_subplot(gs[0, 0])
    ax_truth.imshow(truth, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax_truth.set_title("true wavefront\n0.1 λ RMS, H band, VLT pupil", fontsize=10)
    blank(ax_truth)

    slots = [gs[0, 1], gs[0, 2], gs[0, 3], gs[1, 0], gs[1, 1], gs[1, 2], gs[1, 3]]
    panels = {}
    for trace, slot in zip(focal, slots):
        ax = fig.add_subplot(slot)
        im = ax.imshow(np.zeros_like(truth), origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        name, data = SHORT[trace.key]
        ax.set_title(f"{name}\n{data}", fontsize=10, color=COLORS[trace.key])
        label = ax.text(0.5, -0.06, "", transform=ax.transAxes, ha="center", va="top", fontsize=10)
        blank(ax)
        for spine in ax.spines.values():
            spine.set_edgecolor(COLORS[trace.key])
            spine.set_linewidth(2)
        panels[trace.key] = (im, label)

    race = fig.add_subplot(gs[0:2, 4:6])
    race.set_xscale("log")
    race.set_yscale("log")
    race.set_xlim(1e-3, t_max)
    race.set_ylim(3e-3, 1.5)
    race.set_xlabel("wall-clock time [s]")
    race.set_ylabel("relative wavefront error")
    race.yaxis.set_label_position("right")
    race.yaxis.tick_right()
    race.grid(True, which="both", alpha=0.25)
    race.set_title("error vs time (same wavefront)", fontsize=12)
    lines = {}
    for trace in focal:
        (line,) = race.plot([], [], "-", color=COLORS[trace.key], lw=2.2, label=SHORT[trace.key][0])
        (dot,) = race.plot([], [], "o", color=COLORS[trace.key], ms=7)
        lines[trace.key] = (line, dot)
    race.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
    cursor = race.axvline(1e-3, color="k", lw=1, alpha=0.5)

    gal = []

    def gallery_panel(slot, image, title, cmap, key=None, log=False):
        ax = fig.add_subplot(slot)
        data = np.log10(np.maximum(image, image.max() * 1e-6)) if log else image
        im = ax.imshow(data, cmap=cmap, origin="lower")
        ax.set_title(title, fontsize=10)
        blank(ax)
        text = ax.text(0.5, -0.06, "", transform=ax.transAxes, ha="center", va="top", fontsize=10)
        gal.append((key, im, text))

    g = gallery
    gallery_panel(
        gs[2, 0],
        g["cdi"]["intensity"],
        "CDI: diffraction intensity\n1 pattern (log)",
        "magma",
        log=True,
    )
    gallery_panel(
        gs[2, 1], g["cdi"]["object"], "CDI: HIO→ER + shrinkwrap\n8 batched starts", "gray", "cdi"
    )
    gallery_panel(
        gs[2, 2],
        g["generic"]["phase"],
        "coded diffraction: L-BFGS\n6 masks (phase)",
        "twilight",
        "generic",
    )
    gallery_panel(gs[2, 3], g["tie"]["intensity"], "TIE: intensity at +dz\n3 planes", "gray")
    gallery_panel(
        gs[2, 4], g["tie"]["phase"], "TIE: recovered phase\nnon-uniform intensity", "viridis", "tie"
    )
    info = fig.add_subplot(gs[2, 5])
    info.axis("off")
    info.text(
        0.0,
        0.98,
        f"CPU: {cpu_name()}\none process, double precision\n\n"
        "Focal data: 10⁶ photons/image,\n2 e⁻ read noise, background;\n"
        "extended scene at 1000 photons/pixel.\n\n"
        "Errors: RMS with piston and tip/tilt\nremoved, relative to the aberration.\n\n"
        "github.com/jacotay7/solvephase",
        va="top",
        fontsize=9.5,
        color="#333",
    )

    def at(trace: methods.Trace, t: float) -> int | None:
        idx = [i for i, ti in enumerate(trace.times) if ti <= t]
        return idx[-1] if idx else None

    def frame(i: int) -> list:
        t = clock[i]
        clock_text.set_text(f"t = {t * 1e3:7.1f} ms")
        cursor.set_xdata([max(t, 1e-3)] * 2)
        for trace in focal:
            im, label = panels[trace.key]
            line, dot = lines[trace.key]
            k = at(trace, t)
            if k is None:
                im.set_data(np.zeros_like(truth))
                label.set_text("running…")
                label.set_color("#888")
                line.set_data([], [])
                dot.set_data([], [])
                continue
            im.set_data(trace.frames[k] * 1e9)
            done = t >= trace.times[-1]
            label.set_text(
                f"{'✓ ' if done else ''}error {trace.errors[k]:.1%}  ·  {trace.times[k] * 1e3:.0f} ms"
            )
            label.set_color("#1a7f37" if done else "#222")
            ts = np.maximum(np.asarray(trace.times[: k + 1]), 1e-3)
            es = np.maximum(np.asarray(trace.errors[: k + 1]), 3e-3)
            line.set_data(ts, es)
            dot.set_data([ts[-1]], [es[-1]])
        for key, im, text in gal:
            if key is None:
                continue
            tr = others[key]
            done = t >= tr.seconds
            im.set_alpha(1.0 if done else 0.12)
            text.set_text(
                f"✓ error {tr.error:.1e}  ·  {tr.seconds * 1e3:.0f} ms" if done else "running…"
            )
            text.set_color("#1a7f37" if done else "#888")
        return []

    anim = FuncAnimation(fig, frame, frames=len(clock), blit=False)
    anim.save(OUT, writer=PillowWriter(fps=6), dpi=62)
    print(f"saved {OUT} ({OUT.stat().st_size / 1e6:.2f} MB, {len(clock)} frames)")
    for trace in traces:
        print(f"{trace.key:9s} error {trace.error:.3g}  {trace.seconds * 1e3:.0f} ms")


if __name__ == "__main__":
    main()
