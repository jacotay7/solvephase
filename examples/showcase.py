"""Render the README showcase animation (examples/solvephase_showcase.webp).

A VLT-like pupil with 120 nm RMS of non-common-path aberration is imaged in
focus and with defocus on a getframes camera (or ideal Poisson noise). The
animation follows the default solve: Levenberg-Marquardt on the amplitude
metric from a flat start (capture), then Poisson maximum-likelihood steps,
showing data, model, retrieved wavefront and error at each step.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

import solvephase as sp

OUT = Path(__file__).resolve().parent / "solvephase_showcase.webp"
wavelength = 1.65e-6
pupil = sp.Pupil.vlt(128)
truth = sp.random_aberration(pupil, 120e-9, n_modes=60, start=4, power=0.9, seed=11)
defocus = 0.45e-6
model = sp.FocalPlaneModel(
    pupil, wavelength, 64, sampling=2.0, diversity=sp.zernike_diversity(pupil, 4, [0.0, defocus])
)
try:
    from solvephase.interop import expose

    frames = expose(
        sp.to_numpy(model.images(truth)), "andor_ikon_m934", photons=3e5, background=20, seed=3
    )
    images, read_noise, weights = frames.electrons, frames.read_noise, frames.weights
except ImportError:  # pragma: no cover
    images = sp.simulate_images(model, truth, photons=3e5, background=20, read_noise=3, seed=3)
    read_noise, weights = 3.0, None

stages: list[tuple[str, np.ndarray]] = [("flat start", np.zeros(pupil.shape))]
basis = sp.Basis.zernike(pupil, 66)
common = {"basis": basis, "read_noise": read_noise, "weights": weights, "fit_background": True}
capture = sp.FocalPlaneProblem(model, images, loss="amplitude", **common)
polish = sp.FocalPlaneProblem(model, images, loss="poisson", **common)


def recorder(problem: sp.FocalPlaneProblem, name: str):
    def callback(it: int, x, f: float) -> None:
        phase = problem._phase_map(x[problem.layout.coeffs])
        stages.append(
            (f"{name}, step {it}", np.asarray(sp.to_numpy(phase)) * wavelength / (2 * math.pi))
        )

    return callback


explored = sp.solve(capture, method="lm", callback=recorder(capture, "capture (amplitude LM)"))
result = sp.solve(
    polish, method="lm", start=explored, callback=recorder(polish, "Poisson maximum likelihood")
)
background = float(np.median(result.background))
final_err = sp.wavefront_error(result.opd, truth, pupil, remove="tiptilt")
print(f"final error {final_err * 1e9:.2f} nm RMS")

mask = np.where(pupil.mask, 1.0, np.nan)
vmax = np.nanmax(np.abs(truth * mask)) * 1e9
fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.6))
fig.patch.set_facecolor("white")
data = np.log10(np.maximum(images[1], 1.0))
axes[0].imshow(data, origin="lower", cmap="magma")
axes[0].set_title("defocused frame (log e-)")
model_im = axes[1].imshow(data, origin="lower", cmap="magma")
axes[1].set_title("model")
opd_im = axes[2].imshow(truth * mask * 1e9, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
axes[2].set_title("retrieved OPD [nm]")
err_im = axes[3].imshow(
    truth * mask * 1e9, origin="lower", cmap="RdBu_r", vmin=-vmax / 4, vmax=vmax / 4
)
title = fig.suptitle("")
for a in axes:
    a.set_xticks([])
    a.set_yticks([])
fig.tight_layout(rect=(0, 0, 1, 0.88))


def frame(i: int):
    label, opd = stages[min(i, len(stages) - 1)]
    m = sp.to_numpy(model.images(opd))[1]
    scale = (images[1].sum() - background * m.size) / max(m.sum(), 1e-30)
    model_im.set_data(np.log10(np.maximum(m * scale + background, 1.0)))
    opd_im.set_data(opd * mask * 1e9)
    err_map = sp.remove_modes(opd - truth, pupil, "tiptilt") * mask * 1e9
    err_im.set_data(err_map)
    err = sp.wavefront_error(opd, truth, pupil, remove="tiptilt")
    axes[3].set_title(f"error: {err * 1e9:.1f} nm RMS")
    title.set_text(
        f"solvephase: {label}  |  NCPA {sp.rms(truth, pupil, 'tiptilt') * 1e9:.0f} nm RMS"
    )
    return model_im, opd_im, err_im, title


frames_idx = [0, 0] + list(range(len(stages))) + [len(stages) - 1] * 5
anim = FuncAnimation(fig, frame, frames=frames_idx, blit=False)
anim.save(OUT, writer=PillowWriter(fps=2), dpi=60)
print(f"saved {OUT} ({OUT.stat().st_size / 1e3:.0f} kB, {len(stages)} stages)")
