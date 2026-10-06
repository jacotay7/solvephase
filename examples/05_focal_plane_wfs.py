"""Focal-plane wavefront sensing: LIFT and a Fast & Furious closed loop.

LIFT estimates low-order modes from one astigmatic image and is compared with
its Cramer-Rao bound; Fast & Furious drives a static aberration down with a
DM in closed loop.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from _common import QUICK, device, setup

import solvephase as sp

out = setup(__doc__)
wavelength = 1.6e-6
pupil = sp.Pupil.circular(32, 8.0, obscuration=0.1)
sensor = sp.LIFT(pupil, wavelength, 32, sampling=2.0, n_modes=10, read_noise=1.0, device=device())
truth = sp.random_aberration(pupil, 0.08 * wavelength, n_modes=10, start=2, seed=4)
image = sp.simulate_images(sensor.model, truth, photons=1e5, read_noise=1.0, seed=5)[0]
estimate = sensor.estimate(image)
bound = np.sqrt(np.diag(sensor.crlb(sensor.basis.fit(truth), photons=1e5)))[: sensor.basis.n_modes]
print("LIFT estimate vs truth [nm], with the Cramer-Rao bound:")
for label, est, true, sigma in zip(
    sensor.basis.labels, estimate.coefficients, sensor.basis.fit(truth), bound
):
    print(f"  {label:4s} {est * 1e9:+8.2f} {true * 1e9:+8.2f}  +/- {sigma * 1e9:.2f}")

ff_pupil = sp.Pupil.circular(64, 1.0)
ff = sp.FastAndFurious(ff_pupil, wavelength, 64, sampling=2.0, device=device())
ncpa = sp.random_aberration(ff_pupil, 0.1 * wavelength, n_modes=20, start=2, seed=6)
loop = sp.simulate_closed_loop(ff, ncpa, 10 if QUICK else 30, gain=0.5, photons=1e6, seed=7)
print(
    f"F&F residual: {loop.residual_rms[0] * 1e9:.1f} nm -> {loop.residual_rms[-1] * 1e9:.1f} nm in {loop.n_iter} steps"
)
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.semilogy(np.asarray(loop.residual_rms) * 1e9, "o-")
ax.set_xlabel("iteration")
ax.set_ylabel("residual RMS [nm]")
ax.set_title("Fast & Furious closed loop")
fig.tight_layout()
fig.savefig(out / "05_fast_and_furious.png", dpi=110)
