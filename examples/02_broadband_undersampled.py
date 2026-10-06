"""Broadband, undersampled Hubble-like images with pixel integration.

Five wavelengths across H band, 130 mas pixels (about 1 pixel per lambda/D,
half of Nyquist, like Hubble WFC3/IR),
pixel integration by 3x oversampling, and three diversity planes. Compares the
monochromatic approximation against the exact broadband model.
"""

from __future__ import annotations

import numpy as np
from _common import QUICK, device, setup

import solvephase as sp

out = setup(__doc__)
pupil = sp.Pupil.hst(64)
band = np.linspace(1.5e-6, 1.8e-6, 3 if QUICK else 5)
pixel = 0.13 / 206265.0
truth = sp.random_aberration(pupil, 60e-9, n_modes=25, start=4, seed=3)
div = sp.zernike_diversity(pupil, 4, [-0.4e-6, 0.0, 0.4e-6])
broad = sp.FocalPlaneModel(
    pupil, band, 48, pixel_scale=pixel, oversample=3, diversity=div, device=device()
)
images = sp.simulate_images(broad, truth, photons=2e6, background=5, seed=4)
print(broad)
print(f"Nyquist sampled at the shortest wavelength: {broad.nyquist_sampled}")

basis = sp.Basis.zernike(pupil, 28)
exact = sp.solve(
    sp.FocalPlaneProblem(broad, images, basis=basis, loss="poisson", fit_background=True)
)
mono_model = sp.FocalPlaneModel(
    pupil, band.mean(), 48, pixel_scale=pixel, oversample=3, diversity=div, device=device()
)
mono = sp.solve(
    sp.FocalPlaneProblem(mono_model, images, basis=basis, loss="poisson", fit_background=True)
)
for name, res in [("broadband model", exact), ("monochromatic model", mono)]:
    err = sp.wavefront_error(sp.to_numpy(res.opd), truth, pupil, remove="tiptilt")
    print(
        f"{name:20s}: {err * 1e9:6.2f} nm RMS error ({res.n_iter} LM iterations, {res.elapsed:.2f} s)"
    )
