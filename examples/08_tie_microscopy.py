"""Transport-of-intensity phase imaging of a weakly absorbing sample.

Three intensities (at -dz, 0 and +dz) are simulated with the angular-spectrum
propagator and inverted with Teague's non-uniform-intensity TIE solver.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from _common import device, setup

import solvephase as sp

out = setup(__doc__)
n = 256  # the sample must vanish well inside the window (TIE boundary conditions)
pitch, wavelength = 1e-6, 0.55e-6
yy, xx = (np.mgrid[:n, :n] - n / 2) * pitch
phase = sum(
    a * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * s**2))
    for a, x0, y0, s in [
        (2.0, 20e-6, 10e-6, 12e-6),
        (-1.5, -25e-6, -15e-6, 9e-6),
        (1.0, 5e-6, -30e-6, 15e-6),
    ]
)
amplitude = 1.0 - 0.3 * np.exp(-((xx + 10e-6) ** 2 + (yy - 20e-6) ** 2) / (2 * (20e-6) ** 2))
dz = 2e-6
stack = sp.simulate_defocus_stack(amplitude * np.exp(1j * phase), [-dz, 0.0, dz], pitch, wavelength)
result = sp.tie(
    stack, [-dz, 0.0, dz], pitch=pitch, wavelength=wavelength, method="dct", device=device()
)
recovered = sp.to_numpy(result.phase)
err = np.std((recovered - recovered.mean()) - (phase - phase.mean())) / np.std(phase)
print(f"TIE relative phase error {err:.3e} ({result.elapsed * 1e3:.1f} ms)")
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for a, img, title in [
    (ax[0], phase, "true phase [rad]"),
    (ax[1], recovered, "TIE phase [rad]"),
    (ax[2], sp.to_numpy(stack)[2], "intensity at +dz"),
]:
    im = a.imshow(img, cmap="viridis")
    a.set_title(title)
    fig.colorbar(im, ax=a, fraction=0.046)
fig.tight_layout()
fig.savefig(out / "08_tie_microscopy.png", dpi=110)
