"""Coherent diffraction imaging with shrinkwrap and batched multi-start.

An irregular object is recovered from its oversampled, noisy far-field
intensity, starting from an autocorrelation support. Sixteen random starts
run as one batched FFT per iteration; the best is aligned to the truth.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from _common import QUICK, device, setup
from scipy.ndimage import gaussian_filter

import solvephase as sp

out = setup(__doc__)
rng = np.random.default_rng(0)
n = 64
yy, xx = np.mgrid[:n, :n] - (n - 1) / 2
outline = np.hypot(yy, xx) < 22 * (1 + 0.25 * np.cos(3 * np.arctan2(yy, xx)))
obj = np.where(outline, 0.3 + gaussian_filter(rng.uniform(size=(n, n)), 1.5), 0.0)
data = sp.simulate_cdi(obj, oversampling=2, photons=1e9, seed=1)
support = sp.autocorrelation_support(data.intensity)
result = sp.cdi(
    np.sqrt(data.intensity),
    support,
    schedule="hio:300,er:50" if QUICK else "hio:800,er:200",
    starts=4 if QUICK else 16,
    constraint="positive",
    shrinkwrap={"every": 20, "sigma": 3.0, "sigma_min": 1.0, "threshold": 0.1},
    seed=2,
    device=device(),
)
aligned, error = sp.align_object(result.object, data.object)
print(
    f"best start {result.best_start}, aligned error {error:.3e}, {result.n_iter} iterations, {result.elapsed:.2f} s"
)
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(np.log10(sp.to_numpy(data.intensity) + 1), cmap="magma")
ax[0].set_title("diffraction intensity (log10)")
ax[1].imshow(np.abs(sp.to_numpy(data.object)), cmap="gray")
ax[1].set_title("true object")
ax[2].imshow(np.abs(sp.to_numpy(aligned)), cmap="gray")
ax[2].set_title(f"recovered (error {error:.1e})")
fig.tight_layout()
fig.savefig(out / "06_cdi_multistart.png", dpi=110)
