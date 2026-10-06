"""Coded-diffraction phase retrieval with the Wirtinger-flow family.

A complex image is measured through six random octanary phase masks
(Candes, Li & Soltanolkotabi 2015). Several algorithms from the PhasePack
family recover it from a spectral initialization; their convergence is
compared.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from _common import QUICK, device, setup

import solvephase as sp
from solvephase.algorithms.wirtinger import relative_error, wirtinger
from solvephase.operators import CodedDiffractionOperator

out = setup(__doc__)
n = 32 if QUICK else 64
backend = sp.get_backend(device())
operator = CodedDiffractionOperator.random((n, n), 6, seed=1, backend=backend)
rng = np.random.default_rng(2)
yy, xx = np.mgrid[:n, :n] - n / 2
truth = np.exp(-(xx**2 + yy**2) / (2 * (n / 5) ** 2)) * np.exp(1j * 2 * np.pi * (xx + 0.5 * yy) / n)
truth = truth + 0.2 * (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
measurements = np.abs(sp.to_numpy(operator.forward(backend.asarray(truth, dtype="complex")))) ** 2

fig, ax = plt.subplots(figsize=(6, 4))
for method in ("wf", "twf", "taf", "raf", "lbfgs"):
    result = wirtinger(operator, measurements, method=method, iterations=300 if QUICK else 1000, tol=1e-8, seed=3)
    err = relative_error(result.x, truth)
    print(f"{method:6s} relative error {err:.2e} in {result.n_iter:4d} iterations, {result.elapsed:.2f} s")
    ax.semilogy(result.times, result.history, label=method)
ax.set_xlabel("time [s]")
ax.set_ylabel("objective")
ax.legend()
ax.set_title(f"coded diffraction, {n}x{n}, 6 masks")
fig.tight_layout()
fig.savefig(out / "07_generic_coded_diffraction.png", dpi=110)
