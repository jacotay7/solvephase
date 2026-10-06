"""Image-based phasing of a Keck-like segmented primary.

Each of the 36 segments gets a random piston, tip and tilt. Two defocused
images are inverted in the segment piston/tip/tilt basis (108 parameters)
with Levenberg-Marquardt.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from _common import QUICK, device, setup

import solvephase as sp

out = setup(__doc__)
wavelength = 2.2e-6
pupil = sp.Pupil.keck(96 if QUICK else 128)
basis = sp.Basis.segments(pupil)
rng = np.random.default_rng(2)
coeffs = rng.normal(0.0, 40e-9, basis.n_modes)  # 40 nm RMS per term
truth = basis.synthesize(coeffs)
model = sp.FocalPlaneModel(
    pupil,
    wavelength,
    96,
    sampling=2.0,
    diversity=sp.zernike_diversity(pupil, 4, [-0.5e-6, 0.5e-6]),
    device=device(),
)
images = sp.simulate_images(model, truth, photons=5e6, seed=3)
result = sp.solve(sp.FocalPlaneProblem(model, images, basis=basis, loss="poisson"), method="lm")
err = sp.wavefront_error(sp.to_numpy(result.opd), truth, pupil, remove="tiptilt")
print(
    f"segment errors: {sp.rms(truth, pupil, 'tiptilt') * 1e9:.1f} nm RMS -> residual {err * 1e9:.2f} nm RMS "
    f"in {result.n_iter} iterations ({result.elapsed:.2f} s)"
)

mask = np.where(pupil.mask, 1.0, np.nan)
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for a, img, title in [
    (ax[0], truth * mask * 1e9, "segment errors [nm]"),
    (ax[1], sp.to_numpy(result.opd) * mask * 1e9, "retrieved [nm]"),
    (
        ax[2],
        sp.remove_modes(sp.to_numpy(result.opd) - truth, pupil, "tiptilt") * mask * 1e9,
        "residual [nm]",
    ),
]:
    im = a.imshow(img, origin="lower", cmap="RdBu_r")
    a.set_title(title)
    fig.colorbar(im, ax=a, fraction=0.046)
fig.tight_layout()
fig.savefig(out / "03_segment_phasing.png", dpi=110)
