"""NCPA calibration on a VLT-like pupil from two detector frames.

Simulates the non-common-path aberration of an AO instrument, exposes two
images (in focus and defocused) on a realistic getframes camera when it is
installed (otherwise ideal Poisson + read noise), then retrieves the wavefront
with :func:`solvephase.retrieve` and plots data, model and error.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from _common import QUICK, device, setup

import solvephase as sp

out = setup(__doc__)
wavelength = 1.65e-6
pupil = sp.Pupil.vlt(64 if QUICK else 128)
ncpa = sp.random_aberration(pupil, 90e-9, n_modes=40, start=4, seed=7)
defocus = 0.4e-6  # RMS metres on the second image
model = sp.FocalPlaneModel(
    pupil, wavelength, 64, sampling=2.0, diversity=sp.zernike_diversity(pupil, 4, [0.0, defocus])
)
try:
    from solvephase.interop import expose

    frames = expose(
        sp.to_numpy(model.images(ncpa)), "andor_ikon_m934", photons=4e5, background=30, seed=1
    )
    images, read_noise, weights = frames.electrons, frames.read_noise, frames.weights
    source = "getframes Andor iKon-M"
except ImportError:
    images = sp.simulate_images(model, ncpa, photons=4e5, background=30, read_noise=3.0, seed=1)
    read_noise, weights, source = 3.0, None, "ideal Poisson + read noise"

result = sp.retrieve(
    images,
    pupil,
    wavelength,
    sampling=2.0,
    diversity=[0.0, defocus],
    basis=45,
    read_noise=read_noise,
    weights=weights,
    device=device(),
)
result = result.to_numpy()
err = sp.wavefront_error(result.opd, ncpa, pupil, remove="tiptilt")
print(result.summary())
print(
    f"data: {source}; NCPA {sp.rms(ncpa, pupil, 'tiptilt') * 1e9:.1f} nm RMS, residual {err * 1e9:.2f} nm RMS"
)

fig, ax = plt.subplots(2, 3, figsize=(12, 7.5))
mask = np.where(pupil.mask, 1.0, np.nan)
vmax = np.nanmax(np.abs(ncpa * mask)) * 1e9
error_map = sp.remove_modes(result.opd - ncpa, pupil, "tiptilt") * mask * 1e9
emax = np.nanmax(np.abs(error_map))
for a, img, title, lim in [
    (ax[0, 0], ncpa * mask * 1e9, "true NCPA [nm]", vmax),
    (ax[0, 1], result.opd * mask * 1e9, "retrieved [nm]", vmax),
    (ax[0, 2], error_map, f"error [nm], {err * 1e9:.2f} nm RMS", emax),
]:
    im = a.imshow(img, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    a.set_title(title)
    fig.colorbar(im, ax=a, fraction=0.046)
for k in range(2):
    a = ax[1, k]
    a.imshow(np.log10(np.maximum(images[k], 1)), origin="lower", cmap="magma")
    a.set_title(f"frame {k} (log10 e-)")
resid = (images - result.model_images) / np.sqrt(np.maximum(result.model_images, 1) + read_noise**2)
im = ax[1, 2].imshow(resid[1], origin="lower", cmap="RdBu_r", vmin=-4, vmax=4)
ax[1, 2].set_title("normalized residual, frame 1")
fig.colorbar(im, ax=ax[1, 2], fraction=0.046)
for a in ax.ravel():
    a.set_xticks([]), a.set_yticks([])
fig.tight_layout()
fig.savefig(out / "01_ncpa_calibration.png", dpi=110)
print(f"saved {out / '01_ncpa_calibration.png'}")
