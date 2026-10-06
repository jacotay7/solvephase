"""Phase diversity on an unknown extended scene (Gonsalves/Paxman).

A compact scene of Gaussian blobs is imaged in focus and with one wave peak-to-valley of
defocus; the wavefront and the object are recovered jointly.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from _common import QUICK, device, setup

import solvephase as sp

out = setup(__doc__)
wavelength = 0.6e-6
pupil = sp.Pupil.circular(64, 1.0)
truth = sp.random_aberration(pupil, 0.07 * wavelength, n_modes=20, start=4, seed=5)
n = 64
rng = np.random.default_rng(1)
# A compact scene (Gaussian blobs within 12 pixels of the centre): the
# Fourier-domain metric needs the blurred scene to stay inside the field.
yy, xx = np.mgrid[:n, :n] - (n - 1) / 2
scene = np.full((n, n), 1e-3)
for _ in range(10):
    cy, cx = rng.uniform(-12, 12, 2)
    width = rng.uniform(1.0, 3.0)
    scene += rng.uniform(0.3, 1.0) * np.exp(-0.5 * ((yy - cy) ** 2 + (xx - cx) ** 2) / width**2)
model = sp.FocalPlaneModel(
    pupil,
    wavelength,
    n,
    sampling=2.0,
    diversity=sp.zernike_diversity(pupil, 4, [0.0, wavelength / (2 * math.sqrt(3))]),
    device=device(),
)
psfs = sp.to_numpy(model.images(truth))
kernel = np.fft.rfft2(np.fft.ifftshift(psfs, axes=(-2, -1)))
blurred = np.fft.irfft2(np.fft.rfft2(scene) * kernel, s=(n, n))
images = rng.poisson(np.maximum(blurred, 0) * 1000 / blurred[0].mean()).astype(float)
result = sp.phase_diversity(model, images, basis=20 if QUICK else 30)
err = sp.wavefront_error(sp.to_numpy(result.opd), truth, pupil, remove="tiptilt")
print(
    f"aberration {sp.rms(truth, pupil, 'tiptilt') * 1e9:.1f} nm -> error {err * 1e9:.2f} nm ({result.elapsed:.2f} s)"
)
obj = sp.to_numpy(result.extra["object"])
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for a, img, title in [
    (ax[0], scene, "true scene"),
    (ax[1], images[0], "in-focus image"),
    (ax[2], obj, "recovered object"),
]:
    a.imshow(img, origin="lower", cmap="gray")
    a.set_title(title)
fig.tight_layout()
fig.savefig(out / "04_extended_scene_diversity.png", dpi=110)
