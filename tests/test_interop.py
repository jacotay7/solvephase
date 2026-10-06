from __future__ import annotations

import numpy as np
import pytest

import solvephase as sp
from solvephase import Pupil

pytestmark = pytest.mark.interop


def test_turbulence_opd_has_kolmogorov_amplitude() -> None:
    pytest.importorskip("pyturb")
    from solvephase.interop import turbulence_opd

    pupil = Pupil.circular(64, 2.0)
    screens = turbulence_opd(pupil, 0.2, outer_scale=np.inf, count=64, seed=1)
    lam = 500e-9
    var = np.mean([sp.rms(s, pupil, "piston") ** 2 for s in screens]) * (2 * np.pi / lam) ** 2
    # Noll (1976): piston-removed phase variance 1.0299 (D/r0)^(5/3) rad^2.
    theory = 1.0299 * (2.0 / 0.2) ** (5 / 3)
    assert 0.6 < var / theory < 1.3
    assert np.all(screens[:, ~pupil.mask] == 0)


def test_expose_converts_back_to_electrons_and_masks_saturation() -> None:
    pytest.importorskip("getframes")
    from solvephase.interop import expose

    pupil = Pupil.circular(32, 1.0)
    model = sp.FocalPlaneModel(pupil, 1e-6, 32, sampling=2.0)
    psf = sp.to_numpy(model.images())
    det = expose(psf, "andor_ikon_m934", photons=2e5, background=5.0, seed=3)
    assert det.n_saturated == 0
    resid = (det.electrons - det.truth) / np.sqrt(det.truth + det.read_noise**2)
    assert abs(resid.mean()) < 0.2 and 0.7 < resid.std() < 1.5
    bright = expose(psf, "andor_ikon_m934", photons=5e6, seed=3)
    assert bright.n_saturated > 0
    assert np.all(bright.weights[bright.adu >= 65535] == 0)


def test_retrieval_on_getframes_data() -> None:
    pytest.importorskip("getframes")
    from solvephase.interop import expose

    lam = 1.6e-6
    pupil = Pupil.vlt(48)
    opd = sp.random_aberration(pupil, 0.06 * lam, n_modes=20, seed=2)
    model = sp.FocalPlaneModel(
        pupil, lam, 48, sampling=2.0, diversity=sp.zernike_diversity(pupil, 4, [0.0, 0.3 * lam])
    )
    det = expose(
        sp.to_numpy(model.images(opd)), "andor_ikon_m934", photons=3e5, background=20, seed=5
    )
    res = sp.retrieve(
        det.electrons,
        pupil,
        lam,
        sampling=2.0,
        diversity=[0.0, 0.3 * lam],
        basis=20,
        read_noise=det.read_noise,
        weights=det.weights,
        method="lm",
    )
    assert sp.wavefront_error(res.opd, opd, pupil, remove="tiptilt") < 0.05 * sp.rms(
        opd, pupil, "tiptilt"
    )
