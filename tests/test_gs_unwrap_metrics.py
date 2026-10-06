from __future__ import annotations

import numpy as np
import pytest

import solvephase as sp
from conftest import devices
from solvephase import Basis, FocalPlaneModel, Pupil, zernike_diversity
from solvephase.unwrap import unwrap_phase, wrap

LAM = 1e-6


@pytest.mark.parametrize(
    "pupil",
    [Pupil.circular(96, 1.0, obscuration=0.3, spiders=4, spider_width=0.02), Pupil.keck(96)],
    ids=["spiders", "keck"],
)
@pytest.mark.parametrize("device", devices())
def test_unwrap_recovers_smooth_phase(pupil, device) -> None:
    z = Basis.zernike(pupil, 10)
    phase = z.synthesize(np.random.default_rng(0).standard_normal(10) * 2.0)
    be = sp.get_backend(device, "double")
    out = sp.to_numpy(unwrap_phase(be.asarray(wrap(phase)), pupil.mask, weights=pupil.amplitude))
    diff = (out - phase)[pupil.mask]
    assert np.ptp(phase[pupil.mask]) > 4 * np.pi
    assert np.std(diff) < 1e-6
    # The level is consistent with the wrapped data (multiple of 2 pi).
    assert abs(wrap(np.mean(diff))) < 1e-6


def test_unwrap_bridges_disconnected_regions() -> None:
    pupil = Pupil.circular(96, 1.0, obscuration=0.2, spiders=3, spider_width=0.04)
    y, x = pupil.coordinates()
    phase = 40.0 * x + 25.0 * y  # steep but sampled tilt crossing every vane
    out = unwrap_phase(wrap(phase), pupil.mask, weights=pupil.amplitude)
    diff = (out - phase)[pupil.mask]
    assert np.std(diff) < 1e-6


def test_wrap_range() -> None:
    v = wrap(np.linspace(-20, 20, 1001))
    assert v.min() >= -np.pi - 1e-12 and v.max() <= np.pi + 1e-12


def test_gerchberg_saxton_converges_on_diversity_data() -> None:
    pupil = Pupil.circular(64, 1.0, obscuration=0.15)
    opd = sp.random_aberration(pupil, 0.08 * LAM, n_modes=20, seed=1)
    div = zernike_diversity(pupil, 4, [-0.3 * LAM, 0.0, 0.3 * LAM])
    model = FocalPlaneModel(pupil, LAM, 64, sampling=2.0, diversity=div)
    images = sp.simulate_images(model, opd, noise=False)
    res = sp.gerchberg_saxton(model, images, iterations=300)
    assert res.method == "misell"
    assert res.history[-1] < 1e-3 * res.history[0]
    assert sp.wavefront_error(res.opd, opd, pupil, remove="tiptilt") < 0.1 * sp.rms(
        opd, pupil, "tiptilt"
    )
    modal = sp.gerchberg_saxton(model, images, iterations=300, basis=20)
    assert modal.coefficients is not None and modal.coefficients.shape == (20,)


def test_gerchberg_saxton_mft_path_and_broadband_warning() -> None:
    pupil = Pupil.circular(32, 1.0)
    opd = sp.random_aberration(pupil, 0.05 * LAM, n_modes=10, seed=2)
    div = zernike_diversity(pupil, 4, [0.0, 0.3 * LAM])
    model = FocalPlaneModel(pupil, [0.9 * LAM, 1.1 * LAM], 32, sampling=2.3, diversity=div)
    images = sp.simulate_images(model, opd, noise=False)
    with pytest.warns(UserWarning, match="bandwidth"):
        res = sp.gerchberg_saxton(model, images, iterations=100)
    assert res.history[-1] < res.history[0]


def test_metrics() -> None:
    pupil = Pupil.circular(64, 1.0)
    _, x = pupil.coordinates()
    tilt = np.where(pupil.mask, 1e-7 * x / 0.5 + 3e-8, 0.0)
    assert sp.rms(tilt, pupil, "tiptilt") < 1e-20
    assert sp.rms(tilt, pupil, "piston") > 1e-8
    assert np.isclose(sp.strehl_from_rms(0.0, 1e-6), 1.0)
    assert np.isclose(sp.strehl_from_rms(1e-6 / (2 * np.pi), 1e-6), np.exp(-1))
    z = Basis.zernike(pupil, 10).synthesize(np.linspace(1, 2, 10) * 1e-8)
    twin = -z[::-1, ::-1]
    assert sp.wavefront_error(twin, z, pupil, allow_twin=True) < 1e-15
    assert sp.wavefront_error(twin, z, pupil) > 1e-9
