from __future__ import annotations

import numpy as np
import pytest

from conftest import devices
from solvephase import FocalPlaneModel, Pupil, to_numpy, zernike_diversity


@pytest.fixture(scope="module")
def pupil() -> Pupil:
    return Pupil.circular(32, 1.0, obscuration=0.2)


@pytest.mark.parametrize(
    "kwargs",
    [{"sampling": 2.0}, {"sampling": 2.37, "oversample": 2}, {"sampling": 1.3}],
)
def test_vjp_and_jvp_match_finite_differences(pupil, kwargs, rng) -> None:
    div = zernike_diversity(pupil, 4, [0.0, 0.3e-6, -0.3e-6])
    m = FocalPlaneModel(pupil, [1e-6, 1.1e-6, 1.25e-6], 24, diversity=div, **kwargs)
    phase = rng.standard_normal(pupil.shape) * pupil.mask * 0.5
    amp = pupil.amplitude * (1 + 0.1 * rng.standard_normal(pupil.shape))
    w = rng.standard_normal((3, 24, 24))

    def f(ph, a):
        return float(np.sum(w * m.forward(ph, a).images))

    state = m.forward(phase, amp)
    gp, ga = m.vjp(state, w, amplitude=True)
    d = rng.standard_normal(pupil.shape) * pupil.mask
    h = 1e-6
    fd_phase = (f(phase + h * d, amp) - f(phase - h * d, amp)) / (2 * h)
    fd_amp = (f(phase, amp + h * d) - f(phase, amp - h * d)) / (2 * h)
    assert np.isclose(np.sum(gp.sum(0) * d), fd_phase, rtol=1e-6)
    assert np.isclose(np.sum(ga * d), fd_amp, rtol=1e-6)
    jv = m.jvp(state, d[None])
    assert np.isclose(np.sum(w * jv[0]), fd_phase, rtol=1e-6)


def test_images_sum_to_one_when_the_window_is_large(pupil) -> None:
    m = FocalPlaneModel(pupil, 1e-6, 128, sampling=4.0)
    assert np.isclose(float(m.images().sum()), 1.0, rtol=1e-9)


def test_piston_does_not_change_images(pupil) -> None:
    m = FocalPlaneModel(pupil, 1e-6, 32, sampling=2.0)
    a = to_numpy(m.images(np.zeros(pupil.shape)))
    b = to_numpy(m.images(np.full(pupil.shape, 3.3e-7)))
    np.testing.assert_allclose(a, b, atol=1e-14)


def test_sampling_and_pixel_scale_are_equivalent(pupil) -> None:
    lam = 1.2e-6
    a = FocalPlaneModel(pupil, lam, 32, sampling=2.5)
    b = FocalPlaneModel(pupil, lam, 32, pixel_scale=lam / pupil.diameter / 2.5)
    np.testing.assert_allclose(to_numpy(a.images()), to_numpy(b.images()), atol=1e-15)
    assert np.isclose(a.sampling, 2.5) and a.nyquist_sampled


def test_oversampled_pixels_integrate_flux(pupil) -> None:
    fine = FocalPlaneModel(pupil, 1e-6, 64, sampling=4.0)
    binned = FocalPlaneModel(pupil, 1e-6, 32, sampling=2.0, oversample=2)
    a = to_numpy(fine.images())[0].reshape(32, 2, 32, 2).sum(axis=(1, 3))
    np.testing.assert_allclose(to_numpy(binned.images())[0], a, atol=1e-12)


def test_defocus_diversity_is_sign_symmetric_for_a_flat_wavefront(pupil) -> None:
    div = zernike_diversity(pupil, 4, [0.2e-6, -0.2e-6])
    imgs = to_numpy(FocalPlaneModel(pupil, 1e-6, 32, sampling=2.0, diversity=div).images())
    np.testing.assert_allclose(imgs[0], imgs[1], atol=1e-12)


def test_validation_errors(pupil) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        FocalPlaneModel(pupil, 1e-6, 32)
    with pytest.raises(ValueError, match="do not match"):
        FocalPlaneModel(pupil, 1e-6, 32, sampling=2.0, diversity=np.zeros((1, 8, 8)))
    with pytest.raises(ValueError, match="oversample"):
        FocalPlaneModel(pupil, 1e-6, 32, sampling=2.0, oversample=0)


@pytest.mark.parametrize("device", devices())
def test_device_parity(pupil, device, rng) -> None:
    opd = rng.standard_normal(pupil.shape) * pupil.mask * 5e-8
    ref = FocalPlaneModel(pupil, [1e-6, 1.2e-6], 24, sampling=2.2)
    dev = FocalPlaneModel(
        pupil, [1e-6, 1.2e-6], 24, sampling=2.2, device=device, precision="double"
    )
    np.testing.assert_allclose(to_numpy(dev.images(opd)), to_numpy(ref.images(opd)), atol=1e-12)


def test_amplitude_gradient_is_exact_where_amplitude_is_negative(pupil, rng) -> None:
    m = FocalPlaneModel(pupil, 1e-6, 24, sampling=2.0)
    phase = rng.standard_normal(pupil.shape) * pupil.mask * 0.3
    amp = pupil.amplitude - 0.3 * pupil.mask * (rng.uniform(size=pupil.shape) > 0.7)
    assert np.any(amp < 0)
    w = rng.standard_normal((1, 24, 24))
    _, ga = m.vjp(m.forward(phase, amp), w, amplitude=True)
    d = rng.standard_normal(pupil.shape) * pupil.mask
    h = 1e-6
    fd = (
        np.sum(w * m.forward(phase, amp + h * d).images)
        - np.sum(w * m.forward(phase, amp - h * d).images)
    ) / (2 * h)
    assert np.isclose(np.sum(ga * d), fd, rtol=1e-6)
