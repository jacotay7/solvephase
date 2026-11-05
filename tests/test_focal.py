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


# ---------------------------------------------------------------- blur
def test_gaussian_blur_matrix_limits() -> None:
    from solvephase.focal import gaussian_blur_matrix

    for n in (8, 33, 128):
        k, _ = gaussian_blur_matrix(n, 0.0)
        np.testing.assert_allclose(k, np.eye(n), atol=1e-12)
    k, _ = gaussian_blur_matrix(64, 2.5)
    d = np.arange(64) - 32.0
    sampled = np.exp(-(d**2) / (2 * 2.5**2)) / np.sqrt(2 * np.pi * 2.5**2)
    np.testing.assert_allclose(k[32], sampled, atol=1e-9)
    assert np.isclose(k[32].sum(), 1.0, atol=1e-9)
    np.testing.assert_allclose(k, k.T)
    with pytest.raises(ValueError, match="non-negative"):
        gaussian_blur_matrix(8, -1.0)


@pytest.mark.parametrize("sigma", [0.2, 0.7, 3.0])
def test_gaussian_blur_matrix_derivative(sigma) -> None:
    from solvephase.focal import gaussian_blur_matrix

    h = 1e-6
    _, dk = gaussian_blur_matrix(40, sigma)
    kp, _ = gaussian_blur_matrix(40, sigma + h)
    km, _ = gaussian_blur_matrix(40, sigma - h)
    np.testing.assert_allclose((kp - km) / (2 * h), dk, atol=1e-7 * np.abs(dk).max())


@pytest.mark.parametrize("oversample", [1, 2])
def test_blur_derivatives_and_adjoints(pupil, oversample, rng) -> None:
    div = zernike_diversity(pupil, 4, [0.0, 0.3e-6])
    blur = (0.4, 0.7)
    m = FocalPlaneModel(
        pupil, [1e-6, 1.2e-6], 20, sampling=2.0, diversity=div, oversample=oversample, blur=blur
    )
    phase = rng.standard_normal(pupil.shape) * pupil.mask * 0.4
    w = rng.standard_normal((2, 20, 20))
    state = m.forward(phase)
    assert state.blur == blur
    # Blur widths: analytic derivative against central differences.
    d_img = m.blur_jvp(state)
    h = 1e-6
    for axis in (0, 1):
        up, dn = list(blur), list(blur)
        up[axis] += h
        dn[axis] -= h
        fd = (m.forward(phase, blur=tuple(up)).images - m.forward(phase, blur=tuple(dn)).images) / (
            2 * h
        )
        np.testing.assert_allclose(d_img[axis], fd, atol=1e-6 * np.abs(fd).max())
    # Phase derivatives through the blur: vjp and jvp agree with differences.
    d = rng.standard_normal(pupil.shape) * pupil.mask
    gp, _ = m.vjp(state, w)
    fd = (
        np.sum(w * m.forward(phase + h * d).images) - np.sum(w * m.forward(phase - h * d).images)
    ) / (2 * h)
    assert np.isclose(np.sum(gp.sum(0) * d), fd, rtol=1e-6)
    assert np.isclose(np.sum(w * m.jvp(state, d[None])[0]), fd, rtol=1e-6)


def test_blur_spreads_light_and_conserves_flux(pupil) -> None:
    sharp = FocalPlaneModel(pupil, 1e-6, 64, sampling=3.0)
    soft = FocalPlaneModel(pupil, 1e-6, 64, sampling=3.0, blur=1.5)
    a = to_numpy(sharp.images())[0]
    b = to_numpy(soft.images())[0]
    assert b.max() < 0.6 * a.max()
    assert np.isclose(b.sum(), a.sum(), rtol=2e-3)  # light near the edge only
    # On a point image the blur is a Gaussian of the requested variance per axis.
    point = np.zeros((1, 64, 64))
    point[0, 32, 32] = 1.0
    spread = to_numpy(soft._blur(soft.backend.asarray(point), soft.blur))[0]
    y, x = np.indices(spread.shape) - 32.0
    assert np.isclose(spread.sum(), 1.0, atol=1e-9)
    assert np.isclose(np.sum(spread * x**2), 1.5**2, rtol=1e-6)
    assert np.isclose(np.sum(spread * y**2), 1.5**2, rtol=1e-6)
    with pytest.raises(ValueError, match="blur"):
        FocalPlaneModel(pupil, 1e-6, 16, sampling=2.0, blur=-0.1)
    with pytest.raises(ValueError, match="fine_images"):
        from solvephase.focal import ForwardState

        soft.blur_jvp(ForwardState(None, None, None, None, None))
