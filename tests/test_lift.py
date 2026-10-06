"""LIFT: recovery, Cramer-Rao efficiency, even-mode sign, twin, GPU parity."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import devices
from solvephase import Pupil
from solvephase.algorithms.lift import LIFT, default_astigmatism, lift, lift_crlb, lift_diversity
from solvephase.backend import to_numpy

WL = 1.6e-6
N_MODES = 10  # Noll 2..11
EVEN = np.array([2, 3, 4, 9])  # indices of Z4, Z5, Z6, Z11 (even under x -> -x)


@pytest.fixture(scope="module")
def pupil() -> Pupil:
    return Pupil.circular(32)


@pytest.fixture(scope="module")
def sensor(pupil: Pupil) -> LIFT:
    return LIFT(pupil, WL, 32, sampling=2, n_modes=N_MODES, read_noise=2.0)


def _truth(seed: int, rms_waves: float) -> np.ndarray:
    c = np.random.default_rng(seed).normal(size=N_MODES)
    return c * rms_waves * WL / np.linalg.norm(c)


def _noisy(image: object, rng: np.random.Generator, read_noise: float) -> np.ndarray:
    mean = np.asarray(to_numpy(image), dtype=np.float64)
    return rng.poisson(mean) + rng.normal(0.0, read_noise, mean.shape)


def test_diversity_is_astigmatism(pupil: Pupil) -> None:
    div = lift_diversity(pupil, WL)
    w = pupil.amplitude**2
    rms = np.sqrt(np.sum(w * div**2) / np.sum(w))
    assert rms == pytest.approx(default_astigmatism(WL), rel=0.02)
    # Oblique astigmatism (Noll 5) is symmetric under 180-degree rotation.
    np.testing.assert_allclose(div, div[::-1, ::-1], atol=1e-12 * WL)


def test_recovers_ten_modes_from_one_noisy_image(sensor: LIFT) -> None:
    truth = _truth(1, 0.12)
    photons = 1e5
    rng = np.random.default_rng(7)
    image = _noisy(sensor.expected_image(truth, photons=photons), rng, sensor.read_noise)
    result = sensor.estimate(image)
    sigma = np.sqrt(np.diag(sensor.crlb(truth, photons=photons)))
    err = result.coefficients - truth
    assert result.method == "lift"
    assert np.all(np.abs(err) < 5 * sigma), err / sigma
    assert np.linalg.norm(err) < 0.01 * WL
    assert result.flux[0] == pytest.approx(photons, rel=0.02)


def test_function_matches_class(pupil: Pupil, sensor: LIFT) -> None:
    truth = _truth(2, 0.1)
    image = to_numpy(sensor.expected_image(truth, photons=1e4))
    r1 = lift(image, pupil, WL, sampling=2, n_modes=N_MODES, read_noise=2.0)
    r2 = sensor.estimate(image)
    np.testing.assert_allclose(r1.coefficients, r2.coefficients, atol=1e-6 * WL)
    np.testing.assert_allclose(r1.coefficients, truth, atol=1e-5 * WL)


def test_gaussian_loss_and_warm_start(sensor: LIFT, pupil: Pupil) -> None:
    truth = _truth(3, 0.1)
    rng = np.random.default_rng(3)
    image = _noisy(sensor.expected_image(truth, photons=1e5), rng, 2.0)
    gauss = LIFT(pupil, WL, 32, sampling=2, n_modes=N_MODES, read_noise=2.0, loss="gaussian")
    r_g = gauss.estimate(image)
    r_p = sensor.estimate(image)
    sigma = np.sqrt(np.diag(sensor.crlb(truth, photons=1e5)))
    assert np.all(np.abs(r_g.coefficients - truth) < 6 * sigma)
    # A warm start at the solution stays there and needs few iterations.
    warm = sensor.estimate(image, start=r_p)
    np.testing.assert_allclose(warm.coefficients, r_p.coefficients, atol=0.05 * sigma.min())
    assert warm.n_iter <= 3


def test_even_mode_sign_is_resolved(sensor: LIFT) -> None:
    """Flipping the even modes changes the estimate: the astigmatism breaks the twin."""
    truth = _truth(4, 0.08)
    flipped = truth.copy()
    flipped[EVEN] *= -1
    est = []
    for c in (truth, flipped):
        est.append(sensor.estimate(sensor.expected_image(c, photons=1e5)).coefficients)
    np.testing.assert_allclose(est[0], truth, atol=1e-4 * WL)
    np.testing.assert_allclose(est[1], flipped, atol=1e-4 * WL)
    # Without diversity both aberrations make the same image (up to the twin).
    blind = LIFT(sensor.model.pupil, WL, 32, sampling=2, n_modes=N_MODES, astigmatism=0.0)
    i0 = to_numpy(blind.expected_image(truth))
    twin = truth.copy()
    twin[EVEN] *= -1  # -phi(-x): odd modes unchanged, even modes negated
    np.testing.assert_allclose(to_numpy(blind.expected_image(twin)), i0, atol=1e-12)
    assert not np.allclose(
        to_numpy(sensor.expected_image(twin)), to_numpy(sensor.expected_image(truth)), atol=1e-6
    )


def test_twin_gives_the_same_image(sensor: LIFT) -> None:
    truth = _truth(5, 0.1)
    twin = sensor.twin(truth)
    i0 = to_numpy(sensor.expected_image(truth))
    np.testing.assert_allclose(to_numpy(sensor.expected_image(twin)), i0, atol=1e-12)
    assert not np.allclose(twin, truth)
    # The twin of an aberration with a large negative astigmatism is the smaller one.
    big = np.zeros(N_MODES)
    big[3] = -1.5 * sensor.astigmatism
    result = sensor.estimate(sensor.expected_image(big, photons=1e5))
    assert result.extra["twin_swapped"] or np.allclose(result.coefficients, sensor.twin(big))
    np.testing.assert_allclose(result.coefficients, sensor.twin(big), atol=1e-4 * WL)


def test_fisher_matches_finite_differences(sensor: LIFT) -> None:
    truth = _truth(6, 0.05)
    photons = 1e4
    fisher = sensor.fisher(truth, photons=photons)
    n = N_MODES
    m0 = np.asarray(to_numpy(sensor.expected_image(truth, photons=photons)))
    h = 1e-4 * WL
    jac = []
    for i in range(n):
        dp, dm = truth.copy(), truth.copy()
        dp[i] += h
        dm[i] -= h
        diff = to_numpy(sensor.expected_image(dp, photons=photons)) - to_numpy(
            sensor.expected_image(dm, photons=photons)
        )
        jac.append(np.asarray(diff).ravel() / (2 * h))
    jac.append(m0.ravel())  # d m / d log-flux
    j = np.array(jac)
    expected = (j / (m0.ravel() + sensor.read_noise**2)) @ j.T
    np.testing.assert_allclose(fisher, expected, rtol=1e-4, atol=1e-6 * np.abs(expected).max())


def test_crlb_scaling_and_helper(pupil: Pupil, sensor: LIFT) -> None:
    truth = _truth(7, 0.05)
    noiseless = LIFT(pupil, WL, 32, sampling=2, n_modes=N_MODES)
    c1 = np.diag(noiseless.crlb(truth, photons=1e4))
    c2 = np.diag(noiseless.crlb(truth, photons=1e5))
    np.testing.assert_allclose(c1 / c2, 10.0, rtol=1e-6)  # photon-limited: 1 / N
    helper = lift_crlb(pupil, WL, 32, truth, photons=1e4, read_noise=2.0, sampling=2)
    np.testing.assert_allclose(helper, sensor.crlb(truth, photons=1e4), rtol=1e-8)
    # Read noise only adds variance.
    assert np.all(np.diag(helper) > c1)


def _efficiency(n_draws: int, photons: float) -> tuple[np.ndarray, np.ndarray]:
    """Variance / CRB ratios and bias / sqrt(CRB) over noise realizations."""
    pupil = Pupil.circular(24)
    sensor = LIFT(pupil, WL, 24, sampling=2, n_modes=N_MODES, read_noise=1.0)
    truth = _truth(8, 0.1)
    mean = sensor.expected_image(truth, photons=photons)
    crb = np.diag(sensor.crlb(truth, photons=photons))
    rng = np.random.default_rng(2024)
    est = np.array([sensor.estimate(_noisy(mean, rng, 1.0)).coefficients for _ in range(n_draws)])
    return est.var(axis=0, ddof=1) / crb, (est.mean(axis=0) - truth) / np.sqrt(crb)


def test_estimator_reaches_crlb() -> None:
    """At high flux the estimator variance is close to the Cramer-Rao bound."""
    n = 30
    ratio, bias = _efficiency(n, 1e5)
    # Each variance ratio has a standard error of sqrt(2 / n) ~ 26 %; their mean ~ 8 %.
    assert 0.75 < ratio.mean() < 1.3, ratio
    assert np.all(ratio < 2.0), ratio
    assert np.all(np.abs(bias) < 4 / np.sqrt(n) + 0.05), bias


@pytest.mark.slow
@pytest.mark.parametrize("photons", [1e5, 1e6])
def test_estimator_reaches_crlb_ensemble(photons: float) -> None:
    n = 300
    ratio, bias = _efficiency(n, photons)
    assert 0.85 < ratio.mean() < 1.15, ratio  # efficient: variance = CRB
    assert np.all(ratio < 1.5), ratio
    assert np.all(np.abs(bias) < 4 / np.sqrt(n) + 0.05), bias


def test_errors(pupil: Pupil, sensor: LIFT) -> None:
    from solvephase import Basis

    with pytest.raises(ValueError, match="one focal-plane image"):
        sensor.estimate(np.ones((2, 32, 32)))
    with pytest.raises(ValueError, match="modal basis"):
        LIFT(pupil, WL, 32, sampling=2, basis=Basis.zonal(pupil))
    with pytest.raises(ValueError, match="astigmatism_mode"):
        LIFT(pupil, WL, 32, sampling=2, astigmatism_mode=4)
    with pytest.raises(ValueError, match="loss"):
        LIFT(pupil, WL, 32, sampling=2, loss="amplitude")
    with pytest.raises(ValueError, match="photons"):
        sensor.crlb(photons=0.0)


@pytest.mark.parametrize("device", devices())
def test_device_parity(pupil: Pupil, device: str) -> None:
    truth = _truth(9, 0.1)
    cpu = LIFT(pupil, WL, 32, sampling=2, n_modes=N_MODES, read_noise=2.0)
    rng = np.random.default_rng(11)
    image = _noisy(cpu.expected_image(truth, photons=1e5), rng, 2.0)
    ref = cpu.estimate(image)
    other = LIFT(pupil, WL, 32, sampling=2, n_modes=N_MODES, read_noise=2.0, device=device)
    result = other.estimate(image)
    assert result.device == device
    sigma = np.sqrt(np.diag(cpu.crlb(truth, photons=1e5)))
    np.testing.assert_allclose(result.coefficients, ref.coefficients, atol=0.05 * sigma.min())
    crb_dev = LIFT(
        pupil,
        WL,
        32,
        sampling=2,
        n_modes=N_MODES,
        read_noise=2.0,
        device=device,
        precision="double",
    ).crlb(truth, photons=1e5)
    np.testing.assert_allclose(crb_dev, cpu.crlb(truth, photons=1e5), rtol=1e-6)
