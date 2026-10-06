"""Coherent diffraction imaging: projections, algorithms, shrinkwrap, alignment."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from conftest import devices
from solvephase.algorithms.cdi import (
    CDI_ALGORITHMS,
    CDIResult,
    ShrinkwrapConfig,
    align_object,
    autocorrelation_support,
    cdi,
    parse_schedule,
    simulate_cdi,
)
from solvephase.backend import to_numpy


def _square_object(seed: int, n: int = 32) -> np.ndarray:
    """Random positive object filling an n x n box (the classic hard test)."""
    return np.random.default_rng(seed).uniform(0.2, 1.0, (n, n))


def _blob(seed: int, n: int = 32) -> np.ndarray:
    """Smooth positive object with an irregular, non-centrosymmetric outline."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:n, :n] - (n - 1) / 2
    r, th = np.hypot(yy, xx), np.arctan2(yy, xx)
    edge = n * 0.38 * (1 + 0.25 * np.cos(3 * th + rng.uniform(0, 6)) + 0.1 * np.sin(2 * th))
    tex = gaussian_filter(rng.uniform(0, 1, (n, n)), 1.0)
    tex = 0.3 + (tex - tex.min()) / (tex.max() - tex.min())
    return np.where(r < edge, tex, 0.0)


# ----------------------------------------------------------------------------- schedule
def test_parse_schedule_forms() -> None:
    assert parse_schedule("hio:400, er:100") == [("hio", 400, 0.9), ("er", 100, 1.0)]
    assert parse_schedule([("raar", 10, 0.8), ("dr", 5)], beta=0.7) == [
        ("raar", 10, 0.8),
        ("asr", 5, 0.7),
    ]
    assert parse_schedule(("oss", 20)) == [("oss", 20, 0.9)]
    assert parse_schedule("rrr:3:0.4,er:0") == [("rrr", 3, 0.4)]
    for bad in ("foo:10", "hio", [("hio",)], "er:0", [("hio", 5, -1.0)], "hio:-3"):
        with pytest.raises(ValueError):
            parse_schedule(bad)


def test_input_validation() -> None:
    sim = simulate_cdi(np.ones((4, 4)), 2)
    with pytest.raises(ValueError, match="constraint"):
        cdi(sim.magnitudes, sim.support, constraint="imaginary")
    with pytest.raises(ValueError, match="support has shape"):
        cdi(sim.magnitudes, np.ones((4, 4), bool))
    with pytest.raises(ValueError, match="empty"):
        cdi(sim.magnitudes, np.zeros((8, 8), bool))
    with pytest.raises(ValueError, match="intensity=True"):
        cdi(-sim.magnitudes - 1, sim.support)
    with pytest.raises(ValueError, match="starts"):
        cdi(sim.magnitudes, sim.support, starts=0)
    with pytest.raises(TypeError, match="shrinkwrap"):
        cdi(sim.magnitudes, sim.support, shrinkwrap=3)
    with pytest.raises(ValueError, match="threshold"):
        ShrinkwrapConfig(threshold=1.5)


# ----------------------------------------------------------------------------- simulation
def test_simulate_cdi_parseval_centring_and_photons() -> None:
    obj = _blob(1, 16) * np.exp(0.3j * np.arange(16))[None, :]
    sim = simulate_cdi(obj, oversampling=2)
    mags, padded = to_numpy(sim.magnitudes), to_numpy(sim.object)
    assert mags.shape == padded.shape == (32, 32)
    # Unitary transform: Parseval, and the DC term at the array centre.
    assert np.isclose(np.sum(mags**2), np.sum(np.abs(obj) ** 2), rtol=1e-12)
    assert np.isclose(mags[16, 16], abs(obj.sum()) / 32, rtol=1e-12)
    assert np.array_equal(to_numpy(sim.support), np.abs(padded) > 0)
    noisy = simulate_cdi(obj, 2, photons=1e6, seed=0)
    counts = to_numpy(noisy.intensity)
    assert np.all(counts == np.round(counts))
    assert abs(counts.sum() - 1e6) < 5 * np.sqrt(1e6)
    # The returned truth is scaled to the photon count.
    assert np.isclose(np.sum(np.abs(to_numpy(noisy.object)) ** 2), 1e6, rtol=1e-12)
    assert np.allclose(to_numpy(noisy.magnitudes) ** 2, counts)


def test_autocorrelation_support_contains_object() -> None:
    sim = simulate_cdi(_blob(2), 2)
    auto = to_numpy(autocorrelation_support(sim.intensity, 0.04))
    truth = to_numpy(sim.support)
    assert np.all(auto[truth])
    assert auto.sum() > 2 * truth.sum()  # loose: the autocorrelation is twice as wide


# ----------------------------------------------------------------------------- alignment
def test_align_object_removes_shift_phase_and_twin(rng: np.random.Generator) -> None:
    ref = np.zeros((32, 32), complex)
    ref[8:24, 10:22] = rng.normal(size=(16, 12)) + 1j * rng.normal(size=(16, 12))
    moved = 0.7 * np.exp(1.1j) * np.roll(ref, (3, -5), axis=(0, 1))
    aligned, err = align_object(moved, ref)
    assert err < 1e-12
    assert np.allclose(aligned, ref, atol=1e-12)
    twin = np.exp(-0.4j) * np.roll(np.conj(ref[::-1, ::-1]), (-2, 4), axis=(0, 1))
    assert align_object(twin, ref)[1] < 1e-12
    assert align_object(twin, ref, twin=False)[1] > 0.5
    # Only a global phase: the 0.7 scale stays in the error.
    _, err_phase = align_object(moved, ref, scale=False)
    assert np.isclose(err_phase, 0.3, atol=1e-12)


def test_align_object_subpixel() -> None:
    ref = np.zeros((48, 48))
    ref[12:36, 14:34] = gaussian_filter(np.random.default_rng(4).uniform(size=(24, 20)), 1.5)
    ky = np.fft.fftfreq(48)[:, None]
    kx = np.fft.fftfreq(48)[None, :]
    shift = (2.37, -1.25)  # multiples of 1/upsample are recovered exactly
    ramp = np.exp(-2j * np.pi * (ky * shift[0] + kx * shift[1]))
    moved = np.fft.ifft2(np.fft.fft2(ref) * ramp) * np.exp(0.5j)
    _, err = align_object(moved, ref)
    assert err < 1e-10
    _, err_int = align_object(moved, ref, subpixel=False)
    assert err_int > 1e-2


# ----------------------------------------------------------------------------- recovery
def test_hio_er_recovers_random_positive_object() -> None:
    errors = []
    for seed in range(5):
        sim = simulate_cdi(_square_object(100 + seed), 2)
        res = cdi(
            sim.magnitudes, sim.support, schedule="hio:800,er:100", constraint="positive", seed=seed
        )
        errors.append(align_object(res.object, sim.object)[1])
        assert res.n_iter <= 900
        assert len(res.history) == len(res.times) == len(res.support_history)
    assert sum(e < 1e-3 for e in errors) >= 4, errors


@pytest.mark.parametrize("algorithm", CDI_ALGORITHMS)
def test_each_algorithm_reduces_modulus_error(algorithm: str) -> None:
    sim = simulate_cdi(_square_object(7, 16), 2)
    res = cdi(
        sim.magnitudes,
        sim.support,
        schedule=[(algorithm, 300)],
        constraint="positive",
        starts=4,
        seed=3,
        tol=0,
    )
    first, last = res.history[0], res.modulus_error
    if algorithm == "er":
        # Error reduction never increases the modulus error (Fienup 1982) but stagnates.
        assert np.all(np.diff(res.history) <= 1e-12)
        assert last < 0.7 * first
    else:
        assert last < first / 10, (first, last)
    assert res.modulus_error == pytest.approx(res.start_errors.min())


def test_complex_object_with_dm_then_er() -> None:
    rng = np.random.default_rng(5)
    obj = _blob(5, 24) * np.exp(1j * gaussian_filter(rng.normal(size=(24, 24)), 3) * 4)
    sim = simulate_cdi(obj, 2)
    res = cdi(sim.magnitudes, sim.support, schedule="dm:500,er:50", starts=4, seed=1)
    assert align_object(res.object, sim.object)[1] < 1e-3


def test_shrinkwrap_from_autocorrelation_support() -> None:
    sim = simulate_cdi(_blob(3), 2)
    loose = autocorrelation_support(sim.intensity, 0.04)
    res = cdi(
        sim.magnitudes,
        loose,
        schedule="hio:800,er:200",
        constraint="positive",
        shrinkwrap={"stop": 800},
        starts=2,
        seed=0,
    )
    assert align_object(res.object, sim.object)[1] < 1e-3
    support, truth = to_numpy(res.support), to_numpy(sim.support)
    assert support.sum() < 0.5 * to_numpy(loose).sum()
    # The refined support hugs the object (up to the translation ambiguity).
    assert truth.sum() <= support.sum() < 1.5 * truth.sum()


def test_multistart_picks_best_start() -> None:
    sim = simulate_cdi(_square_object(1), 2)
    res = cdi(
        sim.magnitudes,
        sim.support,
        schedule="hio:400,er:100",
        constraint="positive",
        starts=8,
        seed=1,
    )
    assert res.start_errors.shape == (8,)
    assert res.best_start == int(np.argmin(res.start_errors))
    assert res.modulus_error == res.start_errors.min()
    assert np.array_equal(to_numpy(res.object), to_numpy(res.objects)[res.best_start])
    assert res.start_errors.max() > 10 * res.start_errors.min()  # starts really differ
    assert align_object(res.object, sim.object)[1] < 1e-3


def test_batched_start_matches_single_start() -> None:
    sim = simulate_cdi(_square_object(2, 16), 2)
    kwargs = {"schedule": "hio:60,er:20", "constraint": "positive", "seed": 9, "tol": 0}
    single = cdi(sim.magnitudes, sim.support, starts=1, **kwargs)
    batch = cdi(sim.magnitudes, sim.support, starts=3, **kwargs)
    assert np.allclose(to_numpy(batch.objects)[0], to_numpy(single.object), atol=1e-10)
    assert batch.start_errors[0] == pytest.approx(single.modulus_error, rel=1e-8)


def test_beamstop_unmeasured_pixels_are_left_free() -> None:
    sim = simulate_cdi(_blob(4), 2)
    mags = to_numpy(sim.magnitudes).copy()
    measured = np.ones(mags.shape, bool)
    measured[30:35, 30:35] = False  # 5 x 5 beamstop over the DC peak
    mags[~measured] = np.nan  # unmeasured values are never used
    res = cdi(
        mags,
        sim.support,
        measured=measured,
        schedule="hio:800,er:200",
        constraint="positive",
        seed=0,
    )
    assert align_object(res.object, sim.object)[1] < 1e-4
    # Imposing zeros at the beamstop instead ruins the reconstruction.
    zeroed = np.where(measured, mags, 0.0)
    bad = cdi(zeroed, sim.support, schedule="hio:300,er:50", constraint="positive", seed=0)
    assert align_object(bad.object, sim.object)[1] > 0.1


def test_intensity_input_bounds_and_real_constraint() -> None:
    obj = _square_object(11, 16) - 0.5  # real with both signs
    sim = simulate_cdi(obj, 2)
    res = cdi(
        sim.intensity,
        sim.support,
        intensity=True,
        constraint="real",
        bounds=(None, 0.45),
        schedule="hio:100,er:20",
        seed=0,
    )
    est = to_numpy(res.object)
    assert np.all(est.imag == 0)
    assert np.max(np.abs(est.real)) <= 0.45 + 1e-12
    assert np.all(est[~to_numpy(res.support)] == 0)


def test_tol_callback_and_to_numpy() -> None:
    sim = simulate_cdi(_blob(6, 16), 2)
    res = cdi(
        sim.magnitudes,
        sim.support,
        schedule="hio:2000,er:100",
        constraint="positive",
        tol=1e-8,
        seed=0,
    )
    assert res.converged and res.message == "modulus error reached tol"
    assert res.n_iter < 2100
    assert res.modulus_error < 1e-7
    seen: list[int] = []

    def stop_at_30(it: int, obj: object, err: float) -> bool:
        assert np.shape(obj) == (32, 32)
        seen.append(it)
        return it >= 30

    res = cdi(
        sim.magnitudes, sim.support, schedule="hio:500", seed=0, check_every=10, callback=stop_at_30
    )
    assert seen == [10, 20, 30] and res.n_iter == 30 and res.message == "stopped by callback"
    host = res.to_numpy()
    assert isinstance(host, CDIResult) and isinstance(host.object, np.ndarray)


def test_single_precision_cpu() -> None:
    sim = simulate_cdi(_blob(8, 16), 2)
    res = cdi(
        sim.magnitudes,
        sim.support,
        schedule="hio:600,er:100",
        constraint="positive",
        precision="single",
        seed=2,
    )
    assert res.object.dtype == np.complex64
    assert align_object(res.object, sim.object)[1] < 1e-3


@pytest.mark.slow
def test_poisson_noise_recovery_improves_with_photons() -> None:
    obj = _blob(9, 32)
    errors = []
    for photons in (1e6, 1e9):
        sim = simulate_cdi(obj, 2, photons=photons, seed=1)
        res = cdi(
            sim.magnitudes,
            sim.support,
            schedule="oss:1000,er:50",
            constraint="positive",
            starts=4,
            seed=0,
        )
        errors.append(align_object(res.object, sim.object)[1])
    assert errors[0] < 0.1 and errors[1] < 0.01, errors
    assert errors[1] < errors[0]


@pytest.mark.slow
@pytest.mark.parametrize("device", devices())
def test_recovery_256_multistart(device: str) -> None:
    sim = simulate_cdi(_blob(10, 128), 2, device=device)
    loose = autocorrelation_support(sim.intensity, 0.04)
    res = cdi(
        sim.magnitudes,
        loose,
        schedule="hio:600,er:100",
        constraint="positive",
        shrinkwrap={"stop": 600},
        starts=4,
        seed=0,
        device=device,
    )
    assert res.device == ("gpu" if device == "gpu" else "cpu")
    assert align_object(res.object, sim.object)[1] < 1e-2


# ----------------------------------------------------------------------------- GPU
@pytest.mark.gpu
@pytest.mark.parametrize("algorithm", ["hio", "dm", "raar", "oss"])
def test_gpu_matches_cpu(algorithm: str) -> None:
    sim = simulate_cdi(_square_object(3, 16), 2)
    kwargs = {
        "schedule": [(algorithm, 40), ("er", 10)],
        "constraint": "positive",
        "starts": 3,
        "seed": 4,
        "precision": "double",
        "shrinkwrap": {"every": 10},
        "tol": 0,
    }
    cpu = cdi(sim.magnitudes, sim.support, device="cpu", **kwargs)
    gpu = cdi(sim.magnitudes, sim.support, device="gpu", **kwargs)
    assert gpu.device == "gpu"
    assert np.allclose(to_numpy(gpu.objects), to_numpy(cpu.objects), atol=1e-9)
    assert np.allclose(gpu.start_errors, cpu.start_errors, rtol=1e-6, atol=1e-12)
    assert np.allclose(cpu.history, gpu.history, rtol=1e-6, atol=1e-12)


@pytest.mark.gpu
def test_gpu_single_precision_recovery() -> None:
    sim = simulate_cdi(_square_object(105), 2, device="gpu")
    res = cdi(
        sim.magnitudes,
        sim.support,
        schedule="hio:800,er:100",
        constraint="positive",
        starts=8,
        seed=0,
        device="gpu",
    )
    assert res.object.dtype == np.complex64
    assert type(res.object).__module__.startswith("cupy")
    aligned, err = align_object(res.object, sim.object)
    assert type(aligned).__module__.startswith("cupy")
    assert err < 1e-3


def test_initial_guess_at_the_solution_is_a_fixed_point() -> None:
    sim = simulate_cdi(_blob(12, 16), 2)
    for name in ("er", "hio", "raar", "dm"):
        res = cdi(sim.magnitudes, sim.support, initial=sim.object, schedule=[(name, 20)])
        assert res.modulus_error < 1e-12
        assert np.allclose(to_numpy(res.object), to_numpy(sim.object), atol=1e-12)
