"""Transport-of-intensity retrieval: accuracy against simulated defocus stacks."""

from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import devices
from solvephase.algorithms.tie import (
    TIEResult,
    _axial_weights,
    simulate_defocus_stack,
    tie,
)
from solvephase.backend import to_numpy

PITCH = 10e-6
WAVELENGTH = 633e-9
N = 128


def _grid(n: int = N) -> tuple[np.ndarray, np.ndarray]:
    x = (np.arange(n) - (n - 1) / 2.0) * PITCH
    return np.meshgrid(x, x)


def _gauss(xx: np.ndarray, yy: np.ndarray, x0: float, y0: float, sigma: float) -> np.ndarray:
    return np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2.0 * sigma**2))


def _bumps(n: int = N) -> np.ndarray:
    """A few radians of smooth phase, well inside the window."""
    xx, yy = _grid(n)
    return (
        2.5 * _gauss(xx, yy, 1e-4, -5e-5, 0.9e-4)
        - 1.8 * _gauss(xx, yy, -1.2e-4, 1e-4, 0.8e-4)
        + 1.0 * _gauss(xx, yy, 1e-4, 1.6e-4, 0.6e-4)
    )


def _nonuniform_amplitude(n: int = N) -> np.ndarray:
    xx, yy = _grid(n)
    return np.sqrt(
        0.3
        + 0.7 * _gauss(xx, yy, -0.5e-4, 0.5e-4, 3e-4)
        + 0.4 * _gauss(xx, yy, 2e-4, -2e-4, 1.5e-4)
    )


def _rel_error(phase: np.ndarray, truth: np.ndarray) -> float:
    """RMS error over RMS truth, both piston-free."""
    est = to_numpy(phase)
    t = truth - truth.mean()
    return float(np.sqrt(np.mean((est - est.mean() - t) ** 2)) / np.sqrt(np.mean(t**2)))


def _stack(field: np.ndarray, z: list[float] | np.ndarray) -> np.ndarray:
    return simulate_defocus_stack(field, z, PITCH, WAVELENGTH)


@pytest.fixture(scope="module")
def bumps() -> np.ndarray:
    return _bumps()


# ----------------------------------------------------------- forward model
def test_simulated_stack_conserves_flux_and_focus(bumps: np.ndarray) -> None:
    field = _nonuniform_amplitude() * np.exp(1j * bumps)
    z = [-2e-3, 0.0, 1e-3, 4e-3]
    stack = _stack(field, z)
    assert stack.shape == (4, N, N)
    np.testing.assert_allclose(stack[1], np.abs(field) ** 2, rtol=1e-10, atol=1e-12)
    flux = np.sum(np.abs(field) ** 2)
    np.testing.assert_allclose(stack.sum(axis=(1, 2)), flux, rtol=1e-10)
    # Defocus actually changes the intensity.
    assert np.max(np.abs(stack[3] - stack[1])) > 0.05


def test_tie_sign_matches_propagation(bumps: np.ndarray) -> None:
    """``k dI/dz = -laplacian(phi)`` for a pure phase object (Teague 1983)."""
    stack = _stack(np.exp(1j * bumps), [-1e-4, 0.0, 1e-4])
    didz = (stack[2] - stack[0]) / 2e-4
    k = 2.0 * math.pi / WAVELENGTH
    qy = 2.0 * math.pi * np.fft.fftfreq(N, d=PITCH)[:, None]
    qx = 2.0 * math.pi * np.fft.fftfreq(N, d=PITCH)[None, :]
    neg_lap = np.real(np.fft.ifft2(np.fft.fft2(bumps) * (qy**2 + qx**2)))
    np.testing.assert_allclose(k * didz, neg_lap, atol=2e-3 * np.abs(neg_lap).max())


# ------------------------------------------------------- axial derivative
def test_polyfit_weights_exact_for_polynomials() -> None:
    z = np.array([-3e-3, -1e-3, 0.0, 1.5e-3, 4e-3])
    coeffs = [2.0, -300.0, 4e4, -6e6]  # I(z) = sum c_m z^m
    values = sum(c * z**m for m, c in enumerate(coeffs))
    slope, value, focus = _axial_weights(z, "polyfit", 3)
    assert focus == 2
    assert slope @ values == pytest.approx(coeffs[1], rel=1e-9)
    assert value @ values == pytest.approx(coeffs[0], rel=1e-9)


def test_central_weights() -> None:
    # Symmetric pair: the classic central difference, no weight on z = 0.
    slope, _, focus = _axial_weights(np.array([-2e-3, 0.0, 2e-3]), "central", None)
    np.testing.assert_allclose(slope, [-250.0, 0.0, 250.0], atol=1e-9)
    assert focus == 1
    # Asymmetric three planes: exact for a quadratic (three-point Lagrange).
    z = np.array([-1e-3, 0.0, 3e-3])
    slope, _, _ = _axial_weights(z, "central", None)
    assert slope @ (1.0 + 5.0 * z + 7e3 * z**2) == pytest.approx(5.0, rel=1e-9)
    # Only the planes nearest focus are used.
    z5 = np.array([-4e-3, -2e-3, 0.0, 2e-3, 4e-3])
    slope, _, _ = _axial_weights(z5, "central", None)
    np.testing.assert_allclose(slope, [0.0, -250.0, 0.0, 250.0, 0.0], atol=1e-9)


def test_in_focus_estimated_without_focal_plane(bumps: np.ndarray) -> None:
    field = _nonuniform_amplitude() * np.exp(1j * bumps)
    z = [-5e-4, 5e-4]
    res = tie(_stack(field, z), z, pitch=PITCH, wavelength=WAVELENGTH, method="fft")
    i0 = np.abs(field) ** 2
    assert np.max(np.abs(res.intensity - i0)) < 0.02 * i0.max()
    assert _rel_error(res.phase, bumps) < 0.05


# ----------------------------------------------------------------- solvers
@pytest.mark.parametrize("method", ["fft", "dct", "pcg"])
def test_uniform_amplitude_recovery(bumps: np.ndarray, method: str) -> None:
    z = [-1e-3, 0.0, 1e-3]
    res = tie(_stack(np.exp(1j * bumps), z), z, pitch=PITCH, wavelength=WAVELENGTH, method=method)
    assert isinstance(res, TIEResult)
    err = _rel_error(res.phase, bumps)
    # Spectral solvers: derivative error only (1.2e-4). PCG: five-point stencil (1.4e-3).
    assert err < (3e-4 if method != "pcg" else 3e-3)
    assert abs(float(np.mean(res.phase))) < 1e-12
    np.testing.assert_allclose(res.opd, res.phase * WAVELENGTH / (2.0 * math.pi))
    assert res.converged


@pytest.mark.parametrize("method", ["fft", "dct"])
def test_error_falls_as_dz_squared(bumps: np.ndarray, method: str) -> None:
    field = np.exp(1j * bumps)
    errors = []
    for dz in (8e-3, 4e-3, 2e-3, 1e-3):
        z = [-dz, 0.0, dz]
        res = tie(_stack(field, z), z, pitch=PITCH, wavelength=WAVELENGTH, method=method)
        errors.append(_rel_error(res.phase, bumps))
    ratios = np.array(errors[1:]) / np.array(errors[:-1])
    # Central difference truncation error ~ dz^2: each halving gains about 4x.
    assert np.all(ratios < 0.3), errors
    assert np.all(ratios > 0.2), errors


def test_noise_sets_an_optimal_defocus(bumps: np.ndarray) -> None:
    """Too small a dz amplifies noise, too large a dz is nonlinear."""
    rng = np.random.default_rng(7)
    field = np.exp(1j * bumps)
    errors = {}
    for dz in (16e-3, 4e-3, 5e-4):
        z = [-dz, 0.0, dz]
        clean = _stack(field, z)
        trials = [
            _rel_error(
                tie(
                    clean + 1e-3 * rng.standard_normal(clean.shape),
                    z,
                    pitch=PITCH,
                    wavelength=WAVELENGTH,
                    method="fft",
                    regularization=1e-3,
                ).phase,
                bumps,
            )
            for _ in range(3)
        ]
        errors[dz] = float(np.mean(trials))
    assert errors[4e-3] < 0.5 * errors[16e-3], errors
    assert errors[4e-3] < 0.2 * errors[5e-4], errors


@pytest.mark.parametrize("method", ["fft", "dct", "pcg"])
def test_nonuniform_solution_beats_uniform_approximation(bumps: np.ndarray, method: str) -> None:
    field = _nonuniform_amplitude() * np.exp(1j * bumps)
    z = [-1e-3, 0.0, 1e-3]
    stack = _stack(field, z)
    kwargs = {"pitch": PITCH, "wavelength": WAVELENGTH, "method": method}
    err_u = _rel_error(tie(stack, z, uniform=True, **kwargs).phase, bumps)
    err_n = _rel_error(tie(stack, z, **kwargs).phase, bumps)
    # Measured: uniform ~0.78; Teague (fft/dct) ~0.03-0.04; exact PCG ~1.4e-3.
    assert err_u > 0.5
    assert err_n < (0.05 if method != "pcg" else 4e-3)


def test_polyfit_beats_central_difference_at_large_dz(bumps: np.ndarray) -> None:
    field = np.exp(1j * bumps)
    big = 8e-3
    z5 = np.array([-big, -big / 2, 0.0, big / 2, big])
    stack5 = _stack(field, z5)
    kwargs = {"pitch": PITCH, "wavelength": WAVELENGTH, "method": "fft"}
    central = tie(stack5[[0, 2, 4]], z5[[0, 2, 4]], **kwargs)
    poly = tie(stack5, z5, derivative="polyfit", **kwargs)
    err_c = _rel_error(central.phase, bumps)
    err_p = _rel_error(poly.phase, bumps)
    # Measured: 7.5e-3 vs 1.3e-4.
    assert err_p < err_c / 20.0
    assert poly.derivative == "polyfit"


def test_wavelength_scaling(bumps: np.ndarray) -> None:
    """For a fixed OPD the recovered OPD is wavelength independent, phase ~ 1/lambda."""
    opd = bumps * WAVELENGTH / (2.0 * math.pi)
    z = [-1e-3, 0.0, 1e-3]
    results = []
    for lam in (WAVELENGTH, 2.0 * WAVELENGTH):
        field = np.exp(2j * math.pi * opd / lam)
        stack = simulate_defocus_stack(field, z, PITCH, lam)
        results.append(tie(stack, z, pitch=PITCH, wavelength=lam, method="dct"))
    opd_t = opd - opd.mean()
    for res in results:
        assert np.sqrt(np.mean((res.opd - opd_t) ** 2)) < 1e-3 * np.sqrt(np.mean(opd_t**2))
    ratio = np.sum(results[0].phase * results[1].phase) / np.sum(results[1].phase ** 2)
    assert ratio == pytest.approx(2.0, rel=1e-3)


def test_pitch_units(bumps: np.ndarray) -> None:
    """Reporting a wrong pitch scales the Laplacian, hence the phase, by its square."""
    z = [-1e-3, 0.0, 1e-3]
    stack = _stack(np.exp(1j * bumps), z)
    good = tie(stack, z, pitch=PITCH, wavelength=WAVELENGTH, method="fft")
    off = tie(stack, z, pitch=2 * PITCH, wavelength=WAVELENGTH, method="fft")
    np.testing.assert_allclose(off.phase, 4.0 * good.phase, rtol=1e-9, atol=1e-12)


def test_regularization_attenuates_the_fundamental() -> None:
    """Tikhonov weight ``r`` divides the lowest grid frequency by ``1 + r``."""
    xx, _ = _grid()
    length = N * PITCH
    phase = 0.5 * np.cos(2.0 * math.pi * xx / length)
    z = [-1e-3, 0.0, 1e-3]
    stack = _stack(np.exp(1j * phase), z)
    amps = []
    for reg in (0.0, 1.0):
        res = tie(stack, z, pitch=PITCH, wavelength=WAVELENGTH, method="fft", regularization=reg)
        amps.append(np.sum(res.phase * phase) / np.sum(phase**2))
    assert amps[0] == pytest.approx(1.0, abs=1e-4)
    assert amps[1] == pytest.approx(0.5, abs=1e-4)


def test_dct_handles_non_periodic_field() -> None:
    """A phase step across the window breaks periodicity but not Neumann boundaries."""
    big = 2 * N
    xx, yy = _grid(big)
    phase = 1.5 * np.tanh(xx / 1.5e-4) + _gauss(xx, yy, 0.0, 1e-4, 1e-4)
    z = [-5e-4, 0.0, 5e-4]
    crop = slice(N // 2, N // 2 + N)
    stack = _stack(np.exp(1j * phase), z)[:, crop, crop]
    truth = phase[crop, crop]
    kwargs = {"pitch": PITCH, "wavelength": WAVELENGTH, "uniform": True}
    err_fft = _rel_error(tie(stack, z, method="fft", **kwargs).phase, truth)
    err_dct = _rel_error(tie(stack, z, method="dct", **kwargs).phase, truth)
    # Measured: fft 0.66, dct 2.7e-3.
    assert err_dct < 0.01
    assert err_fft > 0.3


def _aperture(n: int = N) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Soft-edged pupil of radius ``n / 4`` pixels with defocus, astigmatism and coma."""
    xx, yy = _grid(n)
    radius = n / 4 * PITCH
    r = np.hypot(xx, yy)
    rho, theta = r / radius, np.arctan2(yy, xx)
    phase = (
        1.5 * (2 * rho**2 - 1)
        + rho**2 * np.cos(2 * theta)
        + 0.7 * (3 * rho**3 - 2 * rho) * np.cos(theta)
    )
    amplitude = 0.5 * (1.0 - np.tanh((r - radius) / (2.0 * PITCH)))
    return amplitude, phase, r < 0.9 * radius


def test_pcg_recovers_wavefront_inside_illuminated_aperture() -> None:
    """Curvature sensing: the aperture-edge signal carries the boundary slopes."""
    amplitude, phase, inner = _aperture()
    z = [-1e-3, 0.0, 1e-3]
    stack = _stack(amplitude * np.exp(1j * phase), z)
    kwargs = {"pitch": PITCH, "wavelength": WAVELENGTH}

    def rms_inner(res: TIEResult) -> float:
        d = (to_numpy(res.phase) - phase)[inner]
        return float(np.sqrt(np.mean((d - d.mean()) ** 2)))

    pcg = tie(stack, z, method="pcg", **kwargs)
    teague = tie(stack, z, method="dct", **kwargs)
    assert pcg.converged and pcg.n_iter < 500
    assert pcg.history[-1] <= 1e-6
    # Measured: pcg 4.3e-3 rad, Teague/DCT 0.28 rad (phase RMS ~1.3 rad).
    assert rms_inner(pcg) < 0.01
    assert rms_inner(teague) > 0.1
    assert not np.any(to_numpy(pcg.phase)[~to_numpy(pcg.mask)])


def test_pcg_removes_piston_per_region() -> None:
    amplitude, phase, _ = _aperture()
    xx, _ = _grid()
    split = np.abs(xx) > 2 * PITCH  # two half-pupils
    z = [-1e-3, 0.0, 1e-3]
    stack = _stack(amplitude * np.exp(1j * phase), z)
    mask = split & (amplitude**2 > 1e-3)
    res = tie(stack, z, pitch=PITCH, wavelength=WAVELENGTH, method="pcg", mask=mask)
    for side in (xx < 0, xx > 0):
        assert abs(float(np.mean(res.phase[mask & side]))) < 1e-10


@pytest.mark.parametrize("method", ["fft", "dct", "pcg"])
def test_single_precision(bumps: np.ndarray, method: str) -> None:
    z = [-1e-3, 0.0, 1e-3]
    stack = _stack(np.exp(1j * bumps), z).astype(np.float32)
    res = tie(stack, z, pitch=PITCH, wavelength=WAVELENGTH, method=method)
    assert res.phase.dtype == np.float32
    assert _rel_error(res.phase, bumps) < 3e-3


# ------------------------------------------------------------------ errors
@pytest.mark.parametrize(
    ("z", "kwargs", "match"),
    [
        ([0.0, 1e-3], {}, "each side"),
        ([-1e-3, 0.0, 1e-3], {"derivative": "polyfit", "order": 3}, "at least 4 planes"),
        ([-1e-3, 0.0, 1e-3], {"order": 2}, "polyfit"),
        ([-1e-3, 1e-3], {"method": "multigrid"}, "method"),
        ([-1e-3, -1e-3], {}, "distinct"),
        ([-1e-3, 1e-3], {"wavelength": -1.0}, "wavelength"),
        ([-1e-3, 1e-3], {"pitch": 0.0}, "pitch"),
    ],
)
def test_invalid_input(z: list[float], kwargs: dict, match: str) -> None:
    stack = np.ones((len(z), 16, 16))
    kwargs = {"pitch": PITCH, "wavelength": WAVELENGTH, **kwargs}
    with pytest.raises(ValueError, match=match):
        tie(stack, z, **kwargs)
    with pytest.raises(ValueError, match="3 distances for 2"):
        tie(np.ones((2, 16, 16)), [-1e-3, 0.0, 1e-3], pitch=PITCH, wavelength=WAVELENGTH)


def test_one_sided_stack_with_polyfit(bumps: np.ndarray) -> None:
    z = [0.0, 5e-4, 1e-3]
    res = tie(
        _stack(np.exp(1j * bumps), z),
        z,
        pitch=PITCH,
        wavelength=WAVELENGTH,
        method="fft",
        derivative="polyfit",
    )
    assert _rel_error(res.phase, bumps) < 2e-3


# --------------------------------------------------------------------- GPU
@pytest.mark.parametrize("device", devices())
@pytest.mark.parametrize("method", ["fft", "dct", "pcg"])
def test_device_parity(bumps: np.ndarray, device: str, method: str) -> None:
    field = _nonuniform_amplitude() * np.exp(1j * bumps)
    z = [-1e-3, 0.0, 1e-3]
    stack = _stack(field, z)
    ref = tie(stack, z, pitch=PITCH, wavelength=WAVELENGTH, method=method)
    res = tie(stack, z, pitch=PITCH, wavelength=WAVELENGTH, method=method, device=device)
    assert res.device == device
    host = res.to_numpy()
    assert isinstance(host.phase, np.ndarray) and isinstance(host.didz, np.ndarray)
    # GPU runs in single precision by default.
    np.testing.assert_allclose(host.phase, ref.phase, atol=1e-3 * np.abs(ref.phase).max())


@pytest.mark.parametrize("device", devices())
def test_pcg_aperture_single_precision(device: str) -> None:
    """Float32 CG stays stable (null-space projection) and meets the default tol.

    Without projecting out the per-region constant, float32 CG diverged on the
    GPU after reaching a residual of ~1e-5 at this size.
    """
    amplitude, phase, inner = _aperture(256)
    z = [-1e-3, 0.0, 1e-3]
    stack = _stack(amplitude * np.exp(1j * phase), z)
    res = tie(
        stack,
        z,
        pitch=PITCH,
        wavelength=WAVELENGTH,
        method="pcg",
        device=device,
        precision="single",
    ).to_numpy()
    assert res.phase.dtype == np.float32
    assert res.converged and res.n_iter < 500, res.message
    d = (res.phase - phase)[inner]
    assert np.std(d) < 0.01


@pytest.mark.gpu
def test_gpu_simulation_matches_cpu(bumps: np.ndarray) -> None:
    field = _nonuniform_amplitude() * np.exp(1j * bumps)
    z = [-1e-3, 0.0, 2e-3]
    cpu = _stack(field, z)
    gpu = simulate_defocus_stack(field, z, PITCH, WAVELENGTH, device="gpu", precision="double")
    np.testing.assert_allclose(to_numpy(gpu), cpu, rtol=1e-10, atol=1e-12)


# -------------------------------------------------------------------- slow
@pytest.mark.slow
def test_multi_plane_fit_averages_photon_noise() -> None:
    """With shot noise, a least-squares fit over seven planes beats the three nearest.

    The slope noise of a fit over ``z = d * (-3 .. 3)`` is ``sqrt(14)`` times
    smaller than that of the central difference over ``+-d``.
    """
    rng = np.random.default_rng(3)
    n = 256
    phase = _bumps(n)
    field = _nonuniform_amplitude(n) * np.exp(1j * phase)
    d = 1e-3
    z7 = d * np.arange(-3, 4)
    clean = _stack(field, z7)
    photons = 1e6  # per pixel at unit intensity
    kwargs = {"pitch": PITCH, "wavelength": WAVELENGTH, "method": "pcg"}
    err_c, err_p = [], []
    for _ in range(4):
        noisy = rng.poisson(clean * photons) / photons
        err_c.append(_rel_error(tie(noisy[2:5], z7[2:5], **kwargs).phase, phase))
        err_p.append(
            _rel_error(tie(noisy, z7, derivative="polyfit", order=2, **kwargs).phase, phase)
        )
    # Measured: central ~0.22, seven-plane fit ~0.05.
    assert np.mean(err_p) < 0.1, err_p
    assert np.mean(err_p) < 0.4 * np.mean(err_c), (err_c, err_p)
