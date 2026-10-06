"""Wirtinger-flow family: recovery from |A x|^2 for Gaussian and coded-diffraction models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from conftest import devices
from solvephase.algorithms.wirtinger import (
    GenericResult,
    init_optimal_spectral,
    init_orthogonality_promoting,
    init_random,
    init_spectral,
    init_truncated_spectral,
    init_weighted_correlation,
    relative_error,
    wirtinger,
)
from solvephase.backend import get_backend, to_numpy
from solvephase.operators import CodedDiffractionOperator, LinearOperator, MatrixOperator

METHODS = ["wf", "twf", "taf", "raf", "lbfgs"]
INITS = {
    "spectral": init_spectral,
    "truncated": init_truncated_spectral,
    "orthogonal": init_orthogonality_promoting,
    "weighted": init_weighted_correlation,
    "optimal": init_optimal_spectral,
}


def _signal(shape: Any, seed: int) -> np.ndarray:
    rng = np.random.default_rng(10_000 + seed)
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _gaussian(n: int, m: int, seed: int, backend: Any = "cpu") -> tuple[Any, np.ndarray, Any]:
    x = _signal(n, seed)
    op = MatrixOperator.gaussian(m, n, seed=seed, backend=backend)
    y = np.abs(to_numpy(op.matrix) @ x) ** 2
    return op, x, y


def _cdp(size: int, masks: int, seed: int, backend: Any = "cpu") -> tuple[Any, np.ndarray, Any]:
    x = _signal((size, size), seed)
    op = CodedDiffractionOperator.random(size, masks, seed=seed, backend=backend)
    y = to_numpy(np.abs(to_numpy(op.forward(op.backend.asarray(x, dtype="complex")))) ** 2)
    return op, x, y


def _cosine(z: Any, x: np.ndarray) -> float:
    z = to_numpy(z)
    return float(abs(np.vdot(z, x)) / (np.linalg.norm(z) * np.linalg.norm(x)))


# ------------------------------------------------------------------ helpers
def test_relative_error_removes_the_global_phase(rng: np.random.Generator):
    x = _signal(50, 1)
    assert relative_error(np.exp(1.234j) * x, x) == pytest.approx(0.0, abs=1e-14)
    assert relative_error(-x, x) == pytest.approx(0.0, abs=1e-14)
    noise = 1e-3 * _signal(50, 2)
    # Phase-aligned distance is never larger than the plain distance, and equal here.
    assert relative_error(x + noise, x) <= np.linalg.norm(noise) / np.linalg.norm(x) + 1e-15
    assert relative_error(np.exp(0.7j) * (x + noise), x) == pytest.approx(
        relative_error(x + noise, x), rel=1e-10
    )
    assert relative_error(np.zeros(50), x) == pytest.approx(1.0)


# --------------------------------------------------------------- recovery
@pytest.mark.parametrize("method", METHODS)
def test_gaussian_model_recovery(method: str):
    """n = 64, m = 8n, noise-free: every seed reaches relative error < 1e-5."""
    for seed in range(3):
        op, x, y = _gaussian(64, 512, seed)
        result = wirtinger(op, y, method=method, seed=seed, iterations=3000)
        assert result.converged, result.message
        assert relative_error(result.x, x) < 1e-5
        assert result.history[-1] < 1e-6 < result.history[0]
        assert result.init == {"wf": "truncated", "twf": "truncated", "taf": "orthogonal"}.get(
            method, {"raf": "weighted", "lbfgs": "optimal"}.get(method)
        )


@pytest.mark.slow
@pytest.mark.parametrize("method", METHODS)
def test_gaussian_model_success_rate(method: str):
    """At m = 6n, at least 9 of 10 random problems are solved by every method."""
    successes = 0
    for seed in range(10):
        op, x, y = _gaussian(64, 384, 100 + seed)
        result = wirtinger(op, y, method=method, seed=seed, iterations=3000)
        successes += relative_error(result.x, x) < 1e-5
    assert successes >= 9


def test_lbfgs_intensity_loss_and_magnitude_input():
    op, x, y = _gaussian(64, 384, 7)
    intensity = wirtinger(op, y, method="lbfgs", loss="intensity", seed=0)
    assert relative_error(intensity.x, x) < 1e-6
    magnitudes = wirtinger(op, np.sqrt(y), magnitudes=True, method="lbfgs", seed=0)
    assert relative_error(magnitudes.x, x) < 1e-6


@pytest.mark.parametrize("method", ["raf", "lbfgs"])
def test_coded_diffraction_recovery(method: str):
    op, x, y = _cdp(32, 6, 0)
    result = wirtinger(op, y, method=method, seed=0, iterations=2000)
    assert relative_error(result.x, x) < 1e-5


@pytest.mark.slow
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("masks", [6, 8])
def test_coded_diffraction_all_methods(method: str, masks: int):
    for seed in range(2):
        op, x, y = _cdp(32, masks, seed)
        result = wirtinger(op, y, method=method, seed=seed, iterations=3000)
        assert relative_error(result.x, x) < 1e-5


def test_custom_operator_without_exact_frobenius_norm():
    """A user operator relying on the Hutchinson norm estimate still converges."""

    class Wrapped(LinearOperator):
        def __init__(self, inner: MatrixOperator) -> None:
            self.inner = inner
            self.backend = inner.backend
            self.in_shape, self.out_shape = inner.in_shape, inner.out_shape

        def forward(self, x: Any) -> Any:
            return 3.0 * self.inner.forward(x)

        def adjoint(self, y: Any) -> Any:
            return 3.0 * self.inner.adjoint(y)

    inner, x, y = _gaussian(48, 384, 3)
    result = wirtinger(Wrapped(inner), 9.0 * y, method="raf", seed=1)
    assert relative_error(result.x, x) < 1e-5
    # The scale of x is recovered, not only its direction.
    assert np.linalg.norm(result.x) == pytest.approx(np.linalg.norm(x), rel=1e-5)


# ------------------------------------------------------------ initializations
def test_spectral_inits_correlate_and_improve_with_m():
    means: dict[tuple[str, int], float] = {}
    for ratio in (4, 16):
        for name, init in INITS.items():
            sims = []
            for seed in range(4):
                op, x, y = _gaussian(64, ratio * 64, seed)
                z = init(op, y, seed=seed)
                sims.append(_cosine(z, x))
                # Norm estimate sqrt(mean y) is within 10 % of ||x||.
                assert np.linalg.norm(to_numpy(z)) == pytest.approx(np.linalg.norm(x), rel=0.1)
            means[name, ratio] = float(np.mean(sims))
            assert min(sims) > (0.45 if ratio == 4 else 0.75)
    for name in INITS:
        assert means[name, 16] > means[name, 4] + 0.15
    # Optimal preprocessing (Luo et al. 2019) beats the plain spectral method clearly.
    for ratio in (4, 16):
        assert means["optimal", ratio] > means["spectral", ratio] + 0.05


@pytest.mark.slow
def test_large_cdp_inits_are_whitened():
    """256 x 256, 8 octanary masks: without diag(A^H A) whitening, eigenvectors localized
    on pixels where most masks have |d|^2 = 3 beat the signal (cosine ~0.03)."""
    op, x, y = _cdp(256, 8, 1)
    for name in ("truncated", "orthogonal", "weighted", "optimal"):
        assert _cosine(INITS[name](op, y, seed=0), x) > 0.65, name
    result = wirtinger(op, y, method="raf", seed=0)
    assert relative_error(result.x, x) < 1e-5


def test_random_init_is_uncorrelated_and_reproducible():
    op, x, y = _gaussian(64, 512, 0)
    a = init_random(op, y, seed=99)
    np.testing.assert_array_equal(a, init_random(op, y, seed=99))
    assert _cosine(a, x) < 0.4


def test_optimal_init_needs_more_measurements_than_unknowns():
    op, _, y = _gaussian(64, 64, 0)
    with pytest.raises(ValueError, match="more measurements"):
        init_optimal_spectral(op, y)


# ------------------------------------------------------------------- noise
@pytest.mark.parametrize("method", METHODS)
def test_error_is_proportional_to_noise(method: str):
    """Relative intensity noise ``y (1 + sigma n)`` gives an error linear in sigma."""
    op, x, y = _gaussian(64, 512, 5)
    noise = np.random.default_rng(3).standard_normal(y.shape)
    errors = []
    for sigma in (1e-4, 1e-3, 1e-2):
        result = wirtinger(op, y * (1.0 + sigma * noise), method=method, seed=0)
        assert result.converged, result.message
        errors.append(relative_error(result.x, x))
        assert 0.2 * sigma < errors[-1] < 2.0 * sigma
    assert errors[1] / errors[0] == pytest.approx(10.0, rel=0.05)
    assert errors[2] / errors[1] == pytest.approx(10.0, rel=0.05)


# ------------------------------------------------------------- run control
def test_given_start_callback_and_result_fields():
    op, x, y = _gaussian(64, 512, 2)
    start = x + 0.1 * _signal(64, 3)
    calls: list[tuple[int, float]] = []

    def callback(it: int, z: Any, residual: float) -> bool:
        calls.append((it, residual))
        return it >= 20

    result = wirtinger(op, y, method="taf", init=start, check_every=5, callback=callback)
    assert isinstance(result, GenericResult)
    assert result.init == "given" and result.method == "taf"
    assert [c[0] for c in calls] == [5, 10, 15, 20]
    assert result.n_iter == 20 and result.message == "stopped by callback"
    assert len(result.history) == len(result.times) == 5
    assert result.history[-1] < result.history[0]
    assert result.residual == result.history[-1]
    host = result.to_numpy()
    assert isinstance(host.x, np.ndarray) and host.x.shape == (64,)


def test_iteration_limit_reports_not_converged():
    op, _, y = _gaussian(64, 512, 2)
    result = wirtinger(op, y, method="wf", iterations=7, check_every=5)
    assert not result.converged
    assert result.n_iter == 7 and len(result.history) == 3
    assert "limit" in result.message


@pytest.mark.parametrize("precision", ["single", "double"])
def test_precision_is_kept(precision: str):
    be = get_backend("cpu", precision)
    op, x, y = _gaussian(64, 512, 1, backend=be)
    result = wirtinger(op, y.astype(be.real_dtype), method="raf", seed=0)
    assert result.x.dtype == be.complex_dtype
    assert relative_error(result.x, x) < (1e-5 if precision == "single" else 1e-6)


def test_invalid_arguments_raise():
    op, _, y = _gaussian(16, 64, 0)
    with pytest.raises(ValueError, match="method"):
        wirtinger(op, y, method="gs")
    with pytest.raises(ValueError, match="init"):
        wirtinger(op, y, init="bogus")
    with pytest.raises(ValueError, match="unknown option"):
        wirtinger(op, y, method="raf", options={"gamma": 1.0})
    with pytest.raises(ValueError, match="line search"):
        wirtinger(op, y, method="lbfgs", step=0.5)
    with pytest.raises(ValueError, match="shape"):
        wirtinger(op, y[:-1])
    with pytest.raises(ValueError, match="complex"):
        wirtinger(op, y.astype(complex))
    with pytest.raises(ValueError, match="shape"):
        wirtinger(op, y, init=np.ones(3))
    with pytest.raises(ValueError, match="zero"):
        wirtinger(op, np.zeros_like(y))
    with pytest.raises(TypeError, match="MatrixOperator"):
        wirtinger(np.ones((64, 16)), y)  # type: ignore[arg-type]


# --------------------------------------------------------------------- GPU
@pytest.mark.parametrize("device", devices())
def test_device_parity(device: str):
    """Same seed, same start on CPU and GPU; both recover to their precision."""
    be = get_backend(device)
    op, x, y = _gaussian(64, 512, 4, backend=be)
    cpu_op, _, _ = _gaussian(64, 512, 4)
    z_dev = init_weighted_correlation(op, y, seed=1)
    z_cpu = init_weighted_correlation(cpu_op, y, seed=1)
    assert relative_error(to_numpy(z_dev), z_cpu) < 1e-4
    for method in METHODS:
        result = wirtinger(op, y, method=method, seed=1, iterations=3000)
        assert result.device == be.device
        assert result.x.dtype == be.complex_dtype
        assert relative_error(result.x, x) < (1e-5 if be.precision == "single" else 1e-6)


@pytest.mark.gpu
@pytest.mark.parametrize("method", METHODS)
def test_gpu_coded_diffraction(method: str):
    be = get_backend("gpu")
    op, x, y = _cdp(32, 8, 1, backend=be)
    result = wirtinger(op, be.asarray(y), method=method, seed=1, iterations=3000)
    assert type(result.x) is not np.ndarray  # stays on the device
    assert relative_error(result.x, x) < 1e-5
