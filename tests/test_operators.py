"""Generic measurement operators: exact adjoints, norms, mask statistics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from conftest import devices
from solvephase.backend import get_backend, to_numpy
from solvephase.operators import (
    CodedDiffractionOperator,
    LinearOperator,
    MatrixOperator,
    OversampledFourierOperator,
)


def _make(kind: str, backend: Any) -> LinearOperator:
    if kind == "matrix":
        return MatrixOperator.gaussian(40, 12, seed=1, backend=backend)
    if kind == "cdp-octanary":
        return CodedDiffractionOperator.random((6, 8), 3, seed=2, backend=backend)
    if kind == "cdp-uniform":
        return CodedDiffractionOperator.random(7, 2, kind="uniform", seed=3, backend=backend)
    return OversampledFourierOperator((5, 6), (2.0, 1.5), backend=backend)


KINDS = ["matrix", "cdp-octanary", "cdp-uniform", "oversampled"]


def _crandn(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _dense(op: LinearOperator) -> np.ndarray:
    """Dense matrix of ``op`` (columns are images of basis vectors)."""
    be = op.backend
    eye = np.eye(op.in_size).reshape(op.in_size, *op.in_shape)
    out = to_numpy(op.forward(be.asarray(eye, dtype="complex")))
    return out.reshape(op.in_size, op.out_size).T


@pytest.mark.parametrize("device", devices())
@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("kind", KINDS)
def test_adjoint_is_exact(kind: str, device: str, precision: str, rng: np.random.Generator):
    be = get_backend(device, precision)
    op = _make(kind, be)
    x = be.asarray(_crandn(rng, op.in_shape), dtype="complex")
    y = be.asarray(_crandn(rng, op.out_shape), dtype="complex")
    ax, aty = op.forward(x), op.adjoint(y)
    assert ax.shape == op.out_shape and aty.shape == op.in_shape
    assert ax.dtype == be.complex_dtype and aty.dtype == be.complex_dtype
    lhs = complex(be.xp.vdot(ax, y))
    rhs = complex(be.xp.vdot(x, aty))
    scale = float(be.xp.linalg.norm(ax)) * float(be.xp.linalg.norm(y))
    tol = 1e-12 if precision == "double" else 1e-5
    assert abs(lhs - rhs) <= tol * scale


@pytest.mark.parametrize("kind", KINDS)
def test_batched_application_matches_loop(kind: str, rng: np.random.Generator):
    op = _make(kind, "cpu")
    xs = _crandn(rng, (3, *op.in_shape))
    ys = _crandn(rng, (3, *op.out_shape))
    fwd, adj = op.forward(xs), op.adjoint(ys)
    for i in range(3):
        np.testing.assert_allclose(fwd[i], op.forward(xs[i]), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(adj[i], op.adjoint(ys[i]), rtol=1e-12, atol=1e-12)


def test_large_matrix_uses_blas_path_consistently(rng: np.random.Generator):
    # 600 x 500 is above the einsum threshold; the BLAS branch must agree with dense algebra.
    op = MatrixOperator(_crandn(rng, (600, 500)))
    x, y = _crandn(rng, (500,)), _crandn(rng, (600,))
    np.testing.assert_allclose(op.forward(x), op.matrix @ x, rtol=1e-10)
    np.testing.assert_allclose(op.adjoint(y), op.matrix.conj().T @ y, rtol=1e-10)


@pytest.mark.parametrize("kind", KINDS)
def test_frobenius_and_spectral_norms(kind: str):
    op = _make(kind, "cpu")
    dense = _dense(op)
    exact_fro = float(np.sum(np.abs(dense) ** 2))
    assert op.frobenius_norm_squared() == pytest.approx(exact_fro, rel=1e-12)
    gram = np.sum(np.abs(dense) ** 2, axis=0).reshape(op.in_shape)
    np.testing.assert_allclose(op.gram_diagonal(), gram, rtol=1e-12)
    assert LinearOperator.gram_diagonal(op) is None
    # The generic Hutchinson estimate (base class) is unbiased; 512 probes give ~5 %.
    hutchinson = LinearOperator.frobenius_norm_squared(op, probes=512, seed=4)
    assert hutchinson == pytest.approx(exact_fro, rel=0.1)
    spectral = float(np.linalg.norm(dense, 2))
    assert op.norm_estimate(iterations=500, tol=1e-10) == pytest.approx(spectral, rel=1e-4)


def test_oversampled_fourier_is_an_isometry(rng: np.random.Generator):
    op = OversampledFourierOperator((8, 8), 2)
    assert op.out_shape == (16, 16)
    x = _crandn(rng, (8, 8))
    np.testing.assert_allclose(op.adjoint(op.forward(x)), x, atol=1e-12)
    assert np.linalg.norm(op.forward(x)) == pytest.approx(np.linalg.norm(x), rel=1e-12)
    # |FFT| of the padded image is the oversampled Fourier magnitude.
    np.testing.assert_allclose(np.abs(op.forward(x)), np.abs(np.fft.fft2(x, s=(16, 16))) / 16)


def test_octanary_mask_statistics():
    op = CodedDiffractionOperator.random(64, 8, seed=0)
    d = op.masks
    magnitudes = np.abs(d)
    assert set(np.round(np.unique(magnitudes), 12)) == {
        round(1 / np.sqrt(2), 12),
        round(np.sqrt(3), 12),
    }
    phases = np.angle(d) / (np.pi / 2)
    np.testing.assert_allclose(phases, np.round(phases), atol=1e-12)
    # P(|d| = sqrt 3) = 1/5 and E|d|^2 = 1 (Candès et al. 2015).
    assert np.mean(magnitudes > 1) == pytest.approx(0.2, abs=0.01)
    assert np.mean(magnitudes**2) == pytest.approx(1.0, abs=0.02)
    uniform = CodedDiffractionOperator.random(16, 2, kind="uniform", seed=0)
    np.testing.assert_allclose(np.abs(uniform.masks), 1.0)


def test_random_operators_are_reproducible_from_the_seed():
    a = MatrixOperator.gaussian(10, 4, seed=5)
    b = MatrixOperator.gaussian(10, 4, seed=5)
    np.testing.assert_array_equal(a.matrix, b.matrix)
    # E |A_kj|^2 = 1
    big = MatrixOperator.gaussian(400, 100, seed=0)
    assert np.mean(np.abs(big.matrix) ** 2) == pytest.approx(1.0, abs=0.02)


@pytest.mark.gpu
def test_gpu_operators_match_cpu(rng: np.random.Generator):
    for kind in ("matrix", "cdp-octanary", "oversampled"):
        cpu = _make(kind, get_backend("cpu"))
        gpu = _make(kind, get_backend("gpu", "double"))
        x = _crandn(rng, cpu.in_shape)
        y = _crandn(rng, cpu.out_shape)
        np.testing.assert_allclose(
            to_numpy(gpu.forward(gpu.backend.asarray(x))), cpu.forward(x), atol=1e-12
        )
        np.testing.assert_allclose(
            to_numpy(gpu.adjoint(gpu.backend.asarray(y))), cpu.adjoint(y), atol=1e-12
        )


def test_invalid_arguments_raise():
    with pytest.raises(ValueError, match="two-dimensional"):
        MatrixOperator(np.ones(3))
    with pytest.raises(ValueError, match="masks"):
        CodedDiffractionOperator(np.ones((4, 4)))
    with pytest.raises(ValueError, match="kind"):
        CodedDiffractionOperator.random(4, 2, kind="binary")
    with pytest.raises(ValueError, match="at least"):
        OversampledFourierOperator((8, 8), out_shape=(4, 8))
    with pytest.raises(ValueError, match=">= 1"):
        OversampledFourierOperator((8, 8), 0.5)
    with pytest.raises(ValueError, match="positive"):
        MatrixOperator.gaussian(0, 3)
