from __future__ import annotations

import numpy as np
import pytest

from conftest import devices
from solvephase.backend import get_backend, to_numpy
from solvephase.propagation import (
    AngularSpectrumPropagator,
    FFTPropagator,
    FocalPlanePropagator,
    MFTPropagator,
    centered_coordinates,
)


def _cplx(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _adjoint_gap(op, rng, lead=(2,)) -> float:
    be = op.backend
    x = be.asarray(_cplx(rng, (*lead, *op.in_shape)))
    y = be.asarray(_cplx(rng, (*lead, *op.out_shape)))
    lhs = complex(be.xp.vdot(op.forward(x), y))
    rhs = complex(be.xp.vdot(x, op.adjoint(y)))
    return abs(lhs - rhs) / abs(lhs)


def test_centered_coordinates_are_symmetric() -> None:
    np.testing.assert_allclose(centered_coordinates(4, 2.0), [-3.0, -1.0, 1.0, 3.0])
    np.testing.assert_allclose(centered_coordinates(3), [-1.0, 0.0, 1.0])


@pytest.mark.parametrize(
    ("n", "m", "big", "offset"),
    [
        ((32, 32), (20, 24), 64, (0.0, 0.0)),
        ((31, 30), (17, 16), 64, (0.3, -1.5)),
        ((16, 16), (40, 40), 40, 0.5),
    ],
)
def test_fft_equals_mft_and_both_are_exact_adjoints(n, m, big, offset, rng) -> None:
    fft = FFTPropagator(n, m, big, offset=offset)
    mft = MFTPropagator(n, m, big, offset=offset)
    u = _cplx(rng, (3, *n))
    np.testing.assert_allclose(fft.forward(u), mft.forward(u), atol=1e-12)
    assert _adjoint_gap(fft, rng) < 1e-12
    assert _adjoint_gap(mft, rng) < 1e-12


def test_fft_rejects_windows_larger_than_the_fft() -> None:
    with pytest.raises(ValueError, match="repeat"):
        FFTPropagator(16, 40, 32)
    with pytest.raises(ValueError, match="at least the input"):
        FFTPropagator(64, 16, 32)


def test_stacked_mft_matches_individual_transforms(rng) -> None:
    stack = MFTPropagator(16, 20, [32.0, 40.5], stack=True)
    u = _cplx(rng, (2, 2, 16, 16))
    out = stack.forward(u)
    for i, n in enumerate([32.0, 40.5]):
        np.testing.assert_allclose(out[:, i], MFTPropagator(16, 20, n).forward(u[:, i]), atol=1e-12)
    assert _adjoint_gap(stack, rng, lead=(3, 2)) < 1e-12


def test_parseval_when_the_window_captures_everything() -> None:
    n = 32
    x = centered_coordinates(n)
    pupil = (np.hypot(x[:, None], x[None, :]) < 12).astype(float)
    prop = FocalPlanePropagator((n, n), 0.1, 1.0, 1.0 / (0.1 * 128), (128, 128))
    assert prop.method == "fft"
    e = prop.forward(pupil[None])
    assert np.isclose(np.sum(np.abs(e) ** 2), np.sum(pupil**2))


def test_tilt_moves_the_image_towards_positive_x() -> None:
    n, lam, pitch = 64, 1.0, 0.1
    pixel = lam / (pitch * 256)
    x = centered_coordinates(n, pitch)
    pupil = (np.hypot(x[:, None], x[None, :]) < 3.0).astype(float)
    prop = FocalPlanePropagator((n, n), pitch, lam, pixel, (64, 64))
    for shift_x, shift_y in [(5, 0), (0, -3), (2, 4)]:
        opd = shift_x * pixel * x[None, :] + shift_y * pixel * x[:, None]
        u = pupil * np.exp(2j * np.pi * opd / lam)
        img = np.abs(prop.forward(u[None])[0]) ** 2
        yy, xx = np.mgrid[0:64, 0:64]
        cy = np.sum(img * yy) / img.sum() - 31.5
        cx = np.sum(img * xx) / img.sum() - 31.5
        # Window truncation biases the centroid slightly.
        assert abs(cx - shift_x) < 0.1 and abs(cy - shift_y) < 0.1


def test_auto_engine_choice_and_broadband_uses_mft() -> None:
    mono = FocalPlanePropagator(32, 0.1, 1.0, 1.0 / (0.1 * 64), 32)
    assert mono.method == "fft" and mono.n_wavelengths == 1
    odd = FocalPlanePropagator(32, 0.1, 1.0, 1.0 / (0.1 * 64.3), 32)
    assert odd.method == "mft"
    broad = FocalPlanePropagator(32, 0.1, [1.0, 1.1], 1.0 / (0.1 * 64), 32)
    assert broad.method == "mft" and broad.n_wavelengths == 2
    with pytest.raises(ValueError, match="integer"):
        FocalPlanePropagator(32, 0.1, 1.0, 1.0 / (0.1 * 64.3), 32, method="fft")


def test_angular_spectrum_is_identity_at_zero_distance_and_adjoint(rng) -> None:
    n = 32
    prop = AngularSpectrumPropagator(n, 5e-6, 1e-6, [0.0, 1e-4, -2e-4])
    u = _cplx(rng, (n, n))
    out = prop.forward(u)
    np.testing.assert_allclose(out[0], u, atol=1e-12)
    y = _cplx(rng, (3, n, n))
    lhs = np.vdot(prop.forward(u), y)
    rhs = np.vdot(u, prop.adjoint(y))
    assert abs(lhs - rhs) < 1e-12 * abs(lhs)
    # Forward then backward propagation by the same distance is the identity.
    there = AngularSpectrumPropagator(n, 5e-6, 1e-6, 3e-4)
    back = AngularSpectrumPropagator(n, 5e-6, 1e-6, -3e-4)
    np.testing.assert_allclose(back.forward(there.forward(u)), u, atol=1e-10)


@pytest.mark.parametrize("device", devices())
@pytest.mark.parametrize("precision", ["single", "double"])
def test_precision_and_device_parity(device: str, precision: str, rng) -> None:
    be = get_backend(device, precision)
    ref = FocalPlanePropagator(24, 0.1, [1.0, 1.2], 0.15, 20)
    prop = FocalPlanePropagator(24, 0.1, [1.0, 1.2], 0.15, 20, backend=be)
    u = _cplx(rng, (2, 24, 24))
    out = prop.forward(be.asarray(u, dtype="complex"))
    assert out.dtype == be.complex_dtype
    tol = 1e-4 if precision == "single" else 1e-12
    np.testing.assert_allclose(
        to_numpy(out), ref.forward(u), atol=tol * np.abs(ref.forward(u)).max()
    )
