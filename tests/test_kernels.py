"""Fast paths must reproduce the plain array expressions they replace.

The GPU kernels in ``solvephase._kernels`` fuse chains of CuPy operations; the
CPU paths avoid complex ``exp`` and dead Taylor terms. Each test evaluates the
original expression with plain ``xp`` operations and compares. On the
reference platform (CuPy 14, CUDA 12) the GPU results agree to the bit; the
tolerances leave room for compilers that contract multiply-adds differently.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import solvephase as sp
from conftest import devices
from solvephase import _kernels
from solvephase.focal import _unit_phasor
from solvephase.losses import AmplitudeLoss, GaussianLoss, PoissonLoss

LAM = 1.0e-6


def _close(actual: object, expected: object, rtol: float) -> None:
    a = np.asarray(sp.to_numpy(actual))
    e = np.asarray(sp.to_numpy(expected))
    assert a.shape == e.shape
    scale = max(float(np.max(np.abs(e))), 1e-300)
    assert float(np.max(np.abs(a - e))) <= rtol * scale


def _model(device: str, wavelengths: int = 1, oversample: int = 1) -> sp.FocalPlaneModel:
    pupil = sp.Pupil.circular(32, 1.0, obscuration=0.15)
    lams = np.linspace(0.9, 1.1, wavelengths) * LAM if wavelengths > 1 else LAM
    div = sp.zernike_diversity(pupil, 4, [0.0, 0.3 * LAM])
    return sp.FocalPlaneModel(
        pupil, lams, 32, sampling=2.0, diversity=div, oversample=oversample, device=device
    )


@pytest.mark.parametrize("device", devices())
@pytest.mark.parametrize(("wavelengths", "oversample"), [(1, 1), (3, 1), (1, 2)])
def test_focal_model_matches_plain_expressions(
    device: str, wavelengths: int, oversample: int
) -> None:
    model = _model(device, wavelengths, oversample)
    be, xp = model.backend, model.backend.xp
    rng = np.random.default_rng(1)
    phase = be.asarray(rng.standard_normal(model.pupil.shape) * model.pupil.mask, dtype="real")
    state = model.forward(phase)
    rtol = 10 * be.eps
    # forward: images = sum_L w |P(amp exp(i (phase + div) ratio))|^2 / norm
    phi = (phase[None] + model._div)[:, None] * model._ratio
    u = model._amp * xp.exp(1j * phi).astype(be.complex_dtype)
    e = model.propagator.forward(u)
    inten = xp.sum(model._wl * (e.real**2 + e.imag**2), axis=-3) / model._norm
    _close(state.pupil_field, u, rtol)
    _close(state.images, model._bin(inten), rtol)
    # vjp: grad = sum_L Im(conj(u) P^H(g E 2 w / norm)) ratio
    g = be.asarray(rng.standard_normal(state.images.shape), dtype="real")
    grad, _ = model.vjp(state, g)
    v = model.propagator.adjoint(
        model._unbin(g)[:, None] * state.focal_field * (model._wl * (2.0 / model._norm))
    )
    ref = xp.sum((xp.conj(state.pupil_field) * v).imag * model._ratio, axis=1)
    _close(grad, ref, 100 * be.eps)
    # jvp: d images = sum_L w Re(conj(E) P(i d ratio u)) 2 / norm
    d = be.asarray(rng.standard_normal((3, *model.pupil.shape)), dtype="real")
    du = (1j * d[:, None, None] * model._ratio) * state.pupil_field
    de = model.propagator.forward(du.astype(be.complex_dtype))
    ref = xp.sum(model._wl * (xp.conj(state.focal_field) * de).real, axis=-3) * (2 / model._norm)
    _close(model.jvp(state, d), model._bin(ref), 100 * be.eps)


def test_unit_phasor_is_complex_exp(rng: np.random.Generator) -> None:
    be = sp.get_backend("cpu")
    phi = rng.standard_normal((2, 3, 17, 19)) * 50
    np.testing.assert_array_equal(_unit_phasor(be, phi), np.exp(1j * phi))


def _loss_inputs(device: str, rng: np.random.Generator) -> tuple[object, ...]:
    be = sp.get_backend(device)
    data = rng.poisson(50 * rng.random((2, 16, 16)) ** 4).astype(np.float64)
    model = data + rng.standard_normal(data.shape) * 3
    model[0, :2] = -1.0  # below the floor: the Taylor branch
    model[1, 0, 0] = 1e-9
    weights = rng.random(data.shape)
    weights[0, 5] = 0.0
    return (
        be,
        be.asarray(model, dtype="real"),
        be.asarray(data, dtype=np.float64),
        be.asarray(weights, dtype=np.float64),
    )


@pytest.mark.parametrize(
    "loss",
    [
        GaussianLoss(),
        PoissonLoss(),
        PoissonLoss(read_noise=2.0),
        PoissonLoss(floor=0.5),
        AmplitudeLoss(),
        AmplitudeLoss(floor=0.2),
    ],
    ids=["gauss", "poisson", "poisson-rn", "poisson-floor", "amplitude", "amplitude-floor"],
)
@pytest.mark.parametrize("device", devices())
def test_fused_losses_match_evaluate(loss: object, device: str, rng: np.random.Generator) -> None:
    be, model, data, weights = _loss_inputs(device, rng)
    assert isinstance(loss, (GaussianLoss, PoissonLoss, AmplitudeLoss))
    value, grad, curv = loss.evaluate(be, model.astype(np.float64), data, weights)
    fused = loss._evaluate_fused(be, model, data, weights, be.real_dtype)
    if not be.is_gpu:
        assert fused is None
        return
    assert fused is not None
    f_value, f_grad, f_curv = fused
    assert float(f_value) == pytest.approx(float(value), rel=1e-14)
    assert f_grad.dtype == be.real_dtype
    _close(f_grad, grad.astype(be.real_dtype), 1e-6)
    _close(f_curv, xp_broadcast(be, curv, model.shape), 1e-14)


def xp_broadcast(be: sp.Backend, a: object, shape: tuple[int, ...]) -> object:
    return be.xp.broadcast_to(a, shape)


@pytest.mark.parametrize("device", devices())
def test_poisson_taylor_skip_is_exact(device: str, rng: np.random.Generator) -> None:
    """With nothing below the floor the CPU shortcut equals the full expression."""
    be = sp.get_backend(device)
    xp = be.xp
    data = be.asarray(rng.poisson(30, (2, 8, 8)).astype(float), dtype=np.float64)
    model = data + 0.5
    weights = be.asarray(rng.random((2, 8, 8)), dtype=np.float64)
    value, grad, curv = PoissonLoss().evaluate(be, model, data, weights)
    m, d = model, data
    f = (m - d) + d * xp.log(xp.where(d > 0, d / m, 1.0))
    assert float(value) == float(xp.sum(weights * (f + 0.0), dtype=np.float64))
    assert bool(xp.all(grad == weights * (1.0 - d / m)))
    assert bool(xp.all(curv == weights / m))


@pytest.mark.parametrize("device", devices())
def test_dot_matches_backend(device: str, rng: np.random.Generator) -> None:
    be = sp.get_backend(device)
    for dtype in ("real", "complex"):
        a = be.asarray(rng.standard_normal(3001), dtype=dtype)
        b = be.asarray(rng.standard_normal(3001), dtype=dtype)
        assert _kernels.dot(be, a, b) == pytest.approx(be.dot(a, b), rel=1e-6)


@pytest.mark.gpu
def test_fast_furious_graphs_match_eager() -> None:
    """CUDA-graph replay runs the eager kernels: closed loops agree to the bit."""
    pupil = sp.Pupil.circular(32, 1.0)
    truth = sp.random_aberration(pupil, 0.06 * LAM, seed=3)
    runs, sensors = [], []
    for graphs in (True, False):
        ff = sp.FastAndFurious(pupil, LAM, 32, sampling=2.0, device="gpu")
        ff._use_graphs = graphs
        res = sp.simulate_closed_loop(ff, truth, 8, photons=1e7, seed=4)
        image = np.abs(sp.to_numpy(res.residual_opd)) * 1e7  # any fixed frame
        extra = [ff.step(image + 1.0), ff.step(image + 2.0)]  # no DM change, twice
        ff.reset()
        extra += [ff.step(image + 1.0), ff.step(image + 1.5, None)]  # first frame again
        runs.append([res.residual_rms, res.dm_opd, ff.last_odd_opd, ff.last_even_opd, *extra])
        sensors.append(ff)
    assert sensors[0]._graphs is not None
    assert len(sensors[0]._graphs.graphs) == 3  # first, still and change were all replayed
    assert sensors[1]._graphs is None
    for a, b in zip(*runs):
        np.testing.assert_array_equal(sp.to_numpy(a), sp.to_numpy(b))


@pytest.mark.gpu
def test_fast_furious_graphs_accept_device_and_integer_images() -> None:
    pupil = sp.Pupil.circular(32, 1.0)
    model = sp.FocalPlaneModel(pupil, LAM, 32, sampling=2.0, device="gpu")
    image = sp.to_numpy(model.images(sp.random_aberration(pupil, 0.05 * LAM, seed=1)))[0]
    frame = np.round(image * 1e5).astype(np.uint16)
    out = []
    for graphs in (True, False):
        ff = sp.FastAndFurious(pupil, LAM, 32, sampling=2.0, device="gpu")
        ff._use_graphs = graphs
        zero = np.zeros(pupil.shape)
        steps = [ff.step(frame), ff.step(frame, zero), ff.step(frame, zero)]
        steps.append(ff.step(model.backend.asarray(frame[None]), model.backend.asarray(zero)))
        out.append(steps)
    for a, b in zip(*out):
        np.testing.assert_array_equal(sp.to_numpy(a), sp.to_numpy(b))
    with pytest.raises(ValueError, match="does not match"):
        ff.step(np.zeros((8, 8)), np.zeros(pupil.shape))


@pytest.mark.gpu
def test_gerchberg_saxton_gpu_matches_plain_iteration() -> None:
    """The fused modulus projection equals the unfused update on the GPU."""
    model = _model("gpu")
    be, xp = model.backend, model.backend.xp
    images = sp.simulate_images(
        model, sp.random_aberration(model.pupil, 0.08 * LAM, seed=1), photons=1e6, seed=2
    )
    rng = np.random.default_rng(3)
    win = be.asarray(rng.standard_normal((2, 32, 32)) + 1j * rng.standard_normal((2, 32, 32)))
    win = win.astype(be.complex_dtype)
    measured = be.asarray(rng.random((2, 32, 32)) > 0.1)
    sqrt_signal = xp.sqrt(be.asarray(images, dtype="real"))
    scale = be.asarray([0.7, 1.3], dtype="real")[:, None, None]
    mag = xp.abs(win)
    ref = xp.where(measured, sqrt_signal * scale * win / xp.maximum(mag, 1e-30), win)
    _close(_kernels.call("gs_modulus", measured, sqrt_signal, scale, win), ref, 1e-6)
    power = be.empty(win.shape)
    _kernels.call("gs_power", win, measured, power)
    _close(power, (mag * mag) * measured, 1e-6)
    result = sp.gerchberg_saxton(model, images, iterations=30, unwrap=False)
    assert math.isfinite(result.loss)
