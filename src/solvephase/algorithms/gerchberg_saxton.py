"""Gerchberg-Saxton and its multi-plane (Misell) form for focal-plane data.

The pupil amplitude is known, the image intensities are measured, and the
algorithm alternates projections between the two (Gerchberg & Saxton, Optik
35, 237, 1972). With several diversity channels it is Misell's multi-plane
algorithm (J. Phys. D 6, L6, 1973) in its parallel form: each channel's
modulus-corrected field is propagated back, its diversity removed, and the
pupil estimates are averaged as phasors (as in JWST's Hybrid Diversity
Algorithm, Dean et al. 2006).

The modulus projection keeps the model field wherever the data are not
measured (outside the detector window when the engine is an FFT, or at
zero-weight pixels), which is the exact projection onto the measurement set.
The result is a wrapped phase; it is unwrapped (weighted least squares) and
optionally fitted to a modal basis. GS is fast and has a wide capture range,
which makes it the standard initializer for nonlinear optimization.
"""

from __future__ import annotations

import math
import time
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np

from ..basis import Basis
from ..focal import FocalPlaneModel
from ..propagation import FFTPropagator, FocalPlanePropagator, Propagator
from ..result import Result
from ..retrieval import FocalPlaneProblem
from ..unwrap import unwrap_phase

__all__ = ["gerchberg_saxton"]


class _GSPlan:
    """Monochromatic propagation plan used by the iterations."""

    def __init__(self, model: FocalPlaneModel) -> None:
        be = model.backend
        xp = be.xp
        self.backend = be
        os_ = model.oversample
        my, mx = model.image_shape
        self.window = (my * os_, mx * os_)
        samples = model.wavelength / (model.pupil.pitch * model.pixel_scale / os_)
        n_fft = round(samples)
        offset = (model.offset[0] * os_, model.offset[1] * os_)
        self.full = (
            abs(samples - n_fft) < 1e-9 * samples
            and n_fft >= max(model.pupil.shape)
            and n_fft >= max(self.window)
        )
        self.prop: Propagator
        if self.full:
            # Propagate to the whole FFT grid; locate the detector window inside it.
            starts, full_offset = [], []
            for axis in range(2):
                s = (n_fft - self.window[axis]) // 2
                starts.append(s)
                full_offset.append(offset[axis] + (n_fft - self.window[axis]) / 2.0 - s)
            self.prop = FFTPropagator(
                model.pupil.shape, (n_fft, n_fft), n_fft, offset=tuple(full_offset), backend=be
            )
            self.slices = (
                slice(starts[0], starts[0] + self.window[0]),
                slice(starts[1], starts[1] + self.window[1]),
            )
        else:
            self.prop = FocalPlanePropagator(
                model.pupil.shape,
                model.pupil.pitch,
                [model.wavelength],
                model.pixel_scale / os_,
                self.window,
                offset=offset,
                method="mft",
                backend=be,
            )
            self.slices = (slice(None), slice(None))
        self.xp = xp

    def forward(self, u: Any) -> Any:
        if self.full:
            return self.prop.forward(u)
        return self.prop.forward(u[:, None])[:, 0]

    def adjoint(self, e: Any) -> Any:
        if self.full:
            return self.prop.adjoint(e)
        return self.prop.adjoint(e[:, None])[:, 0]


def gerchberg_saxton(
    model: FocalPlaneModel,
    images: Any,
    *,
    iterations: int = 200,
    start: Any = None,
    basis: Basis | int | None = None,
    background: Any = 0.0,
    weights: Any = None,
    momentum: float = 0.0,
    tol: float = 1e-6,
    check_every: int = 10,
    unwrap: bool = True,
    callback: Callable[[int, Any, float], bool | None] | None = None,
) -> Result:
    """Gerchberg-Saxton / Misell phase retrieval with a known pupil amplitude.

    Parameters
    ----------
    model:
        Forward model. Broadband models are run at the reference wavelength
        (a warning is issued when the fractional bandwidth exceeds 5%).
    images:
        ``(K, my, mx)`` measured images (any linear units).
    iterations:
        Maximum number of iterations.
    start:
        Initial OPD map in metres (default flat).
    basis:
        Optional basis (or number of Zernike modes) the unwrapped phase is
        fitted to; the returned OPD is then the modal fit.
    background:
        Per-channel background subtracted before taking square roots.
    weights:
        Per-pixel weights; zero marks unmeasured pixels, whose model value is
        kept.
    momentum:
        Over-relaxation ``beta`` of the accelerated GS update
        ``theta <- P(theta) + beta (P(theta) - P(theta_prev))`` (0 = plain GS).
    tol:
        Stop when the relative change of the error metric over
        ``check_every`` iterations is below ``tol``.
    check_every:
        Error-metric interval (each check synchronizes the GPU).
    unwrap:
        Unwrap the final phase (otherwise the OPD is the wrapped phase).
    callback:
        ``callback(iteration, phase, error)``; return True to stop.

    Returns
    -------
    Result
        ``extra["wrapped_phase"]`` holds the wrapped pupil phase and
        ``history`` the normalized amplitude error
        ``sum (|E| - sqrt(d))^2 / sum d`` over measured pixels.
    """
    t0 = time.perf_counter()
    be = model.backend
    xp = be.xp
    bandwidth = (model.wavelengths.max() - model.wavelengths.min()) / model.wavelength
    if bandwidth > 0.05:
        warnings.warn(
            f"Gerchberg-Saxton runs monochromatically at the reference wavelength; this "
            f"model has {bandwidth:.0%} bandwidth. Refine with solvephase.solve().",
            stacklevel=2,
        )
    plan = _GSPlan(model)
    data = be.asarray(images, dtype="real")
    if data.ndim == 2:
        data = data[None]
    k = model.n_channels
    if data.shape != (k, *model.image_shape):
        raise ValueError(
            f"images have shape {tuple(data.shape)}, expected {(k, *model.image_shape)}"
        )
    bg = be.asarray(np.broadcast_to(np.asarray(background, dtype=np.float64), (k,)), dtype="real")
    signal = xp.maximum(data - bg[:, None, None], 0.0)
    os_ = model.oversample
    if os_ > 1:
        signal = xp.repeat(xp.repeat(signal, os_, axis=-2), os_, axis=-1) / os_**2
    measured = xp.ones(signal.shape, dtype=bool)
    if weights is not None:
        w = be.asarray(np.broadcast_to(be.to_numpy(weights), tuple(data.shape)), dtype="real") > 0
        if os_ > 1:
            w = xp.repeat(xp.repeat(w, os_, axis=-2), os_, axis=-1)
        measured = w
    sqrt_signal = xp.sqrt(signal)
    sig_energy = xp.sum(signal * measured, axis=(1, 2))

    amp = be.asarray(model.pupil.amplitude, dtype="real")
    div = model._div  # (K, ny, nx) radians at the reference wavelength
    if start is None:
        theta = xp.zeros(model.pupil.shape, dtype=be.real_dtype)
    elif isinstance(start, Result):
        theta = be.asarray(start.phase, dtype="real")
    else:
        theta = be.asarray(start, dtype="real") * (2 * math.pi / model.wavelength)
    ys, xs = plan.slices
    prev_proj = None
    history: list[float] = []
    times: list[float] = []
    err_prev = math.inf
    message, converged = "iteration limit reached", False
    it = 0
    for it in range(1, iterations + 1):
        u = amp * xp.exp(1j * (theta[None] + div))
        u = u.astype(be.complex_dtype, copy=False)
        e = plan.forward(u)
        win = e[:, ys, xs]
        mag = xp.abs(win)
        # Match the data's energy to the model's in the measured window.
        model_energy = xp.sum((mag * mag) * measured, axis=(1, 2))
        scale = xp.sqrt(model_energy / xp.maximum(sig_energy, 1e-300))
        target = sqrt_signal * scale[:, None, None]
        new_win = xp.where(measured, target * win / xp.maximum(mag, 1e-30), win)
        check = it % check_every == 0 or it == iterations
        if check:
            resid = xp.sum(((mag - target) ** 2) * measured)
            err = float(resid / xp.maximum(xp.sum(model_energy), 1e-300))
            history.append(err)
            times.append(time.perf_counter() - t0)
        e[:, ys, xs] = new_win
        back = plan.adjoint(e) * xp.exp(-1j * div)
        proj = xp.angle(xp.sum(back, axis=0))
        if momentum and prev_proj is not None:
            step = xp.angle(xp.exp(1j * (proj - prev_proj)))
            theta = proj + momentum * step
        else:
            theta = proj
        prev_proj = proj
        if check:
            if callback is not None and callback(it, theta, err):
                message, converged = "stopped by callback", True
                break
            if abs(err_prev - err) <= tol * max(err, 1e-300):
                message, converged = "error metric stalled below tol", True
                break
            err_prev = err
    theta = xp.where(be.asarray(model.pupil.mask), theta, 0.0)
    wrapped = theta
    if unwrap:
        theta = unwrap_phase(theta, model.pupil.mask, weights=model.pupil.amplitude, device=be)
        theta = be.asarray(theta, dtype="real")
    if isinstance(basis, (int, np.integer)):
        basis = Basis.zernike(model.pupil, int(basis))
    opd = theta * (model.wavelength / (2 * math.pi))
    problem = FocalPlaneProblem(model, data, basis=basis, background=background)
    x = problem.initial(be.to_numpy(opd))
    result = problem.result(
        x, method="gs" if k == 1 else "misell", elapsed=time.perf_counter() - t0
    )
    result.history, result.times = history, times
    result.n_iter, result.converged, result.message = it, converged, message
    result.loss = history[-1] if history else math.nan
    result.extra["wrapped_phase"] = wrapped
    if basis is None:
        # Keep the unwrapped zonal phase exactly (the zonal fit is the identity).
        result.opd, result.phase = opd, theta
    return result
