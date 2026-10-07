"""Focal-plane phase retrieval as a nonlinear inverse problem.

:class:`FocalPlaneProblem` couples a :class:`~solvephase.focal.FocalPlaneModel`
with measured images, a wavefront :class:`~solvephase.basis.Basis`, a data
:mod:`loss <solvephase.losses>`, nuisance parameters and regularization. It
exposes the objective, its analytic gradient and a Gauss-Newton system, all
evaluated on the model's device, and :func:`solve` minimizes it with L-BFGS,
Levenberg-Marquardt or Adam.

This is the "nonlinear optimization" family of phase retrieval (Fienup 1993;
Jurling & Fienup 2014) and, with a Poisson loss, the maximum-likelihood phase
diversity estimator of Paxman, Schulz & Fienup (1992) for a known (point)
object.

Internal parameters are dimensionless and well scaled: phase coefficients in
radians at the reference wavelength, log-flux, background in units of the
data's standard deviation. Results are reported in physical units (metres of
OPD, data units for flux and background).
"""

from __future__ import annotations

import itertools
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .backend import Backend
from .basis import Basis
from .focal import FocalPlaneModel
from .losses import Loss, PoissonLoss, make_loss
from .optimize import OptimizeResult, adam, lbfgs, levenberg_marquardt
from .result import Result

__all__ = ["FocalPlaneProblem", "noise_weights", "solve"]


def noise_weights(
    images: Any, read_noise: float = 0.0, *, gain: float = 1.0, floor: float = 1.0
) -> np.ndarray:
    """Inverse-variance weights ``1 / (gain * max(d, 0) + read_noise^2)`` for a Gaussian loss.

    ``images`` are in photo-electrons (or data units with Poisson ``gain``);
    ``floor`` bounds the variance from below.
    """
    data = np.asarray(images, dtype=np.float64)
    return 1.0 / np.maximum(gain * np.maximum(data, 0.0) + read_noise**2, floor)


@dataclass(frozen=True)
class _Layout:
    coeffs: slice
    amplitude: slice
    tilts: slice
    log_flux: slice
    background: slice
    size: int


class FocalPlaneProblem:
    """Phase retrieval from one or more focal-plane images.

    Parameters
    ----------
    model:
        The forward model (pupil, wavelengths, sampling, diversity channels).
    images:
        Measured images ``(K, my, mx)`` (or ``(my, mx)`` for one channel), in
        any linear units (ADU, photo-electrons). Use photo-electrons with the
        Poisson loss.
    basis:
        Wavefront parameterization. ``None`` means zonal (one value per
        pupil pixel); an int ``n`` means ``Basis.zernike(pupil, n)``.
    loss:
        ``"gaussian"`` (weighted least squares), ``"poisson"`` (maximum
        likelihood for photon-counting data) or ``"amplitude"``; or a
        :class:`~solvephase.losses.Loss`.
    weights:
        Per-pixel weights ``(K, my, mx)``, broadcastable; zero masks a pixel.
        For the Gaussian loss, inverse variances make it a likelihood (see
        :func:`noise_weights`). Defaults to 1.
    read_noise:
        Read noise in data units, used by the Poisson loss.
    fit_flux:
        Fit each channel's total flux. Otherwise ``flux`` is used.
    flux:
        Fixed per-channel flux (scalar or ``(K,)``) when ``fit_flux`` is
        False; by default the data sum.
    fit_background:
        Fit a constant background per channel. Otherwise ``background`` is
        used.
    background:
        Fixed per-channel background (data units per pixel). Default 0.
    fit_tilt:
        Fit a tip/tilt per channel relative to channel 0 (image
        registration); requires at least two channels.
    fit_amplitude:
        Also retrieve the pupil amplitude (zonal, on the pupil support).
    regularization:
        Tikhonov weight on the phase coefficients (radians^-2).
    prior:
        Add the Gaussian prior ``0.5 * sum c^2 / var`` from
        :attr:`Basis.variances` (e.g. a KL basis), i.e. maximum a posteriori.
    smoothness:
        Weight of a first-difference smoothness penalty on a zonal phase.
    """

    def __init__(
        self,
        model: FocalPlaneModel,
        images: Any,
        *,
        basis: Basis | int | None = None,
        loss: str | Loss = "gaussian",
        weights: Any = None,
        read_noise: float = 0.0,
        fit_flux: bool = True,
        flux: Any = None,
        fit_background: bool = False,
        background: Any = 0.0,
        fit_tilt: bool = False,
        fit_amplitude: bool = False,
        regularization: float = 0.0,
        prior: bool = False,
        smoothness: float = 0.0,
    ) -> None:
        self.model = model
        self.backend: Backend = model.backend
        be, xp = self.backend, self.backend.xp
        pupil = model.pupil
        if isinstance(basis, (int, np.integer)):
            basis = Basis.zernike(pupil, int(basis))
        self.basis = basis if basis is not None else Basis.zonal(pupil)
        if self.basis.pupil.shape != pupil.shape or self.basis.pupil.n_valid != pupil.n_valid:
            raise ValueError("the basis is defined on a different pupil than the model")
        if isinstance(loss, str) and loss.lower() in ("poisson", "ml") and read_noise:
            loss = PoissonLoss(read_noise=read_noise)
        self.loss = make_loss(loss)

        data = be.asarray(images, dtype="real")
        if data.ndim == 2:
            data = data[None]
        k = model.n_channels
        if data.shape != (k, *model.image_shape):
            raise ValueError(
                f"images have shape {tuple(data.shape)} but the model expects "
                f"{(k, *model.image_shape)} ({k} channel(s))"
            )
        if not bool(xp.all(xp.isfinite(data))):
            raise ValueError("images contain non-finite values; mask them with weights=0 instead")
        self.data = data
        if weights is None:
            w = xp.ones_like(data)
        else:
            w = be.asarray(np.broadcast_to(be.to_numpy(weights), tuple(data.shape)), dtype="real")
            if bool(xp.any(w < 0)):
                raise ValueError("weights must be non-negative")
        self.weights = w
        # The data term is evaluated in float64 even on a single-precision
        # backend: images are small next to the propagation fields, and float32
        # loss values are too noisy for line-search and LM acceptance tests.
        self._data64 = data.astype(np.float64)
        self._weights64 = w.astype(np.float64)
        if fit_tilt and k < 2:
            raise ValueError(
                "fit_tilt needs at least two channels (tilts are relative to channel 0)"
            )
        self.fit_flux = bool(fit_flux)
        self.fit_background = bool(fit_background)
        self.fit_tilt = bool(fit_tilt)
        self.fit_amplitude = bool(fit_amplitude)
        self.regularization = float(regularization)
        self.smoothness = float(smoothness)
        if prior and self.basis.variances is None:
            raise ValueError("prior=True needs a basis with variances (e.g. Basis.kl)")
        self.prior = bool(prior)

        # Fixed device arrays.
        mask = pupil.mask
        self._flat_index = xp.asarray(np.flatnonzero(mask.ravel()))
        self._amp0 = be.asarray(pupil.amplitude, dtype="real")
        self._amp_scale = float(np.mean(pupil.amplitude[mask]))
        self._modes = None if self.basis.is_zonal else self.basis.on(be)
        if self.prior:
            assert self.basis.variances is not None
            var_rad = self.basis.variances * (2.0 * math.pi / model.wavelength) ** 2
            self._inv_var = be.asarray(1.0 / var_rad, dtype="real")
        y, x = pupil.coordinates()
        w2 = pupil.amplitude**2
        ramps = []
        for c in (x, y):
            r = np.where(mask, c - np.sum(w2 * c) / np.sum(w2), 0.0)
            ramps.append(r / math.sqrt(np.sum(w2 * r**2) / np.sum(w2)))
        self._ramps = be.asarray(np.stack(ramps), dtype="real")  # (2, ny, nx), unit RMS
        if self.smoothness:
            self._pair_x = be.asarray(mask[:, 1:] & mask[:, :-1], dtype="real")
            self._pair_y = be.asarray(mask[1:, :] & mask[:-1, :], dtype="real")
        host = be.to_numpy(data)
        border = np.concatenate(
            [host[:, 0, :], host[:, -1, :], host[:, :, 0], host[:, :, -1]], axis=1
        )
        # Background is parameterized in units of the border noise (robust MAD),
        # not the image's spread, which a bright PSF core inflates enormously.
        mad = 1.4826 * np.median(np.abs(border - np.median(border, axis=1)[:, None]), axis=1)
        peak = np.max(np.abs(host.reshape(k, -1)), axis=1)
        self._scale = np.maximum(mad, 1e-4 * peak) + 1e-30
        if self.fit_background:
            self._bg0 = np.median(border, axis=1)
        else:
            self._bg0 = np.broadcast_to(np.asarray(background, dtype=np.float64), (k,)).copy()
        self._flux_fixed = (
            None
            if flux is None
            else np.broadcast_to(np.asarray(flux, dtype=np.float64), (k,)).copy()
        )
        self._flux0: np.ndarray | None = None
        # Device copies of the small host vectors above, refreshed when their values change.
        self._device_cache: dict[str, tuple[np.ndarray, Any]] = {}
        self._mode_maps: Any = None

        n_c = self.basis.n_modes
        n_a = pupil.n_valid if self.fit_amplitude else 0
        n_t = 2 * (k - 1) if self.fit_tilt else 0
        n_f = k if self.fit_flux else 0
        n_b = k if self.fit_background else 0
        edges = np.cumsum([0, n_c, n_a, n_t, n_f, n_b]).tolist()
        sl = [slice(a, b) for a, b in itertools.pairwise(edges)]
        self.layout = _Layout(
            coeffs=sl[0],
            amplitude=sl[1],
            tilts=sl[2],
            log_flux=sl[3],
            background=sl[4],
            size=edges[-1],
        )

    # ---------------------------------------------------------------- basics
    @property
    def n_params(self) -> int:
        """Length of the internal parameter vector."""
        return self.layout.size

    @property
    def n_channels(self) -> int:
        """Number of image channels."""
        return self.model.n_channels

    def __repr__(self) -> str:
        return (
            f"FocalPlaneProblem(basis={self.basis.kind!r}[{self.basis.n_modes}], "
            f"loss={self.loss!r}, channels={self.n_channels}, params={self.n_params}, "
            f"backend={self.backend})"
        )

    # ------------------------------------------------------------ synthesis
    def _phase_map(self, coeffs: Any) -> Any:
        xp = self.backend.xp
        values = coeffs if self._modes is None else coeffs @ self._modes
        flat = xp.zeros(self.model.pupil.shape[0] * self.model.pupil.shape[1], dtype=values.dtype)
        flat[self._flat_index] = values
        return flat.reshape(self.model.pupil.shape)

    def _analyze(self, grad_map: Any) -> Any:
        values = grad_map.reshape(-1)[self._flat_index]
        return values if self._modes is None else self._modes @ values

    def _amplitude(self, x: Any) -> Any:
        if not self.fit_amplitude:
            return self._amp0
        flat = self._amp0.reshape(-1).copy()
        flat[self._flat_index] = flat[self._flat_index] + self._amp_scale * x[self.layout.amplitude]
        return flat.reshape(self.model.pupil.shape)

    def _channel_phase(self, x: Any) -> Any:
        phase = self._phase_map(x[self.layout.coeffs])
        if not self.fit_tilt:
            return phase
        xp = self.backend.xp
        t = x[self.layout.tilts].reshape(-1, 2)
        zero = xp.zeros((1, 2), dtype=t.dtype)
        t = xp.concatenate([zero, t])
        tilt_maps = xp.tensordot(t, self._ramps, axes=(1, 0))  # (K, ny, nx)
        return phase[None] + tilt_maps

    def _on_device(self, name: str, host: np.ndarray) -> Any:
        """``backend.asarray(host, dtype="real")``, cached while ``host`` keeps its values.

        Saves a host-to-device copy (a synchronizing transfer on the GPU) per
        objective evaluation; the comparison costs a few microseconds.
        """
        cached = self._device_cache.get(name)
        if cached is None or not np.array_equal(cached[0], host):
            cached = (np.array(host, copy=True), self.backend.asarray(host, dtype="real"))
            self._device_cache[name] = cached
        return cached[1]

    def _flux_background(self, x: Any, psf: Any) -> tuple[Any, Any]:
        xp = self.backend.xp
        if self._flux0 is None:
            self._init_flux(psf)
        assert self._flux0 is not None
        flux = self._on_device("flux0", self._flux0)
        if self.fit_flux:
            flux = flux * xp.exp(x[self.layout.log_flux])
        bg = self._on_device("bg0", self._bg0)
        if self.fit_background:
            bg = bg + self._on_device("scale", self._scale) * x[self.layout.background]
        return flux, bg

    def _init_flux(self, psf: Any) -> None:
        if self._flux_fixed is not None:
            self._flux0 = self._flux_fixed
            return
        be = self.backend
        data = be.to_numpy(self.data).astype(np.float64)
        p = be.to_numpy(psf).astype(np.float64)
        w = be.to_numpy(self.weights).astype(np.float64)
        signal = data - self._bg0[:, None, None]
        num = np.sum(w * signal, axis=(1, 2))
        den = np.sum(w * p, axis=(1, 2))
        flux = num / np.maximum(den, 1e-300)
        bad = ~np.isfinite(flux) | (flux <= 0)
        flux[bad] = np.maximum(np.sum(np.abs(data[bad]), axis=(1, 2)), 1e-30)
        self._flux0 = flux

    # ------------------------------------------------------------- objective
    def initial(self, start: Any = None) -> Any:
        """Internal starting vector.

        ``start`` may be ``None`` (flat wavefront), an OPD map in metres
        ``(ny, nx)``, a coefficient vector in metres, or a :class:`Result`
        (its OPD, amplitude, flux and background are reused).
        """
        be, xp = self.backend, self.backend.xp
        x = xp.zeros(self.n_params, dtype=be.real_dtype)
        k_wave = 2.0 * math.pi / self.model.wavelength
        if isinstance(start, Result):
            result = start
            opd = be.to_numpy(result.opd)
            x[self.layout.coeffs] = be.asarray(self.basis.fit(opd) * k_wave, dtype="real")
            if self.fit_amplitude and result.amplitude is not None:
                amp = be.to_numpy(result.amplitude)[self.model.pupil.mask]
                rel = (amp - self.model.pupil.amplitude[self.model.pupil.mask]) / self._amp_scale
                x[self.layout.amplitude] = be.asarray(rel, dtype="real")
            if (
                result.flux is not None
                and self._flux_fixed is None
                and len(result.flux) == self.n_channels
            ):
                self._flux0 = np.asarray(be.to_numpy(result.flux), dtype=np.float64)
            if (
                self.fit_background
                and result.background is not None
                and len(result.background) == self.n_channels
            ):
                self._bg0 = np.asarray(be.to_numpy(result.background), dtype=np.float64)
            if self.fit_tilt and result.tilts is not None and len(result.tilts) == self.n_channels:
                t = np.asarray(be.to_numpy(result.tilts))[1:] * k_wave
                x[self.layout.tilts] = be.asarray(t.reshape(-1), dtype="real")
        elif start is not None:
            arr = np.asarray(be.to_numpy(start), dtype=np.float64)
            if arr.shape == self.model.pupil.shape:
                coeffs = self.basis.fit(arr)
            elif arr.shape == (self.basis.n_modes,):
                coeffs = arr
            else:
                raise ValueError(
                    f"start must be an OPD map {self.model.pupil.shape}, {self.basis.n_modes} "
                    f"coefficients or a Result; got shape {arr.shape}"
                )
            x[self.layout.coeffs] = be.asarray(coeffs * k_wave, dtype="real")
        if self._flux0 is None:
            psf = self.model.forward(self._channel_phase(x), self._amplitude(x)).images
            self._init_flux(psf)
        return x

    def _regularization(self, x: Any) -> tuple[float, Any]:
        xp = self.backend.xp
        value = 0.0
        grad = xp.zeros_like(x)
        c = x[self.layout.coeffs]
        if self.regularization:
            value += 0.5 * self.regularization * float(xp.sum(c * c))
            grad[self.layout.coeffs] += self.regularization * c
        if self.prior:
            value += 0.5 * float(xp.sum(c * c * self._inv_var))
            grad[self.layout.coeffs] += c * self._inv_var
        if self.smoothness:
            phase = self._phase_map(c)
            dx = (phase[:, 1:] - phase[:, :-1]) * self._pair_x
            dy = (phase[1:, :] - phase[:-1, :]) * self._pair_y
            value += 0.5 * self.smoothness * float(xp.sum(dx * dx) + xp.sum(dy * dy))
            gmap = xp.zeros_like(phase)
            gmap[:, 1:] += dx
            gmap[:, :-1] -= dx
            gmap[1:, :] += dy
            gmap[:-1, :] -= dy
            grad[self.layout.coeffs] += self.smoothness * self._analyze(gmap)
        return value, grad

    def _model_images(self, x: Any) -> tuple[Any, Any, Any, Any]:
        state = self.model.forward(self._channel_phase(x), self._amplitude(x))
        flux, bg = self._flux_background(x, state.images)
        model = flux[:, None, None] * state.images + bg[:, None, None]
        return state, flux, bg, model

    def _loss(self, model: Any) -> tuple[Any, Any, Any]:
        """Data term in float64; gradient in working precision."""
        rdt = self.backend.real_dtype
        fused = self.loss._evaluate_fused(self.backend, model, self._data64, self._weights64, rdt)
        if fused is not None:
            return fused
        val, grad, curv = self.loss.evaluate(
            self.backend, model.astype(np.float64), self._data64, self._weights64
        )
        return val, grad.astype(rdt, copy=False), curv

    def value(self, x: Any) -> float:
        """Objective value at ``x`` (data term + regularization)."""
        _, _, _, model = self._model_images(x)
        val, _, _ = self._loss(model)
        reg, _ = self._regularization(x) if self._has_reg else (0.0, None)
        return float(val) + reg

    @property
    def _has_reg(self) -> bool:
        return bool(self.regularization or self.prior or self.smoothness)

    def objective(self, x: Any) -> tuple[float, Any]:
        """Objective value and its gradient with respect to the internal vector ``x``."""
        xp = self.backend.xp
        state, flux, _, model = self._model_images(x)
        val, g_model, _ = self._loss(model)
        g_psf = g_model * flux[:, None, None]
        g_phase, g_amp = self.model.vjp(state, g_psf, amplitude=self.fit_amplitude)
        # Blocks in layout order, joined by one concatenate (not zeros + one scatter each).
        parts = [self._analyze(xp.sum(g_phase, axis=0))]
        if self.fit_amplitude:
            parts.append(self._amp_scale * g_amp.reshape(-1)[self._flat_index])
        if self.fit_tilt:
            gt = xp.tensordot(g_phase[1:], self._ramps, axes=([1, 2], [1, 2]))
            parts.append(gt.reshape(-1))
        if self.fit_flux:
            parts.append(flux * xp.sum(g_model * state.images, axis=(1, 2)))
        if self.fit_background:
            scale = self._on_device("scale", self._scale)
            parts.append(scale * xp.sum(g_model, axis=(1, 2)))
        grad = xp.concatenate([p.astype(x.dtype, copy=False) for p in parts])
        value = float(val)
        if self._has_reg:
            reg_val, reg_grad = self._regularization(x)
            value += reg_val
            grad = grad + reg_grad
        return value, grad

    def gauss_newton(self, x: Any, *, chunk: int | None = None) -> tuple[float, Any, Any]:
        """Objective, gradient and Gauss-Newton (Fisher) matrix at ``x``.

        Builds the image Jacobian column-block by column-block with forward-mode
        derivatives; ``chunk`` directions are propagated at once (default: as
        many as fit in about 64 MB of propagation workspace). Not available
        with ``fit_amplitude`` or a zonal basis (too many parameters); use
        L-BFGS there.
        """
        if self.fit_amplitude or self.basis.is_zonal:
            raise ValueError(
                "Gauss-Newton needs a modal basis without fitted amplitude; use L-BFGS"
            )
        be, xp = self.backend, self.backend.xp
        state, flux, _, model = self._model_images(x)
        val, g_model, curv = self._loss(model)
        k = self.n_channels
        npix = int(np.prod(self.model.image_shape))
        rows = []
        n_c = self.basis.n_modes
        ny, nx = self.model.pupil.shape
        assert self._modes is not None
        if chunk is None:
            chunk = self._jacobian_chunk()
        maps = self._direction_maps()
        for start in range(0, n_c, chunk):
            stop = min(start + chunk, n_c)
            if maps is not None:
                dirs = maps[start:stop]
            else:
                dirs = xp.zeros(((stop - start), ny * nx), dtype=be.real_dtype)
                dirs[:, self._flat_index] = self._modes[start:stop]
            d_img = self.model.jvp(state, dirs.reshape(-1, ny, nx))  # (P, K, my, mx)
            rows.append((d_img * flux[None, :, None, None]).reshape(stop - start, k * npix))
        if self.fit_tilt:
            dirs = xp.zeros((2 * (k - 1), k, ny, nx), dtype=be.real_dtype)
            for c in range(1, k):
                dirs[2 * (c - 1) : 2 * c, c] = self._ramps
            d_img = self.model.jvp(state, dirs, per_channel=True)
            rows.append((d_img * flux[None, :, None, None]).reshape(-1, k * npix))
        diag = xp.arange(k)
        if self.fit_flux:
            block = xp.zeros((k, k, npix), dtype=be.real_dtype)
            block[diag, diag] = flux[:, None] * state.images.reshape(k, npix)
            rows.append(block.reshape(k, k * npix))
        if self.fit_background:
            block = xp.zeros((k, k, npix), dtype=be.real_dtype)
            block[diag, diag] = self._on_device("scale", self._scale)[:, None]
            rows.append(block.reshape(k, k * npix))
        # Propagation runs in working precision, but the normal equations are
        # accumulated in float64: near the optimum J^T g is a sum of large,
        # cancelling terms that float32 cannot resolve. Concatenating straight
        # into float64 saves a pass over the Jacobian.
        jac = xp.concatenate(rows, axis=0, dtype=np.float64)
        h = curv.reshape(-1).astype(np.float64)
        with be.blas_limit(float(jac.shape[0]) ** 2 * jac.shape[1]):
            hess = (jac * h[None, :]) @ jac.T
            grad = jac @ g_model.reshape(-1).astype(np.float64)
        value = float(val)
        if self._has_reg:
            reg_val, reg_grad = self._regularization(x)
            value += reg_val
            grad = grad + reg_grad
            diag = xp.zeros(self.n_params, dtype=np.float64)
            if self.regularization:
                diag[self.layout.coeffs] += self.regularization
            if self.prior:
                diag[self.layout.coeffs] += self._inv_var
            hess = hess + xp.diag(diag)
        return value, grad, hess

    def _direction_maps(self, budget_bytes: float = 64e6) -> Any:
        """Mode maps ``(n_modes, ny * nx)`` (zero off the pupil), built once, or ``None``.

        They are the forward-mode directions of every Gauss-Newton matrix;
        caching them saves a zero-fill and a scatter per batch. ``None`` when
        they would take more than ``budget_bytes`` (built per batch instead).
        """
        if self._mode_maps is None:
            assert self._modes is not None
            ny, nx = self.model.pupil.shape
            size = self.basis.n_modes * ny * nx * self.backend.real_dtype.itemsize
            if size > budget_bytes:
                return None
            maps = self.backend.zeros((self.basis.n_modes, ny * nx))
            maps[:, self._flat_index] = self._modes
            self._mode_maps = maps
        return self._mode_maps

    def _jacobian_chunk(self, budget_bytes: float = 64e6) -> int:
        """Directions per forward-mode batch so one batch's fields fit ``budget_bytes``."""
        prop = self.model.propagator
        n_fft = max(
            [max(getattr(e, "n_fft", (0, 0))) for e in getattr(prop, "_engines", [])] or [0]
        )
        my, mx = prop.out_shape
        plane = max(n_fft * n_fft, my * mx, self.model.pupil.n_valid)
        per_dir = 4 * self.n_channels * self.model.n_wavelengths * plane
        per_dir *= self.backend.complex_dtype.itemsize
        return int(max(1, min(64, budget_bytes // max(per_dir, 1))))

    # --------------------------------------------------------------- output
    def result(
        self,
        x: Any,
        *,
        method: str,
        opt: OptimizeResult | None = None,
        elapsed: float = 0.0,
        extra: dict[str, Any] | None = None,
    ) -> Result:
        """Package an internal vector as a :class:`~solvephase.result.Result`."""
        be = self.backend
        k_wave = 2.0 * math.pi / self.model.wavelength
        coeffs = x[self.layout.coeffs]
        phase = self._phase_map(coeffs)
        _, flux, bg, model = self._model_images(x)
        tilts = None
        if self.fit_tilt:
            t = np.zeros((self.n_channels, 2))
            t[1:] = be.to_numpy(x[self.layout.tilts]).reshape(-1, 2) / k_wave
            tilts = t
        loss_value = opt.fun if opt is not None else self.value(x)
        return Result(
            method=method,
            opd=phase / k_wave,
            phase=phase,
            amplitude=self._amplitude(x),
            pupil=self.model.pupil,
            wavelength=self.model.wavelength,
            coefficients=None if self.basis.is_zonal else np.asarray(be.to_numpy(coeffs)) / k_wave,
            basis=None if self.basis.is_zonal else self.basis,
            model_images=model,
            flux=np.asarray(be.to_numpy(flux), dtype=np.float64),
            background=np.asarray(be.to_numpy(bg), dtype=np.float64),
            tilts=tilts,
            loss=float(loss_value),
            history=list(opt.history) if opt else [],
            times=list(opt.times) if opt else [],
            n_iter=opt.n_iter if opt else 0,
            converged=opt.converged if opt else False,
            message=opt.message if opt else "",
            elapsed=elapsed,
            device=be.device,
            extra=extra or {},
        )


def solve(
    problem: FocalPlaneProblem,
    *,
    method: str = "auto",
    start: Any = None,
    max_iter: int | None = None,
    callback: Callable[[int, Any, float], bool | None] | None = None,
    **options: Any,
) -> Result:
    """Minimize a :class:`FocalPlaneProblem`.

    Parameters
    ----------
    problem:
        The problem to solve.
    method:
        ``"lbfgs"`` (default for zonal or amplitude problems), ``"lm"``
        (Levenberg-Marquardt; default for modal problems with at most 300
        parameters), ``"adam"``, or ``"auto"``.
    start:
        Starting point, see :meth:`FocalPlaneProblem.initial`.
    max_iter:
        Iteration limit (defaults: 300 for L-BFGS, 50 for LM, 1000 for Adam).
    callback:
        ``callback(iteration, x, f)``; return ``True`` to stop.
    **options:
        Passed to the optimizer (``ftol``, ``gtol``, ``memory``, ``damping``,
        ``learning_rate`` ...).
    """
    t0 = time.perf_counter()
    method = method.lower()
    if method == "auto":
        small = not problem.basis.is_zonal and not problem.fit_amplitude and problem.n_params <= 300
        method = "lm" if small else "lbfgs"
    x0 = problem.initial(start)
    be = problem.backend
    if method == "lbfgs":
        opt = lbfgs(
            problem.objective, x0, be, max_iter=max_iter or 300, callback=callback, **options
        )
    elif method in ("lm", "levenberg-marquardt", "gauss-newton", "gn"):
        method = "lm"
        opt = levenberg_marquardt(
            problem.gauss_newton,
            problem.value,
            x0,
            be,
            max_iter=max_iter or 50,
            callback=callback,
            **options,
        )
    elif method == "adam":
        opt = adam(
            problem.objective, x0, be, max_iter=max_iter or 1000, callback=callback, **options
        )
    else:
        raise ValueError(f"unknown method {method!r}; use 'lbfgs', 'lm', 'adam' or 'auto'")
    be.synchronize()
    return problem.result(opt.x, method=method, opt=opt, elapsed=time.perf_counter() - t0)
