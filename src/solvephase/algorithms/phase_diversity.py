"""Phase diversity with an unknown extended object.

``K`` images ``d_k = o * s_k + n_k`` of the same unknown scene ``o`` are
recorded through the unknown wavefront plus a known diversity per channel
(defocus, astigmatism ...). Gonsalves (Opt. Eng. 21, 829, 1982) showed that
the object can be eliminated analytically. In Fourier space, with
``D_k = F d_k`` and the optical transfer functions ``S_k = F s_k``, the
least-squares object for fixed ``S_k`` is the multi-frame Wiener estimate

    O = sum_k D_k conj(S_k) / (sum_k |S_k|^2 + gamma),

and substituting it back leaves a metric of the wavefront alone (Paxman,
Schulz & Fienup, JOSA A 9, 1072, 1992):

    L = sum_f [ sum_k |D_k|^2 - |sum_k D_k conj(S_k)|^2 / (sum_k |S_k|^2 + gamma) ].

``gamma`` regularizes the division (the inverse signal-to-noise power ratio
of a Wiener filter). Following Löfdahl & Scharmer (A&AS 107, 243, 1994) the
data are apodized against periodic-convolution edge effects and the sum is
restricted to spatial frequencies inside the diffraction cutoff, where the
transfer functions carry information. Mugnier, Blanc & Idier (Adv. Imaging
Electron Phys. 141, 1, 2006) review the method.

:class:`PhaseDiversityProblem` evaluates ``L`` and its exact gradient with
respect to the modal coefficients: ``dL/dS_k`` follows from the envelope
theorem (``O`` is optimal, so its own variation drops out), the real FFT is
back-propagated to ``dL/d psf_k``, and
:meth:`~solvephase.focal.FocalPlaneModel.vjp` takes it to the pupil phase.
The metric is evaluated in the equivalent form
``sum_k |D_k - O S_k|^2 + gamma |O|^2``, a sum of non-negative terms that
keeps its accuracy in single precision. :func:`phase_diversity` minimizes it
with L-BFGS, optionally coarse-to-fine over the number of modes.

Internal parameters are those of :class:`~solvephase.retrieval.FocalPlaneProblem`:
phase coefficients in radians at the reference wavelength (reported in
metres), a relative zonal amplitude, and per-channel tip/tilt for registration.
"""

from __future__ import annotations

import copy
import math
import time
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .. import _kernels
from ..backend import Backend
from ..basis import Basis
from ..focal import FocalPlaneModel
from ..optimize import OptimizeResult, lbfgs
from ..result import Result

__all__ = ["PhaseDiversityProblem", "phase_diversity"]


def _taper(n: int, kind: str, alpha: float) -> np.ndarray:
    """One-dimensional apodization profile, symmetric about the array centre."""
    t = (np.arange(n) + 0.5) / n
    if kind == "hann":
        return np.asarray(np.sin(math.pi * t) ** 2)
    edge = np.minimum(t, 1.0 - t)
    half = alpha / 2.0
    out = np.ones(n)
    taper = edge < half
    out[taper] = 0.5 * (1.0 - np.cos(math.pi * edge[taper] / half))
    return out


def _window(window: Any, shape: tuple[int, int], alpha: float) -> np.ndarray:
    """Apodization window ``(my, mx)``: ``"hann"``, ``"tukey"``, ``None`` or an array."""
    if window is None:
        return np.ones(shape)
    if isinstance(window, str):
        kind = window.lower()
        if kind not in ("hann", "tukey"):
            raise ValueError(f"window must be 'hann', 'tukey', None or an array; got {window!r}")
        if kind == "tukey" and not 0.0 < alpha <= 1.0:
            raise ValueError(f"window_alpha must be in (0, 1], got {alpha}")
        return _taper(shape[0], kind, alpha)[:, None] * _taper(shape[1], kind, alpha)[None, :]
    arr = np.asarray(window, dtype=np.float64)
    if arr.shape != shape:
        raise ValueError(f"window has shape {arr.shape}; the images are {shape}")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError("window must be finite and non-negative")
    return arr


def _otf_extent(model: FocalPlaneModel) -> float:
    """Pupil extent in metres that sets the OTF cutoff (its widest baseline, bounded above)."""
    pupil = model.pupil
    y, x = pupil.coordinates()
    r = np.hypot(y[pupil.mask], x[pupil.mask])
    return float(max(pupil.diameter, 2.0 * (r.max() + pupil.pitch / math.sqrt(2.0))))


class PhaseDiversityProblem:
    """Phase diversity of an unknown extended object, with the object eliminated.

    Parameters
    ----------
    model:
        Forward model with one channel per image. Its ``image_shape`` must be
        the data shape; its normalized PSFs (sum ~1) are the convolution
        kernels. Sample at Nyquist or better at the shortest wavelength.
    images:
        ``(K, my, mx)`` images of the same scene, ``K >= 2``, in any linear
        units (a constant background is absorbed by the object). Channels
        must be registered to a pixel or better (see ``fit_tilt``).
    basis:
        Wavefront parameterization. ``None`` means zonal; an int ``n`` means
        ``Basis.zernike(pupil, n, start=4)`` (defocus upwards), because a
        common tip/tilt only moves the unknown object and is not measurable.
    regularization:
        ``gamma`` of the metric, added to ``sum_k |S_k|^2`` (dimensionless:
        the OTFs are 1 at zero frequency). It only needs to stabilize the
        division where every OTF vanishes; larger values bias the wavefront
        towards sharper PSFs. ``"auto"`` is the measured noise power over the
        peak in-band data power (the inverse of the best per-frequency SNR,
        squared).
    object_regularization:
        ``gamma`` of the reported object estimate (a Wiener filter, which
        needs more damping than the metric). ``"auto"`` is the noise power
        over the mean in-band data power.
    window:
        Apodization: each image minus its edge level (the mean weighted by
        ``1 - window``) is multiplied by the window and the level added back,
        so the frame is periodic and a uniform background passes untouched.
        ``"tukey"`` (cosine-tapered edges, flat centre), ``"hann"``, ``None``
        or a ``(my, mx)`` array. Use ``None`` only for scenes whose images
        already fade to a uniform level at the frame edges.
    window_alpha:
        Tapered fraction of each axis for the Tukey window.
    frequency_mask:
        ``"diffraction"`` sums the metric over spatial frequencies inside the
        diffraction cutoff ``D / lambda_min``; a float ``a`` uses ``a`` times
        that radius; ``None`` uses every frequency. Zero frequency (the total
        flux, which carries no wavefront information) is always excluded.
    fit_amplitude:
        Also retrieve the pupil amplitude (zonal, on the pupil support).
    fit_tilt:
        Fit a tip/tilt per channel relative to channel 0 (image
        registration).
    equalize_flux:
        Scale every image to the mean total flux, so unequal exposures do not
        bias the metric.

    Notes
    -----
    The objective is the reduced metric divided by the noise power per
    frequency, so it is a chi-square-like number of order
    ``(K - 1) x (in-band frequencies)`` at the solution when the noise power
    can be measured outside the cutoff (otherwise it is divided by the mean
    in-band data power).
    """

    def __init__(
        self,
        model: FocalPlaneModel,
        images: Any,
        *,
        basis: Basis | int | None = None,
        regularization: float | str = "auto",
        object_regularization: float | str = "auto",
        window: Any = "tukey",
        window_alpha: float = 0.25,
        frequency_mask: str | float | None = "diffraction",
        fit_amplitude: bool = False,
        fit_tilt: bool = False,
        equalize_flux: bool = True,
    ) -> None:
        self.model = model
        self.backend: Backend = model.backend
        be, xp = self.backend, self.backend.xp
        pupil = model.pupil
        if isinstance(basis, (int, np.integer)):
            basis = Basis.zernike(pupil, int(basis), start=4)
        self.basis = basis if basis is not None else Basis.zonal(pupil)
        if self.basis.pupil.shape != pupil.shape or self.basis.pupil.n_valid != pupil.n_valid:
            raise ValueError("the basis is defined on a different pupil than the model")

        data = be.asarray(images, dtype="real")
        if data.ndim == 2:
            data = data[None]
        k = model.n_channels
        if data.shape != (k, *model.image_shape):
            raise ValueError(
                f"images have shape {tuple(data.shape)} but the model expects "
                f"{(k, *model.image_shape)} ({k} channel(s))"
            )
        if k < 2:
            raise ValueError(
                "phase diversity with an unknown object needs at least two images with "
                "different diversity; with one image the object absorbs the aberration. "
                "For a known point source use solvephase.FocalPlaneProblem."
            )
        if not bool(xp.all(xp.isfinite(data))):
            raise ValueError("images contain non-finite values; replace or inpaint them first")
        if not model.nyquist_sampled:
            warnings.warn(
                "the images are undersampled at the shortest wavelength (< 2 pixels per "
                "lambda/D); the transfer function aliases and the object model is approximate",
                stacklevel=2,
            )
        self.data = data
        self.fit_amplitude = bool(fit_amplitude)
        self.fit_tilt = bool(fit_tilt)
        self.equalize_flux = bool(equalize_flux)
        my, mx = model.image_shape
        self._shape = (my, mx)

        # ---- data preprocessing: flux equalization and apodization
        host = np.asarray(be.to_numpy(data), dtype=np.float64)
        totals = host.reshape(k, -1).sum(axis=1)
        if self.equalize_flux:
            if np.any(totals <= 0):
                raise ValueError("equalize_flux needs images with positive total flux")
            self.flux_scale = totals.mean() / totals
        else:
            self.flux_scale = np.ones(k)
        win = _window(window, (my, mx), window_alpha)
        self.window = be.asarray(win, dtype="real")
        # Subtract the level at the field edge (the taper-weighted mean), apodize, add it back:
        # a constant convolves to itself, so only the scene's structure meets the taper.
        edge = 1.0 - win
        level = np.sum(host * edge, axis=(1, 2)) / max(float(edge.sum()), 1e-300)
        level_b = be.asarray((level * self.flux_scale)[:, None, None], dtype="real")
        scaled = data * be.asarray(self.flux_scale[:, None, None], dtype="real")
        self.apodized = level_b + self.window * (scaled - level_b)
        self._dhat = be.rfft2(self.apodized)  # (K, my, mx//2 + 1)

        # ---- frequency grid, diffraction cutoff and Hermitian weights
        fy = np.fft.fftfreq(my)[:, None]
        fx = np.fft.rfftfreq(mx)[None, :]
        rho = np.hypot(fy, fx)
        extent = _otf_extent(model)
        self.cutoff = float(model.pixel_scale * extent / model.wavelengths.min())  # cycles/pixel
        if frequency_mask is None:
            inband = np.ones_like(rho, dtype=bool)
        elif isinstance(frequency_mask, str):
            if frequency_mask.lower() != "diffraction":
                raise ValueError(
                    f"frequency_mask must be 'diffraction', a float or None; got {frequency_mask!r}"
                )
            inband = rho < self.cutoff
        else:
            frac = float(frequency_mask)
            if not frac > 0:
                raise ValueError("a numeric frequency_mask is a positive fraction of the cutoff")
            inband = rho < frac * self.cutoff
        inband[0, 0] = False
        if not inband.any():
            raise ValueError("the frequency mask selects no frequencies")
        hermitian = np.full(fx.shape, 2.0)
        hermitian[0, 0] = 1.0
        if mx % 2 == 0:
            hermitian[0, -1] = 1.0
        self.frequency_mask = inband
        self._mask = be.asarray(inband.astype(np.float64), dtype="real")
        self._fweight = be.asarray(inband * hermitian, dtype="real")
        self._object_mask = be.asarray((inband | (rho == 0)).astype(np.float64), dtype="real")
        # Optical axis in pixels: the model's offset moves the detector window, so the
        # axis sits at the window centre minus the offset (FocalPlanePropagator convention).
        shift = [(n - 1) / 2.0 - off for n, off in zip((my, mx), model.offset)]
        ramp = np.exp(-2j * math.pi * (fy * shift[0] + fx * shift[1]))
        self._axis_ramp = be.asarray(ramp, dtype="complex")

        # ---- noise power (outside the cutoff) and normalization
        power = np.asarray(be.to_numpy(xp.abs(self._dhat) ** 2), dtype=np.float64)
        outband = rho > self.cutoff
        mean_power = power.mean(axis=0)[inband]
        mean_inband = float(np.mean(mean_power))
        peak_inband = float(np.max(mean_power))
        self.noise_power: float | None = None
        if np.count_nonzero(outband) >= 8:
            self.noise_power = float(np.mean(power[:, outband]))
        self.regularization = self._gamma(regularization, "regularization", peak_inband)
        self.object_regularization = self._gamma(
            object_regularization, "object_regularization", mean_inband
        )
        norm = self.noise_power if self.noise_power else mean_inband
        self._norm = float(max(norm, 1e-300))

        # ---- parameterization (as in FocalPlaneProblem)
        mask = pupil.mask
        self._flat_index = xp.asarray(np.flatnonzero(mask.ravel()))
        self._amp0 = be.asarray(pupil.amplitude, dtype="real")
        self._amp_scale = float(np.mean(pupil.amplitude[mask]))
        self._modes = None if self.basis.is_zonal else self.basis.on(be)
        yy, xx = pupil.coordinates()
        w2 = pupil.amplitude**2
        ramps = []
        for c in (xx, yy):
            r = np.where(mask, c - np.sum(w2 * c) / np.sum(w2), 0.0)
            ramps.append(r / math.sqrt(np.sum(w2 * r**2) / np.sum(w2)))
        self._ramps = be.asarray(np.stack(ramps), dtype="real")
        self._set_layout()

    def _gamma(self, value: float | str, name: str, signal_power: float) -> float:
        """Resolve a Wiener regularization: a number, or ``"auto"`` = noise / signal power."""
        if isinstance(value, str):
            if value != "auto":
                raise ValueError(f"{name} must be 'auto' or a number, got {value!r}")
            if self.noise_power is None:
                raise ValueError(
                    f"{name}='auto' measures the noise power outside the diffraction cutoff, "
                    "but these images have no frequencies there (undersampled). Give "
                    f"{name} as a number (e.g. 1e-4)."
                )
            return max(self.noise_power / max(signal_power, 1e-300), 1e-12)
        gamma = float(value)
        if not (gamma >= 0 and math.isfinite(gamma)):
            raise ValueError(f"{name} must be finite and non-negative, got {gamma}")
        return gamma

    def _set_layout(self) -> None:
        n_c = self.basis.n_modes
        n_a = self.model.pupil.n_valid if self.fit_amplitude else 0
        n_t = 2 * (self.n_channels - 1) if self.fit_tilt else 0
        self._coeffs = slice(0, n_c)
        self._amp_sl = slice(n_c, n_c + n_a)
        self._tilt_sl = slice(n_c + n_a, n_c + n_a + n_t)
        self._size = n_c + n_a + n_t

    def with_basis(self, basis: Basis) -> PhaseDiversityProblem:
        """The same data and settings with another basis (used for coarse-to-fine solves)."""
        pupil = self.model.pupil
        if basis.pupil.shape != pupil.shape or basis.pupil.n_valid != pupil.n_valid:
            raise ValueError("the basis is defined on a different pupil than the model")
        other = copy.copy(self)
        other.basis = basis
        other._modes = None if basis.is_zonal else basis.on(self.backend)
        other._set_layout()
        return other

    # ---------------------------------------------------------------- basics
    @property
    def n_params(self) -> int:
        """Length of the internal parameter vector."""
        return self._size

    @property
    def n_channels(self) -> int:
        """Number of images ``K``."""
        return self.model.n_channels

    def __repr__(self) -> str:
        return (
            f"PhaseDiversityProblem(basis={self.basis.kind!r}[{self.basis.n_modes}], "
            f"channels={self.n_channels}, gamma={self.regularization:.3g}, "
            f"frequencies={int(self.frequency_mask.sum())}, params={self.n_params}, "
            f"backend={self.backend})"
        )

    # ------------------------------------------------------------ synthesis
    def _phase_map(self, coeffs: Any) -> Any:
        xp = self.backend.xp
        values = coeffs if self._modes is None else coeffs @ self._modes
        ny, nx = self.model.pupil.shape
        flat = xp.zeros(ny * nx, dtype=values.dtype)
        flat[self._flat_index] = values
        return flat.reshape(ny, nx)

    def _analyze(self, grad_map: Any) -> Any:
        values = grad_map.reshape(-1)[self._flat_index]
        return values if self._modes is None else self._modes @ values

    def _amplitude(self, x: Any) -> Any:
        if not self.fit_amplitude:
            return self._amp0
        flat = self._amp0.reshape(-1).copy()
        flat[self._flat_index] = flat[self._flat_index] + self._amp_scale * x[self._amp_sl]
        return flat.reshape(self.model.pupil.shape)

    def _channel_phase(self, x: Any) -> Any:
        phase = self._phase_map(x[self._coeffs])
        if not self.fit_tilt:
            return phase
        xp = self.backend.xp
        t = x[self._tilt_sl].reshape(-1, 2)
        t = xp.concatenate([xp.zeros((1, 2), dtype=t.dtype), t])
        return phase[None] + xp.tensordot(t, self._ramps, axes=(1, 0))

    def initial(self, start: Any = None) -> Any:
        """Internal starting vector.

        ``start`` may be ``None`` (flat wavefront), an OPD map in metres
        ``(ny, nx)``, a coefficient vector in metres, or a :class:`Result`
        (its OPD, and its amplitude and tilts when fitted, are reused).
        """
        be, xp = self.backend, self.backend.xp
        x = xp.zeros(self.n_params, dtype=be.real_dtype)
        k_wave = 2.0 * math.pi / self.model.wavelength
        pupil = self.model.pupil
        if isinstance(start, Result):
            opd = np.asarray(be.to_numpy(start.opd), dtype=np.float64)
            x[self._coeffs] = be.asarray(self.basis.fit(opd) * k_wave, dtype="real")
            if self.fit_amplitude and start.amplitude is not None:
                amp = np.asarray(be.to_numpy(start.amplitude))[pupil.mask]
                rel = (amp - pupil.amplitude[pupil.mask]) / self._amp_scale
                x[self._amp_sl] = be.asarray(rel, dtype="real")
            if self.fit_tilt and start.tilts is not None and len(start.tilts) == self.n_channels:
                t = np.asarray(be.to_numpy(start.tilts))[1:] * k_wave
                x[self._tilt_sl] = be.asarray(t.reshape(-1), dtype="real")
        elif start is not None:
            arr = np.asarray(be.to_numpy(start), dtype=np.float64)
            if arr.shape == pupil.shape:
                coeffs = self.basis.fit(arr)
            elif arr.shape == (self.basis.n_modes,):
                coeffs = arr
            else:
                raise ValueError(
                    f"start must be an OPD map {pupil.shape}, {self.basis.n_modes} "
                    f"coefficients or a Result; got shape {arr.shape}"
                )
            x[self._coeffs] = be.asarray(coeffs * k_wave, dtype="real")
        return x

    # ------------------------------------------------------------- objective
    def _evaluate(self, x: Any) -> tuple[Any, Any, Any, Any]:
        """Forward state, OTFs ``S``, Wiener object ``O`` and residuals ``D - O S``."""
        be, xp = self.backend, self.backend.xp
        state = self.model.forward(self._channel_phase(x), self._amplitude(x))
        otf = be.rfft2(state.images)
        if be.is_gpu:  # pragma: no cover - GPU only
            # Fused kernels for the elementwise parts; the sums over channels stay CuPy's.
            num = xp.sum(self._dhat * xp.conj(otf), axis=0)
            den = xp.sum(_kernels.abs2(otf), axis=0) + self.regularization
            obj = num / den
            resid = _kernels.call("pd_resid", self._dhat, obj, otf)
            return state, otf, obj, resid
        num = xp.sum(self._dhat * xp.conj(otf), axis=0)
        den = xp.sum(otf.real**2 + otf.imag**2, axis=0) + self.regularization
        obj = num / den
        resid = self._dhat - obj * otf
        return state, otf, obj, resid

    def _metric(self, obj: Any, resid: Any) -> float:
        be, xp = self.backend, self.backend.xp
        if be.is_gpu:  # pragma: no cover - GPU only
            terms = xp.sum(_kernels.abs2(resid), axis=0)
            if self.regularization:
                rdt = be.real_dtype.type
                weighted = _kernels.call(
                    "pd_terms", terms, self._fweight, rdt(self.regularization), obj
                )
            else:
                weighted = self._fweight * terms
            return float(xp.sum(weighted)) / self._norm
        terms = xp.sum(resid.real**2 + resid.imag**2, axis=0)
        if self.regularization:
            terms = terms + self.regularization * (obj.real**2 + obj.imag**2)
        return float(xp.sum(self._fweight * terms)) / self._norm

    def value(self, x: Any) -> float:
        """Reduced metric at ``x`` (normalized by the noise power)."""
        _, _, obj, resid = self._evaluate(x)
        return self._metric(obj, resid)

    def objective(self, x: Any) -> tuple[float, Any]:
        """Reduced metric and its gradient with respect to the internal vector ``x``."""
        be, xp = self.backend, self.backend.xp
        state, _, obj, resid = self._evaluate(x)
        value = self._metric(obj, resid)
        # dL/d conj(S_k) = -w O* R_k (O fixed: it minimizes the unreduced metric).
        # Through the real FFT, dL/dp = N irfft2(-2 m O* R_k), with m the in-band mask.
        my, mx = self._shape
        scale = -2.0 * my * mx / self._norm
        if be.is_gpu:  # pragma: no cover - GPU only
            rdt = be.real_dtype.type
            g_hat = _kernels.call("pd_grad", rdt(scale), self._mask, obj, resid)
        else:
            g_hat = scale * (self._mask * xp.conj(obj)) * resid
        g_psf = be.irfft2(g_hat, self._shape)
        g_phase, g_amp = self.model.vjp(state, g_psf, amplitude=self.fit_amplitude)
        # Blocks in layout order, joined by one concatenate (not zeros + one scatter each).
        parts = [self._analyze(xp.sum(g_phase, axis=0))]
        if self.fit_amplitude:
            parts.append(self._amp_scale * g_amp.reshape(-1)[self._flat_index])
        if self.fit_tilt:
            gt = xp.tensordot(g_phase[1:], self._ramps, axes=([1, 2], [1, 2]))
            parts.append(gt.reshape(-1))
        grad = xp.concatenate([p.astype(x.dtype, copy=False) for p in parts])
        return value, grad

    def object_estimate(self, x: Any) -> Any:
        """Wiener object estimate ``(my, mx)`` at ``x`` on the data grid, in data units.

        Uses :attr:`object_regularization`. The estimate is registered with the
        optical axis, band-limited to the frequency mask (plus the mean), and
        carries the apodization window and the flux equalization.
        """
        state = self.model.forward(self._channel_phase(x), self._amplitude(x))
        return self._object_image(self.backend.rfft2(state.images))

    def _object_image(self, otf: Any) -> Any:
        xp = self.backend.xp
        num = xp.sum(self._dhat * xp.conj(otf), axis=0)
        den = xp.sum(otf.real**2 + otf.imag**2, axis=0) + self.object_regularization
        obj = num / den * self._object_mask * self._axis_ramp
        return self.backend.irfft2(obj, self._shape)

    # --------------------------------------------------------------- output
    def result(
        self,
        x: Any,
        *,
        opt: OptimizeResult | None = None,
        elapsed: float = 0.0,
        method: str = "phase_diversity",
    ) -> Result:
        """Package an internal vector as a :class:`~solvephase.result.Result`.

        ``model_images`` is the model of the apodized, flux-equalized data
        (``extra["apodized_images"]``). ``extra`` also holds ``"object"``
        (:meth:`object_estimate`), ``"otf"`` (``(K, my, mx // 2 + 1)`` real-FFT
        transfer functions, 1 at zero frequency for an unbounded detector),
        ``"regularization"`` and ``"object_regularization"`` (the gammas),
        ``"noise_power"``, ``"window"`` and
        ``"frequency_mask"`` (real-FFT layout).
        """
        be = self.backend
        k_wave = 2.0 * math.pi / self.model.wavelength
        coeffs = x[self._coeffs]
        phase = self._phase_map(coeffs)
        _, otf, obj, _ = self._evaluate(x)
        model_images = be.irfft2(obj * self._object_mask * otf, self._shape)
        tilts = None
        if self.fit_tilt:
            t = np.zeros((self.n_channels, 2))
            t[1:] = np.asarray(be.to_numpy(x[self._tilt_sl])).reshape(-1, 2) / k_wave
            tilts = t
        loss_value = opt.fun if opt is not None else self.value(x)
        extra = {
            "object": self._object_image(otf),
            "otf": otf,
            "apodized_images": self.apodized,
            "regularization": self.regularization,
            "object_regularization": self.object_regularization,
            "noise_power": self.noise_power,
            "window": self.window,
            "frequency_mask": self.frequency_mask,
        }
        return Result(
            method=method,
            opd=phase / k_wave,
            phase=phase,
            amplitude=self._amplitude(x),
            pupil=self.model.pupil,
            wavelength=self.model.wavelength,
            coefficients=None
            if self.basis.is_zonal
            else np.asarray(be.to_numpy(coeffs), dtype=np.float64) / k_wave,
            basis=None if self.basis.is_zonal else self.basis,
            model_images=model_images,
            tilts=tilts,
            loss=float(loss_value),
            history=list(opt.history) if opt else [],
            times=list(opt.times) if opt else [],
            n_iter=opt.n_iter if opt else 0,
            converged=opt.converged if opt else False,
            message=opt.message if opt else "",
            elapsed=elapsed,
            device=be.device,
            extra=extra,
        )


def phase_diversity(
    model: FocalPlaneModel,
    images: Any,
    *,
    basis: Basis | int | None = None,
    regularization: float | str = "auto",
    object_regularization: float | str = "auto",
    window: Any = "tukey",
    window_alpha: float = 0.25,
    frequency_mask: str | float | None = "diffraction",
    fit_amplitude: bool = False,
    fit_tilt: bool = False,
    equalize_flux: bool = True,
    start: Any = None,
    coarse_to_fine: Sequence[int] | None = None,
    max_iter: int = 300,
    callback: Callable[[int, Any, float], bool | None] | None = None,
    **options: Any,
) -> Result:
    """Retrieve a wavefront from phase-diverse images of an unknown extended object.

    Builds a :class:`PhaseDiversityProblem` (see it for ``basis``,
    ``regularization``, ``object_regularization``, ``window``, ``window_alpha``, ``frequency_mask``,
    ``fit_amplitude``, ``fit_tilt`` and ``equalize_flux``) and minimizes the
    reduced metric with L-BFGS.

    Parameters
    ----------
    model:
        Forward model with one channel per image (diversity included).
    images:
        ``(K, my, mx)`` images of the same scene.
    start:
        Starting point: ``None`` (flat), an OPD map or coefficient vector in
        metres, or a :class:`~solvephase.result.Result`.
    coarse_to_fine:
        Increasing mode counts, e.g. ``(5, 10)``: solve with the first ``n``
        modes of ``basis``, then continue with more, ending with the full
        basis. Widens the capture range for large aberrations. Needs a modal
        basis.
    max_iter:
        L-BFGS iteration limit per stage.
    callback:
        ``callback(iteration, x, f)``; return ``True`` to stop the stage.
    **options:
        Passed to :func:`solvephase.optimize.lbfgs` (``ftol``, ``gtol``,
        ``memory`` ...).

    Returns
    -------
    Result
        ``method="phase_diversity"``; ``coefficients`` are RMS OPD in metres
        and ``extra["object"]`` is the Wiener object estimate on the data grid,
        in data units (see :meth:`PhaseDiversityProblem.result`). A common
        tip/tilt is not measurable and stays at its starting value.

    Examples
    --------
    >>> pupil = Pupil.circular(64)  # doctest: +SKIP
    >>> div = zernike_diversity(pupil, 4, [0.0, 0.29 * 1.6e-6])  # 1 wave PV
    >>> model = FocalPlaneModel(pupil, 1.6e-6, 128, sampling=2.0, diversity=div)
    >>> result = phase_diversity(model, images, basis=21)
    """
    t0 = time.perf_counter()
    problem = PhaseDiversityProblem(
        model,
        images,
        basis=basis,
        regularization=regularization,
        object_regularization=object_regularization,
        window=window,
        window_alpha=window_alpha,
        frequency_mask=frequency_mask,
        fit_amplitude=fit_amplitude,
        fit_tilt=fit_tilt,
        equalize_flux=equalize_flux,
    )
    full = problem.basis
    stages: list[int] = []
    if coarse_to_fine:
        if full.is_zonal:
            raise ValueError("coarse_to_fine needs a modal basis (give basis= a Basis or an int)")
        stages = sorted({int(n) for n in coarse_to_fine if 0 < int(n) < full.n_modes})
    stages.append(full.n_modes)
    be = problem.backend
    history: list[float] = []
    times: list[float] = []
    n_iter = n_eval = 0
    opt: OptimizeResult | None = None
    current: Any = start
    stage_problem = problem
    for n in stages:
        stage_problem = (
            problem if n == full.n_modes else problem.with_basis(full.subset(slice(0, n)))
        )
        x0 = stage_problem.initial(current)
        offset = time.perf_counter() - t0
        opt = lbfgs(
            stage_problem.objective, x0, be, max_iter=max_iter, callback=callback, **options
        )
        history += opt.history
        times += [offset + t for t in opt.times]
        n_iter += opt.n_iter
        n_eval += opt.n_eval
        current = stage_problem.result(opt.x, opt=opt)
    assert opt is not None
    be.synchronize()
    result = stage_problem.result(opt.x, opt=opt, elapsed=time.perf_counter() - t0)
    result.history, result.times, result.n_iter = history, times, n_iter
    result.extra["n_eval"] = n_eval
    return result
