"""Focal-plane image formation with analytic derivatives.

:class:`FocalPlaneModel` turns a pupil OPD into a stack of ``K`` detector
images, one per *channel*. Each channel adds a known diversity OPD (defocus,
astigmatism, a DM probe ...) to the unknown wavefront. Light is broadband: the
image is the spectrally weighted sum of monochromatic PSFs, each propagated
with its own wavelength-scaled sampling. Detector pixels can be integrated by
modelling an ``oversample``-times finer grid and summing blocks.

The model exposes three operations, all batched over channels and
wavelengths and running on the selected device:

* :meth:`~FocalPlaneModel.forward` - images from a phase map;
* :meth:`~FocalPlaneModel.vjp` - the gradient of any scalar function of the
  images with respect to phase and amplitude (reverse mode, one adjoint
  propagation);
* :meth:`~FocalPlaneModel.jvp` - image derivatives along given phase
  directions (forward mode), used to build Gauss-Newton Jacobians.

Units: phase maps are radians at the *reference wavelength*
(:attr:`~FocalPlaneModel.wavelength`); OPD is ``phase * wavelength / 2 pi``.
Images are normalized so that, for an unbounded detector, each channel sums
to 1 (the fraction of the pupil's light per pixel).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import _kernels
from .backend import Backend, BackendLike, get_backend
from .basis import Basis
from .propagation import FocalPlanePropagator, _pair, _shape
from .pupil import Pupil

__all__ = ["FocalPlaneModel", "ForwardState", "zernike_diversity"]


@dataclass
class ForwardState:
    """Intermediates of :meth:`FocalPlaneModel.forward`, reused by its ``vjp``."""

    images: Any  # (K, my, mx) normalized images
    pupil_field: Any  # (K, L, ny, nx)
    focal_field: Any  # (K, L, My, Mx) on the oversampled grid
    amplitude: Any  # (ny, nx)
    phasor: Any  # (K, L, ny, nx) exp(i phase)


def _unit_phasor(backend: Backend, phi: Any) -> Any:
    """``exp(1j * phi)`` in the backend's complex dtype.

    On the CPU it is built as ``cos + i sin``: the same values to the bit,
    about 1.4x faster than NumPy's complex ``exp``.
    """
    xp = backend.xp
    if backend.is_gpu:  # pragma: no cover - GPU only
        return xp.exp(1j * phi).astype(backend.complex_dtype, copy=False)
    out = xp.empty(phi.shape, dtype=backend.complex_dtype)
    xp.cos(phi, out=out.real)
    xp.sin(phi, out=out.imag)
    return out


def zernike_diversity(
    pupil: Pupil, noll: int, amounts: Sequence[float], *, annular: bool | None = None
) -> np.ndarray:
    """Diversity OPD maps ``(K, ny, nx)``: one Zernike mode at several RMS amplitudes.

    Parameters
    ----------
    pupil:
        The pupil.
    noll:
        Noll index of the mode (4 = defocus, 5/6 = astigmatism).
    amounts:
        RMS OPD of the mode for each channel, in metres (0 for an in-focus
        channel).

    Examples
    --------
    Two images, in focus and with 0.5 waves RMS of defocus at 1.6 um:

    >>> div = zernike_diversity(pupil, 4, [0.0, 0.8e-6])  # doctest: +SKIP
    """
    mode = Basis.zernike(pupil, 1, start=noll, annular=annular).mode_maps()[0]
    return np.asarray(amounts, dtype=np.float64)[:, None, None] * mode[None]


class FocalPlaneModel:
    """Pupil-to-detector forward model for one or more diversity channels.

    Parameters
    ----------
    pupil:
        The entrance pupil.
    wavelength:
        Wavelength in metres, or an array of wavelengths sampling a band.
    image_shape:
        Detector image shape ``(my, mx)`` (an int for square images).
    pixel_scale:
        Detector pixel scale in radians. Give this or ``sampling``.
    sampling:
        Detector pixels per ``lambda_ref / D`` (2 = Nyquist), an alternative to
        ``pixel_scale``.
    weights:
        Relative spectral weights (photon flux) of the wavelengths; normalized
        to sum to 1. Uniform by default.
    reference_wavelength:
        Wavelength at which phase is expressed. Defaults to the
        weight-averaged wavelength.
    diversity:
        Known OPD added in each channel, ``(K, ny, nx)`` in metres (or a
        sequence of maps; ``None`` entries mean zero). ``None`` gives a single
        channel with no diversity.
    oversample:
        Model each detector pixel with ``oversample x oversample`` samples
        and integrate (sum) them, which models the pixel's response.
    offset:
        Detector-window offset in pixels ``(dy, dx)``: pixel ``j`` samples the
        angle ``(j - (m - 1) / 2 + offset) * pixel_scale``, so the optical axis
        falls on pixel ``(m - 1) / 2 - offset``. ``offset=-0.5`` puts it on
        pixel ``m / 2`` of an even-sized image (the HCIPy/``fftshift``
        convention).
    method:
        Propagation engine: ``"auto"``, ``"fft"`` or ``"mft"``.
    device, precision:
        Backend selection, see :func:`solvephase.get_backend`.
    """

    def __init__(
        self,
        pupil: Pupil,
        wavelength: float | Sequence[float],
        image_shape: Any,
        *,
        pixel_scale: float | None = None,
        sampling: float | None = None,
        weights: Sequence[float] | None = None,
        reference_wavelength: float | None = None,
        diversity: Any = None,
        oversample: int = 1,
        offset: Any = 0.0,
        method: str = "auto",
        device: BackendLike = "cpu",
        precision: str | None = None,
    ) -> None:
        self.backend: Backend = get_backend(device, precision)
        self.pupil = pupil
        self.wavelengths = np.atleast_1d(np.asarray(wavelength, dtype=np.float64))
        if self.wavelengths.ndim != 1 or np.any(self.wavelengths <= 0):
            raise ValueError("wavelength must be positive (a scalar or 1-D array)")
        if weights is None:
            spectral = np.ones_like(self.wavelengths)
        else:
            spectral = np.asarray(weights, dtype=np.float64)
            if spectral.shape != self.wavelengths.shape or np.any(spectral < 0):
                raise ValueError("weights must be non-negative, one per wavelength")
        if spectral.sum() <= 0:
            raise ValueError("spectral weights sum to zero")
        self.spectral_weights = spectral / spectral.sum()
        self.wavelength = float(
            reference_wavelength
            if reference_wavelength is not None
            else np.sum(self.spectral_weights * self.wavelengths)
        )
        self.image_shape = _shape(image_shape, "image_shape")
        if (pixel_scale is None) == (sampling is None):
            raise ValueError(
                "give exactly one of pixel_scale (radians) or sampling (pixels per lambda/D)"
            )
        if sampling is not None:
            if sampling <= 0:
                raise ValueError("sampling must be positive")
            pixel_scale = self.wavelength / pupil.diameter / float(sampling)
        assert pixel_scale is not None
        if pixel_scale <= 0:
            raise ValueError("pixel_scale must be positive")
        self.pixel_scale = float(pixel_scale)
        if int(oversample) != oversample or oversample < 1:
            raise ValueError("oversample must be a positive integer")
        self.oversample = int(oversample)
        self.offset = _pair(offset, "offset")

        xp = self.backend.xp
        rdt = self.backend.real_dtype
        div: np.ndarray
        if diversity is None:
            div = np.zeros((1, *pupil.shape))
        else:
            maps = [
                np.zeros(pupil.shape) if d is None else np.asarray(d, dtype=np.float64)
                for d in diversity
            ]
            div = np.stack(maps) if maps else np.zeros((1, *pupil.shape))
            if div.ndim == 2:
                div = div[None]
        if div.shape[1:] != pupil.shape:
            raise ValueError(f"diversity maps {div.shape[1:]} do not match the pupil {pupil.shape}")
        if not np.all(np.isfinite(div)):
            raise ValueError("diversity contains non-finite values")
        self.diversity_opd = div
        self._k = 2.0 * math.pi / self.wavelength
        self._div = xp.asarray(div * self._k, dtype=rdt)
        self._amp = xp.asarray(pupil.amplitude, dtype=rdt)
        self._mask = xp.asarray(pupil.mask)
        ratio = self.wavelength / self.wavelengths
        self._ratio = xp.asarray(ratio[:, None, None], dtype=rdt)
        self._wl = xp.asarray(self.spectral_weights[:, None, None], dtype=rdt)
        self._norm = float(np.sum(pupil.amplitude**2))
        # Constant factors hoisted out of forward/vjp/jvp (same values as in-line).
        self._unit_ratio = bool(np.all(ratio == 1.0))  # one wavelength: phi * ratio == phi
        self._wl_grad = self._wl * (2.0 / self._norm)
        os_ = self.oversample
        my, mx = self.image_shape
        self.propagator = FocalPlanePropagator(
            pupil.shape,
            pupil.pitch,
            self.wavelengths,
            self.pixel_scale / os_,
            (my * os_, mx * os_),
            offset=(self.offset[0] * os_, self.offset[1] * os_),
            method=method,
            backend=self.backend,
        )

    # -------------------------------------------------------------- metadata
    @property
    def n_channels(self) -> int:
        """Number of diversity channels ``K``."""
        return int(self.diversity_opd.shape[0])

    @property
    def n_wavelengths(self) -> int:
        """Number of wavelength samples ``L``."""
        return int(self.wavelengths.size)

    @property
    def sampling(self) -> float:
        """Detector pixels per ``lambda_ref / D``."""
        return self.wavelength / self.pupil.diameter / self.pixel_scale

    @property
    def nyquist_sampled(self) -> bool:
        """Whether the shortest wavelength has at least 2 pixels per lambda/D."""
        return bool(self.wavelengths.min() / self.pupil.diameter / self.pixel_scale >= 2.0 - 1e-9)

    def opd_to_phase(self, opd: Any) -> Any:
        """Convert OPD in metres to phase in radians at the reference wavelength."""
        return opd * self._k

    def phase_to_opd(self, phase: Any) -> Any:
        """Convert reference-wavelength phase in radians to OPD in metres."""
        return phase / self._k

    def __repr__(self) -> str:
        return (
            f"FocalPlaneModel(pupil={self.pupil.name!r}, channels={self.n_channels}, "
            f"wavelengths={self.n_wavelengths}, image_shape={self.image_shape}, "
            f"sampling={self.sampling:.3g} px/(lambda/D), oversample={self.oversample}, "
            f"engine={self.propagator.method!r}, backend={self.backend})"
        )

    # ------------------------------------------------------------- binning
    def _bin(self, images: Any) -> Any:
        os_ = self.oversample
        if os_ == 1:
            return images
        my, mx = self.image_shape
        lead = images.shape[:-2]
        return images.reshape(*lead, my, os_, mx, os_).sum(axis=(-3, -1))

    def _unbin(self, images: Any) -> Any:
        os_ = self.oversample
        if os_ == 1:
            return images
        xp = self.backend.xp
        return xp.repeat(xp.repeat(images, os_, axis=-2), os_, axis=-1)

    # ------------------------------------------------------------- forward
    def _phase_stack(self, phase: Any) -> Any:
        """Broadcast a phase map (or per-channel maps) to ``(K, L, ny, nx)`` phase."""
        if phase.ndim == 2:
            phase = phase[None]
        phi = (phase + self._div)[:, None, :, :]
        return phi if self._unit_ratio else phi * self._ratio

    def _sum_wavelengths(self, values: Any) -> Any:
        """Sum ``(..., L, y, x)`` over ``L``; a view, not a copy, for one wavelength."""
        if values.shape[-3] == 1:
            return values[..., 0, :, :]
        return self.backend.xp.sum(values, axis=-3)

    def forward(self, phase: Any, amplitude: Any = None) -> ForwardState:
        """Model images for a phase map.

        Parameters
        ----------
        phase:
            ``(ny, nx)`` phase in radians at the reference wavelength (shared by
            all channels), or ``(K, ny, nx)`` per channel. Backend array.
        amplitude:
            Optional ``(ny, nx)`` pupil amplitude; defaults to the pupil's.

        Returns
        -------
        ForwardState
            ``state.images`` holds the ``(K, my, mx)`` normalized images.
        """
        be = self.backend
        xp = be.xp
        amp = self._amp if amplitude is None else amplitude
        if be.is_gpu and phase.dtype == amp.dtype == be.real_dtype:  # pragma: no cover - GPU only
            # One fused kernel for phase, exp(i phi) and amp * exp(i phi).
            ph = phase[None] if phase.ndim == 2 else phase
            phasor, u = _kernels.pupil_field(
                ph[:, None], self._div[:, None], self._ratio, amp, be.complex_dtype
            )
            e = self.propagator.forward(u)
            if self.n_wavelengths == 1:
                inten = _kernels.call("intensity", e, self._wl, self._norm)[:, 0]
            else:
                inten = xp.sum(_kernels.call("intensity", e, self._wl, 1.0), axis=-3) / self._norm
        else:
            phasor = _unit_phasor(be, self._phase_stack(phase))
            u = amp * phasor
            e = self.propagator.forward(u)
            inten = self._sum_wavelengths(self._wl * (e.real**2 + e.imag**2)) / self._norm
        return ForwardState(
            images=self._bin(inten), pupil_field=u, focal_field=e, amplitude=amp, phasor=phasor
        )

    def images(self, opd: Any = None, *, amplitude: Any = None) -> Any:
        """Convenience: normalized ``(K, my, mx)`` images for an OPD map in metres.

        Accepts NumPy or backend arrays; returns a backend array.
        """
        if opd is None:
            phase = self.backend.zeros(self.pupil.shape)
        else:
            phase = self.backend.asarray(opd, dtype="real") * self._k
        amp = None if amplitude is None else self.backend.asarray(amplitude, dtype="real")
        return self.forward(phase, amp).images

    # ------------------------------------------------------------- reverse
    def vjp(
        self, state: ForwardState, grad_images: Any, *, amplitude: bool = False
    ) -> tuple[Any, Any]:
        """Back-propagate ``dL/d images`` to the phase (and optionally amplitude).

        Parameters
        ----------
        state:
            The :class:`ForwardState` of the matching :meth:`forward` call.
        grad_images:
            ``(K, my, mx)`` real gradient of a scalar with respect to
            ``state.images``.
        amplitude:
            Also return the amplitude gradient.

        Returns
        -------
        grad_phase, grad_amplitude
            ``(K, ny, nx)`` per-channel phase gradient (sum over channels for
            a shared phase), and ``(ny, nx)`` amplitude gradient or ``None``.
        """
        be, xp = self.backend, self.backend.xp
        g = self._unbin(grad_images)[:, None, :, :]
        if be.is_gpu and g.dtype == be.real_dtype:  # pragma: no cover - GPU only
            v = self.propagator.adjoint(
                _kernels.call("scale_field", g, state.focal_field, self._wl_grad)
            )
            cross = _kernels.call("cross_imag", state.pupil_field, v, self._ratio)
        else:
            v = self.propagator.adjoint(g * state.focal_field * self._wl_grad)
            cross = (xp.conj(state.pupil_field) * v).imag
            if not self._unit_ratio:
                cross = cross * self._ratio
        grad_phase = self._sum_wavelengths(cross)
        grad_amp = None
        if amplitude:
            # d u / d a = exp(i phi): exact everywhere, including a <= 0.
            grad_amp = xp.sum((xp.conj(state.phasor) * v).real, axis=(0, 1))
        return grad_phase, grad_amp

    # ------------------------------------------------------------- forward-mode
    def jvp(self, state: ForwardState, directions: Any, *, per_channel: bool = False) -> Any:
        """Image derivatives along phase directions.

        Parameters
        ----------
        state:
            The :class:`ForwardState` at the linearization point.
        directions:
            ``(P, ny, nx)`` phase directions in radians (applied to every
            channel), or with ``per_channel=True`` ``(P, K, ny, nx)``.

        Returns
        -------
        ``(P, K, my, mx)`` array of ``d images / d direction``.
        """
        be = self.backend
        xp = be.xp
        d = directions[:, None] if not per_channel else directions
        scale = 2.0 / self._norm
        if be.is_gpu and d.dtype == be.real_dtype:  # pragma: no cover - GPU only
            du = _kernels.call("direction_field", d[:, :, None], self._ratio, state.pupil_field)
            de = self.propagator.forward(du)
            if self.n_wavelengths == 1:
                di = _kernels.call("cross_real", state.focal_field, de, self._wl, scale)[:, :, 0]
            else:
                cross = _kernels.call("cross_real", state.focal_field, de, self._wl, 1.0)
                di = xp.sum(cross, axis=-3) * scale
            return self._bin(di)
        du = (1j * d[:, :, None, :, :] * self._ratio) * state.pupil_field
        de = self.propagator.forward(du.astype(be.complex_dtype, copy=False))
        cross = xp.conj(state.focal_field) * de
        di = self._sum_wavelengths(self._wl * cross.real) * scale
        return self._bin(di)
