"""Fast & Furious: sequential small-phase focal-plane wavefront sensing.

Fast & Furious (F&F; Keller et al., Proc. SPIE 8447, 2012; Korkiakoski et
al., Appl. Opt. 53, 4565, 2014; on sky: Bos et al., A&A 639, A52, 2020)
measures a pupil-plane wavefront from focal-plane images in a closed loop with
a deformable mirror (DM). It builds on the weak-phase solution of Gonsalves
(Opt. Lett. 26, 684, 2001).

For a weak phase ``phi = phi_e + phi_o`` (even and odd parts under
``x -> -x``) and a real, centro-symmetric pupil amplitude ``A``, the focal
field is ``E = a + i v - y`` to first order, with ``a = F{A}``,
``v = F{A phi_e}`` (real, even) and ``i y = F{A phi_o}`` (``y`` real, odd), so
the image is

    p = a^2 + v^2 + y^2 - 2 a y.

* Its odd part ``p_o = -2 a y`` gives the odd phase directly.
* Its even part ``p_e = a^2 + v^2 + y^2`` gives ``|v|`` but not its sign.
* The sign comes from the previous image, taken with a known phase
  difference ``phi_d`` (the DM change between the two frames):
  ``p_e,prev - p_e = 2 v v_d + v_d^2 + 2 y y_d + y_d^2``.

The sign is taken from that difference and the magnitude from the current
image alone (Korkiakoski et al. 2014, Eq. 19), which keeps the noise of the
two-image difference out of the estimate.

Implemented with the modifications of Korkiakoski et al. (2014):

* images are normalized to the energy of the unaberrated PSF ``a^2`` over the
  detector window (Eq. 12), and a scaled ``a^2`` is added so the image peak
  matches ``max a^2`` (Eq. 13), an improved first-order model that accounts
  for the Strehl loss;
* the division by ``a`` is regularized, ``y = -a p_o / (2 a^2 + epsilon)``
  (Eq. 16; their ``y`` has the opposite sign convention);
* ``|v| = |p_e - a^2 - y^2|^(1/2)`` (absolute value inside the root, Eq. 19);
* the focal-plane field ``v + i y`` is filtered by a concave parabola before
  the inverse transform (Eq. 21), damping noisy high spatial frequencies.

Each step costs one forward FFT (of the DM change) and one inverse FFT, all on
the backend. Units: OPD in metres; images in any linear unit (they are
normalized).

The first frame has no predecessor. :class:`FastAndFurious` then estimates the
odd part and, with ``first_even=True``, the even part with positive signs
everywhere: a valid even phase whose correction is the even diversity the
next frame needs (with only an odd correction the DM change would have no
even part and the signs could never be resolved).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import _kernels
from ..backend import Backend, BackendLike, get_backend
from ..basis import Basis
from ..focal import FocalPlaneModel
from ..propagation import FFTPropagator, _shape
from ..pupil import Pupil

__all__ = ["ClosedLoopResult", "FastAndFurious", "simulate_closed_loop"]


class FastAndFurious:
    """Sequential F&F wavefront sensor with its own previous-image state.

    Parameters
    ----------
    pupil:
        The pupil. F&F assumes a centro-symmetric amplitude (circular,
        annular, symmetric spiders).
    wavelength:
        Monochromatic wavelength in metres.
    image_shape:
        Detector image shape ``(my, mx)``; the optical axis is at the window
        centre.
    sampling, pixel_scale:
        Detector pixels per ``lambda / D`` or pixel scale in radians (give
        one). ``wavelength / (pupil.pitch * pixel_scale)`` must be an integer
        FFT size at least as large as the pupil and image grids, e.g.
        ``sampling=2`` with a pupil filling its grid.
    epsilon:
        Regularization of the division by ``a`` (Korkiakoski et al. Eq. 16),
        relative to the unaberrated peak ``max a^2 = 1``. Korkiakoski et al.
        recommend 50-500 times the per-pixel noise level of the normalized
        image. Smaller values recover more of the faint-ring signal (higher
        spatial frequencies) but amplify noise there.
    filter_radius:
        Radius of the parabolic focal-plane filter ``max(0, 1 - (r / R)^2)``
        in ``lambda / D``. Default: the largest circle inside the detector
        window.
    strehl_compensation:
        Add the scaled ``a^2`` term of Korkiakoski et al. Eq. (13).
    first_even:
        On the first frame, estimate the even part with positive signs (see
        the module docstring). With False the first estimate is odd only.
    device, precision:
        Backend selection.

    Examples
    --------
    >>> ff = FastAndFurious(pupil, 1.6e-6, 64, sampling=2)  # doctest: +SKIP
    >>> change = None
    >>> for _ in range(20):  # doctest: +SKIP
    ...     estimate = ff.step(camera.read(), change)
    ...     change = -0.5 * estimate  # wavefront change made by the DM
    ...     dm.apply(change)
    """

    def __init__(
        self,
        pupil: Pupil,
        wavelength: float,
        image_shape: Any,
        *,
        sampling: float | None = None,
        pixel_scale: float | None = None,
        epsilon: float = 1e-4,
        filter_radius: float | None = None,
        strehl_compensation: bool = True,
        first_even: bool = True,
        device: BackendLike = "cpu",
        precision: str | None = None,
    ) -> None:
        self.backend: Backend = get_backend(device, precision)
        be = self.backend
        self.pupil = pupil
        self.wavelength = float(wavelength)
        if not (self.wavelength > 0 and math.isfinite(self.wavelength)):
            raise ValueError(
                "wavelength must be a positive number of metres (F&F is monochromatic)"
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
        samples = self.wavelength / (pupil.pitch * self.pixel_scale)
        n_fft = round(samples)
        if abs(samples - n_fft) > 1e-6 * samples or n_fft < max(*pupil.shape, *self.image_shape):
            raise ValueError(
                "Fast & Furious inverts the pupil-to-focal FFT, so wavelength / (pupil pitch x "
                f"pixel scale) = {samples:.6g} must be an integer at least as large as the pupil "
                f"{pupil.shape} and image {self.image_shape} grids; adjust sampling or the "
                "pupil grid"
            )
        self.n_fft = n_fft
        if epsilon < 0:
            raise ValueError("epsilon must be >= 0")
        self.epsilon = float(epsilon)
        self.strehl_compensation = bool(strehl_compensation)
        self.first_even = bool(first_even)

        amp = pupil.amplitude
        if not np.allclose(amp, amp[::-1, ::-1], atol=1e-6 * float(amp.max())):
            raise ValueError(
                "Fast & Furious assumes a centro-symmetric pupil amplitude (Korkiakoski et al. "
                "2014); this pupil is not symmetric under a 180-degree rotation"
            )
        self.propagator = FFTPropagator(pupil.shape, self.image_shape, n_fft, backend=be)
        self._k = 2.0 * math.pi / self.wavelength
        a_field = self.propagator.forward(be.asarray(amp, dtype="complex"))
        a_host = np.asarray(be.to_numpy(a_field.real), dtype=np.float64)
        peak = float(np.max(np.abs(a_host)))
        self._scale = 1.0 / peak  # F_n = scale * F has max |a| = 1
        a = a_host * self._scale
        self._a = be.asarray(a, dtype="real")
        self._a2 = be.asarray(a * a, dtype="real")
        self._sum_a2 = float(np.sum(a * a))
        self._max_a2 = float(np.max(a * a))
        self._y_den = be.asarray(2.0 * a * a + self.epsilon, dtype="real")

        my, mx = self.image_shape
        samp = self.sampling
        fy = (np.arange(my) - (my - 1) / 2.0)[:, None] / samp
        fx = (np.arange(mx) - (mx - 1) / 2.0)[None, :] / samp
        radius = (min(my, mx) / 2.0) / samp if filter_radius is None else float(filter_radius)
        if radius <= 0:
            raise ValueError("filter_radius must be positive (lambda / D)")
        self.filter_radius = radius
        window = np.maximum(0.0, 1.0 - (fy**2 + fx**2) / radius**2)
        self._window = be.asarray(window / self._scale, dtype="real")  # folds in 1 / scale

        self._amp = be.asarray(amp, dtype="real")
        floor = (0.25 * float(amp.max())) ** 2
        inv = np.where(amp > 0, amp / np.maximum(amp * amp, floor), 0.0)
        self._inv_amp = be.asarray(inv / self._k, dtype="real")  # also radians -> metres
        self._prev_even: Any = None
        self.n_steps = 0
        self.last_odd_opd: Any = None
        self.last_even_opd: Any = None

    # ------------------------------------------------------------- metadata
    @property
    def sampling(self) -> float:
        """Detector pixels per ``lambda / D``."""
        return self.wavelength / self.pupil.diameter / self.pixel_scale

    @property
    def has_previous(self) -> bool:
        """Whether a previous image is stored (the even-part signs can be resolved)."""
        return self._prev_even is not None

    def __repr__(self) -> str:
        return (
            f"FastAndFurious(image_shape={self.image_shape}, sampling={self.sampling:.3g}, "
            f"n_fft={self.n_fft}, epsilon={self.epsilon:g}, steps={self.n_steps}, "
            f"backend={self.backend})"
        )

    def reset(self) -> None:
        """Forget the previous image (e.g. after opening the loop)."""
        self._prev_even = None
        self.n_steps = 0
        self.last_odd_opd = None
        self.last_even_opd = None

    # ------------------------------------------------------------ the step
    def _normalize(self, image: Any) -> Any:
        """Korkiakoski et al. Eqs. (12)-(13): energy normalization and Strehl compensation."""
        xp = self.backend.xp
        p = self.backend.asarray(image, dtype="real")
        if p.ndim == 3 and p.shape[0] == 1:
            p = p[0]
        if tuple(p.shape) != self.image_shape:
            raise ValueError(f"image shape {tuple(p.shape)} does not match {self.image_shape}")
        total = xp.sum(p)
        if self._fused(p):
            # One kernel after the two reductions. max(c p) == c max(p) exactly for
            # c > 0 (rounding is monotonic), so the peak is taken before scaling.
            rdt = self.backend.real_dtype.type
            if self.strehl_compensation:
                return _kernels.call(
                    "ff_normalize",
                    p,
                    total,
                    xp.max(p),
                    rdt(self._sum_a2),
                    rdt(self._max_a2),
                    self._a2,
                )
            return _kernels.call("ff_normalize_plain", p, total, rdt(self._sum_a2))
        p = p * (self._sum_a2 / xp.where(total > 0, total, 1.0))
        if self.strehl_compensation:
            p = p + (1.0 - xp.max(p) / self._max_a2) * self._a2
        return p

    def _fused(self, p: Any) -> bool:
        """Whether the fused GPU kernels apply (contiguous working-precision image)."""
        return bool(
            self.backend.is_gpu and p.dtype == self.backend.real_dtype and p.flags.c_contiguous
        )

    def _pupil_opd(self, focal: Any) -> Any:
        """OPD in metres from the focal field ``F{A phi}`` (one inverse FFT, divide by ``A``)."""
        a_phi = self.propagator.adjoint(focal).real
        return a_phi * self._inv_amp

    def step(self, image: Any, dm_change_opd: Any = None) -> Any:
        """Process one image and return the estimated pupil OPD.

        Parameters
        ----------
        image:
            ``(my, mx)`` focal-plane image (any linear unit, background
            subtracted), NumPy or backend array.
        dm_change_opd:
            ``(ny, nx)`` change of the wavefront OPD in metres made between
            the previous image and this one (the DM correction applied after
            the previous step, as the camera sees it). Ignored on the first
            frame; ``None`` means no change, so the even-part signs are not
            resolved and the even estimate is zero.

        Returns
        -------
        Backend array ``(ny, nx)``: the estimated wavefront OPD in metres at
        this image, zero outside the pupil. Apply ``-gain`` times it with the
        DM to correct. The odd and even parts are kept in
        :attr:`last_odd_opd` and :attr:`last_even_opd`.
        """
        xp = self.backend.xp
        p = self._normalize(image)
        if self._fused(p):
            return self._step_fused(p, dm_change_opd)
        p_flip = p[::-1, ::-1]
        p_even = 0.5 * (p + p_flip)
        p_odd = 0.5 * (p - p_flip)
        y = -self._a * p_odd / self._y_den
        v_abs = xp.sqrt(xp.abs(p_even - self._a2 - y * y))
        if self._prev_even is None:
            sign = 1.0 if self.first_even else 0.0
            v = v_abs * sign
        elif dm_change_opd is None:
            v = xp.zeros_like(v_abs)
        else:
            div = self._change_field(dm_change_opd)
            v_d, y_d = div.real, div.imag
            diff = self._prev_even - p_even - v_d * v_d - y_d * y_d - 2.0 * y * y_d
            v = v_abs * xp.sign(diff * v_d)
        self._prev_even = p_even
        self.n_steps += 1
        # v + i y = F{A phi_e} + F{A phi_o}: one inverse FFT gives A phi, whose even and odd
        # parts are separated again by symmetry.
        opd = self._pupil_opd((v + 1j * y) * self._window)
        opd_flip = opd[::-1, ::-1]
        self.last_odd_opd = 0.5 * (opd - opd_flip)
        self.last_even_opd = 0.5 * (opd + opd_flip)
        return opd

    def _change_field(self, dm_change_opd: Any) -> Any:
        """Normalized focal field of the previous frame's phase relative to this one."""
        change = self.backend.asarray(dm_change_opd, dtype="real")
        if tuple(change.shape) != self.pupil.shape:
            raise ValueError(
                f"dm_change_opd shape {tuple(change.shape)} does not match the pupil "
                f"{self.pupil.shape}"
            )
        # Previous frame relative to this one: phi_d = -k * change.
        be = self.backend
        if be.is_gpu and change.flags.c_contiguous:
            field = be.empty(self.pupil.shape, dtype="complex")
            _kernels.call("ff_change", self._amp, be.real_dtype.type(-self._k), change, field)
        else:
            field = (self._amp * (-self._k) * change).astype(be.complex_dtype)
        return self.propagator.forward(field) * self._scale

    def _step_fused(self, p: Any, dm_change_opd: Any) -> Any:
        """:meth:`step` on the GPU in five fused kernels plus the FFTs (same arithmetic)."""
        xp = self.backend.xp
        rdt = self.backend.real_dtype.type
        p_even, y, v_abs = _kernels.call("ff_parts", p, self._a, self._a2, self._y_den)
        if self._prev_even is None or dm_change_opd is None:
            # First frame: v = |v| (or 0 without first_even); no DM change: v = 0.
            first = self._prev_even is None
            sign = rdt(1.0 if first and self.first_even else 0.0)
            field = xp.empty(p.shape, dtype=self.backend.complex_dtype)
            _kernels.call("ff_field", v_abs if first else rdt(0.0), sign, y, self._window, field)
        else:
            div = self._change_field(dm_change_opd)
            field = _kernels.call(
                "ff_field_signed", v_abs, y, self._window, self._prev_even, p_even, div
            )
        self._prev_even = p_even
        self.n_steps += 1
        a_phi = self.propagator.adjoint(field)
        opd, odd, even = (xp.empty(self.pupil.shape, dtype=p.dtype) for _ in range(3))
        _kernels.call("ff_opd", a_phi, self._inv_amp, opd, odd, even)
        self.last_odd_opd, self.last_even_opd = odd, even
        return opd


@dataclass
class ClosedLoopResult:
    """Record of :func:`simulate_closed_loop`.

    Attributes
    ----------
    residual_rms:
        ``(n_iter + 1,)`` piston-removed, intensity-weighted RMS residual OPD
        in metres before each frame (entry 0 is the open-loop aberration).
    strehl:
        Marechal Strehl ratio of each :attr:`residual_rms`.
    residual_opd:
        Final residual OPD map in metres (backend array).
    dm_opd:
        Final DM OPD in metres (backend array; the correction it applies).
    step_times:
        Wall-clock seconds of each F&F step (synchronized on the GPU).
    history:
        :attr:`residual_rms` as a list.
    n_iter:
        Loop iterations run.
    converged:
        Whether the residual fell below ``tol``.
    message:
        Why the loop stopped.
    elapsed:
        Total wall-clock seconds.
    device:
        ``"cpu"`` or ``"gpu"``.
    """

    residual_rms: np.ndarray
    strehl: np.ndarray
    residual_opd: Any
    dm_opd: Any
    step_times: np.ndarray
    history: list[float] = field(default_factory=list)
    n_iter: int = 0
    converged: bool = False
    message: str = ""
    elapsed: float = 0.0
    device: str = "cpu"


def simulate_closed_loop(
    ff: FastAndFurious,
    true_opd: Any,
    n_iterations: int = 20,
    *,
    gain: float = 0.5,
    leak: float = 1.0,
    photons: float | None = None,
    read_noise: float = 0.0,
    seed: Any = None,
    basis: Basis | None = None,
    tol: float | None = None,
    reset: bool = True,
) -> ClosedLoopResult:
    """Run F&F in a simulated closed loop against a static aberration.

    Each iteration images the residual ``true_opd + dm`` (monochromatic,
    :class:`~solvephase.FocalPlaneModel` with the sensor's sampling), adds
    noise, calls :meth:`FastAndFurious.step` with the previous DM change, and
    updates the DM with the leaky integrator of Korkiakoski et al. (Eq. 22),
    ``dm <- leak * dm - gain * estimate``.

    Parameters
    ----------
    ff:
        The sensor (its state is reset first when ``reset``).
    true_opd:
        ``(ny, nx)`` static aberration in metres (e.g. NCPA).
    n_iterations:
        Number of frames.
    gain, leak:
        Integrator gain and leak (``leak = 1`` is a pure integrator).
    photons:
        Detected photons per frame (full PSF) for Poisson noise; ``None``
        gives noise-free images.
    read_noise:
        Gaussian read noise per pixel in photo-electrons (with ``photons``).
    seed:
        Seed or :class:`numpy.random.Generator` for the noise (host draws).
    basis:
        Optional DM modal basis: each estimate is projected onto it (a DM
        cannot make arbitrary shapes). ``None`` is a zonal DM at pupil
        resolution.
    tol:
        Stop once the residual RMS (metres) is below this.
    reset:
        Call :meth:`FastAndFurious.reset` first.

    Returns
    -------
    ClosedLoopResult
    """
    t0 = time.perf_counter()
    if n_iterations < 1:
        raise ValueError("n_iterations must be >= 1")
    if not 0.0 < gain <= 2.0 or not 0.0 <= leak <= 1.0:
        raise ValueError("gain must be in (0, 2] and leak in [0, 1]")
    if photons is not None and photons <= 0:
        raise ValueError("photons must be positive (or None for noise-free images)")
    be, xp = ff.backend, ff.backend.xp
    if reset:
        ff.reset()
    model = FocalPlaneModel(
        ff.pupil, ff.wavelength, ff.image_shape, pixel_scale=ff.pixel_scale, device=be
    )
    rng = be.random(seed)
    truth = be.asarray(true_opd, dtype="real")
    if tuple(truth.shape) != ff.pupil.shape:
        raise ValueError(
            f"true_opd shape {tuple(truth.shape)} does not match the pupil {ff.pupil.shape}"
        )
    mask = be.asarray(ff.pupil.mask, dtype="real")
    weight = be.asarray(ff.pupil.amplitude**2, dtype="real")
    w_sum = float(np.sum(ff.pupil.amplitude**2))
    projector = None
    if basis is not None:
        if basis.is_zonal:
            basis = None
        else:
            modes = basis.modes
            assert modes is not None
            w = (ff.pupil.amplitude**2)[ff.pupil.mask]
            pinv = np.linalg.pinv((modes * np.sqrt(w)).T) * np.sqrt(w)[None, :]
            projector = (be.asarray(pinv.T, dtype="real"), basis.on(be))
            flat_index = xp.asarray(np.flatnonzero(ff.pupil.mask.ravel()))

    def residual_rms(opd: Any) -> Any:
        mean = xp.sum(weight * opd) / w_sum
        return xp.sqrt(xp.sum(weight * (opd - mean) ** 2) / w_sum)

    dm = be.zeros(ff.pupil.shape)
    change: Any = None
    rms_values = [residual_rms(truth)]
    step_times: list[float] = []
    converged, message = False, f"completed {n_iterations} iterations"
    it = 0
    for it in range(1, n_iterations + 1):
        residual = truth + dm
        image = model.images(residual * mask)[0]
        if photons is not None:
            host = np.asarray(be.to_numpy(image), dtype=np.float64) * photons
            noisy = rng.poisson(np.maximum(host, 0.0)).astype(np.float64)
            if read_noise:
                noisy = noisy + rng.normal(0.0, read_noise, noisy.shape)
            image = be.asarray(noisy, dtype="real")
        be.synchronize()
        ts = time.perf_counter()
        estimate = ff.step(image, change)
        be.synchronize()
        step_times.append(time.perf_counter() - ts)
        if projector is not None:
            pinv_t, modes_d = projector
            coeffs = estimate.reshape(-1)[flat_index] @ pinv_t
            flat = xp.zeros(estimate.size, dtype=estimate.dtype)
            flat[flat_index] = coeffs @ modes_d
            estimate = flat.reshape(estimate.shape)
        new_dm = leak * dm - gain * estimate * mask
        change = new_dm - dm
        dm = new_dm
        rms_values.append(residual_rms(truth + dm))
        if tol is not None and float(rms_values[-1]) < tol:
            converged, message = True, f"residual RMS below tol after {it} iterations"
            break
    be.synchronize()
    rms_host = np.array([float(r) for r in rms_values])
    strehl = np.exp(-((2.0 * math.pi * rms_host / ff.wavelength) ** 2))
    return ClosedLoopResult(
        residual_rms=rms_host,
        strehl=strehl,
        residual_opd=(truth + dm) * mask,
        dm_opd=dm,
        step_times=np.asarray(step_times),
        history=list(rms_host),
        n_iter=it,
        converged=converged,
        message=message,
        elapsed=time.perf_counter() - t0,
        device=be.device,
    )
