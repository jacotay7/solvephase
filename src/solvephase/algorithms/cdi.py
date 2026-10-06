"""Coherent diffraction imaging (CDI): iterative projection algorithms.

A compact object is illuminated coherently and only the modulus of its
far-field (Fourier) transform is measured. When the diffraction pattern is
oversampled (the object occupies less than half of the reconstruction window
along each axis, Miao, Sayre & Chapman, JOSA A 15, 1662, 1998) the phase is
fixed by the measured modulus together with an object-domain constraint (a
support, and optionally realness or positivity). The algorithms here
alternate projections onto the two constraint sets:

* ``P_M`` - the Fourier-modulus projection: keep the phase of the current
  transform and impose the measured modulus on measured pixels;
* ``P_S`` - the object-domain projection: zero outside the support and, inside
  it, the nearest complex/real/non-negative value (and amplitude bounds).

Implemented update rules (``x`` is the iterate, ``y = P_M x``,
``R = 2 P - I``):

* **ER**, error reduction ``x <- P_S P_M x`` (Gerchberg & Saxton, Optik 35,
  237, 1972; Fienup, Opt. Lett. 3, 27, 1978; Appl. Opt. 21, 2758, 1982);
* **HIO**, hybrid input-output (Fienup 1982): ``y`` where it satisfies the
  object constraints, ``x - beta y`` elsewhere;
* **DM**, the difference map (Elser, JOSA A 20, 40, 2003) with
  ``gamma_S = -1/beta`` and ``gamma_M = 1/beta``;
* **RAAR**, relaxed averaged alternating reflections (Luke, Inverse Problems
  21, 37, 2005);
* **RRR**, relax-reflect-reflect (Elser, Lan & Bendory, SIAM J. Imaging
  Sci. 11, 2429, 2018);
* **ASR**, averaged successive reflections = Douglas-Rachford (Bauschke,
  Combettes & Luke, JOSA A 19, 1334, 2002);
* **HPR**, hybrid projection-reflection (Bauschke, Combettes & Luke, JOSA A
  20, 1025, 2003);
* **OSS**, oversampling smoothness (Rodriguez, Xu, Chen, Zou & Miao, J.
  Appl. Cryst. 46, 312, 2013): HIO whose out-of-support region is low-pass
  filtered by a Gaussian of decreasing width;
* **shrinkwrap** support refinement (Marchesini et al., Phys. Rev. B 68,
  140101(R), 2003), usable with any of the above.

Algorithms are chained with a schedule (``"hio:400,er:100"``), and many
random starts run simultaneously as a leading batch dimension, so each
iteration is one batched FFT pair. The far-field arrays are centred (DC at
``n // 2``); they are shifted once on entry and once on exit, never inside
the loop. Sub-pixel registration in :func:`align_object` follows
Guizar-Sicairos, Thurman & Fienup, Opt. Lett. 33, 156 (2008).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np

from ..backend import Backend, BackendLike, backend_of, get_backend, to_numpy

__all__ = [
    "CDI_ALGORITHMS",
    "DEFAULT_BETA",
    "CDIResult",
    "ShrinkwrapConfig",
    "SimulatedCDI",
    "align_object",
    "autocorrelation_support",
    "cdi",
    "parse_schedule",
    "simulate_cdi",
]

CDI_ALGORITHMS = ("er", "hio", "dm", "raar", "rrr", "asr", "hpr", "oss")
"""Names accepted in a :func:`cdi` schedule."""

_ALIASES = {
    "douglas-rachford": "asr",
    "dr": "asr",
    "difference-map": "dm",
    "difference_map": "dm",
    "error-reduction": "er",
}
_CONSTRAINTS = ("complex", "real", "positive")
DEFAULT_BETA = {
    "er": 1.0,
    "hio": 0.9,
    "dm": 0.9,
    "raar": 0.98,
    "rrr": 0.5,
    "asr": 1.0,
    "hpr": 0.9,
    "oss": 0.9,
}
"""Per-algorithm ``beta`` used when neither the stage nor :func:`cdi` sets one.

ER and ASR have no parameter (their entry is informational)."""

ScheduleLike = str | Sequence[Any]


# ----------------------------------------------------------------------------- configuration
@dataclass(frozen=True)
class ShrinkwrapConfig:
    """Shrinkwrap support refinement (Marchesini et al. 2003).

    Every :attr:`every` iterations the support becomes
    ``G_sigma * |object| > threshold * max(G_sigma * |object|)``, where
    ``G_sigma`` is a Gaussian of standard deviation ``sigma`` pixels and
    ``|object|`` is the modulus of the current data-consistent image
    ``P_M x``. ``sigma`` starts at :attr:`sigma` and is multiplied by
    :attr:`decay` after each update until it reaches :attr:`sigma_min`.

    Attributes
    ----------
    every:
        Iterations between support updates.
    threshold:
        Fraction of the maximum of the blurred modulus kept in the support.
    sigma:
        Initial Gaussian standard deviation in pixels.
    sigma_min:
        Smallest standard deviation in pixels.
    decay:
        Factor applied to ``sigma`` after every update (``<= 1``).
    start:
        No update happens before this (global) iteration.
    stop:
        No update happens after this iteration (``None``: never stop). Useful
        to let a final ER stage refine with a fixed support.
    """

    every: int = 20
    threshold: float = 0.2
    sigma: float = 3.0
    sigma_min: float = 1.5
    decay: float = 0.99
    start: int = 0
    stop: int | None = None

    def __post_init__(self) -> None:
        if self.every < 1:
            raise ValueError(f"shrinkwrap every must be >= 1, got {self.every}")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError(f"shrinkwrap threshold must be in (0, 1), got {self.threshold}")
        if self.sigma < 0 or self.sigma_min < 0 or self.sigma_min > self.sigma:
            raise ValueError("shrinkwrap needs 0 <= sigma_min <= sigma (pixels)")
        if not 0.0 < self.decay <= 1.0:
            raise ValueError(f"shrinkwrap decay must be in (0, 1], got {self.decay}")


ShrinkwrapLike = ShrinkwrapConfig | dict | bool | None


def _shrinkwrap_config(value: ShrinkwrapLike) -> ShrinkwrapConfig | None:
    if value is None or value is False:
        return None
    if value is True:
        return ShrinkwrapConfig()
    if isinstance(value, ShrinkwrapConfig):
        return value
    if isinstance(value, dict):
        return ShrinkwrapConfig(**value)
    raise TypeError(
        f"shrinkwrap must be None, True, a dict or a ShrinkwrapConfig, got {type(value).__name__}"
    )


def parse_schedule(
    schedule: ScheduleLike, beta: float | None = None
) -> list[tuple[str, int, float]]:
    """Normalize a CDI schedule to ``[(algorithm, iterations, beta), ...]``.

    Parameters
    ----------
    schedule:
        Either a string ``"hio:400,er:100"`` (an optional third field sets the
        stage's beta: ``"raar:300:0.8"``) or a sequence of ``(name, n)`` /
        ``(name, n, beta)`` tuples. Names are those in
        :data:`CDI_ALGORITHMS` (``"dr"`` is an alias of ``"asr"``).
    beta:
        Feedback/relaxation parameter of stages that do not set their own;
        None uses :data:`DEFAULT_BETA`.

    Returns
    -------
    list of tuple
        One ``(name, iterations, beta)`` per non-empty stage.
    """
    if isinstance(schedule, str):
        items: list[Sequence[Any]] = []
        for part in schedule.split(","):
            if part.strip():
                items.append([bit.strip() for bit in part.split(":")])
    elif isinstance(schedule, tuple) and schedule and isinstance(schedule[0], str):
        items = [schedule]
    else:
        items = list(schedule)
    stages: list[tuple[str, int, float]] = []
    for item in items:
        if isinstance(item, str) or len(item) not in (2, 3):
            raise ValueError(
                f"schedule stage {item!r} must be (name, iterations) or (name, iterations, beta), "
                "e.g. [('hio', 400), ('er', 100)] or 'hio:400,er:100'"
            )
        name = str(item[0]).strip().lower()
        name = _ALIASES.get(name, name)
        if name not in CDI_ALGORITHMS:
            raise ValueError(f"unknown CDI algorithm {item[0]!r}; choose from {CDI_ALGORITHMS}")
        n = int(item[1])
        if n < 0:
            raise ValueError(f"stage {name!r} has a negative iteration count {n}")
        if len(item) == 3 and item[2] is not None:
            b = float(item[2])
        else:
            b = DEFAULT_BETA[name] if beta is None else float(beta)
        if not b > 0.0:
            raise ValueError(f"stage {name!r} needs beta > 0, got {b}")
        if n:
            stages.append((name, n, b))
    if not stages:
        raise ValueError("the schedule has no iterations; e.g. use schedule='hio:400,er:100'")
    return stages


# ----------------------------------------------------------------------------- result
@dataclass
class CDIResult:
    """Outcome of :func:`cdi`.

    Array fields are backend arrays (CuPy for GPU solves) until you call
    :meth:`to_numpy`. All images are centred like the inputs.

    Attributes
    ----------
    object:
        ``(ny, nx)`` complex object estimate of the best start (the support
        projection of the final data-consistent image).
    support:
        ``(ny, nx)`` boolean support of the best start (after shrinkwrap).
    objects:
        ``(starts, ny, nx)`` final estimates of every start (for averaging or
        phase-retrieval transfer functions).
    history:
        Fourier-modulus error of the best start at each check; the last entry
        is the final :attr:`modulus_error`.
    support_history:
        Object-domain constraint violation of the best start at the same
        points.
    times:
        Seconds since the start at each :attr:`history` entry.
    n_iter:
        Iterations performed.
    converged:
        Whether the modulus error reached ``tol`` (or the callback stopped).
    message:
        Why the run stopped.
    elapsed:
        Total wall-clock seconds, including setup.
    device:
        ``"cpu"`` or ``"gpu"``.
    best_start:
        Index of the start with the lowest final modulus error.
    start_errors:
        Final modulus error of every start (host array).
    modulus_error:
        Final normalized Fourier-modulus error of :attr:`object`,
        ``sqrt(sum_meas (|F o| - A)^2 / sum_meas A^2)``.
    support_error:
        Final normalized constraint violation
        ``||u - P_S u|| / ||u||`` of the image ``u`` :attr:`object` was
        projected from.
    schedule:
        The parsed schedule ``[(name, iterations, beta), ...]``.
    """

    object: Any
    support: Any
    objects: Any
    history: list[float]
    support_history: list[float]
    times: list[float]
    n_iter: int
    converged: bool
    message: str
    elapsed: float
    device: str
    best_start: int
    start_errors: np.ndarray
    modulus_error: float
    support_error: float
    schedule: list[tuple[str, int, float]]

    def to_numpy(self) -> CDIResult:
        """Copy with every array field on the host."""
        changes = {
            f.name: to_numpy(getattr(self, f.name))
            for f in fields(self)
            if hasattr(getattr(self, f.name), "shape")
            and not isinstance(getattr(self, f.name), np.ndarray)
        }
        return replace(self, **changes)

    def __repr__(self) -> str:
        return (
            f"CDIResult(device={self.device!r}, n_iter={self.n_iter}, starts="
            f"{len(self.start_errors)}, best_start={self.best_start}, "
            f"modulus_error={self.modulus_error:.4g}, converged={self.converged})"
        )


# ----------------------------------------------------------------------------- projections
class _Projector:
    """Fourier-modulus and object-domain projections on uncentred arrays."""

    def __init__(
        self,
        be: Backend,
        amplitude: Any,
        measured: Any,
        constraint: str,
        bounds: tuple[float | None, float | None] | None,
    ) -> None:
        self.be = be
        self.xp = be.xp
        self.amplitude = amplitude
        self.measured = measured
        self.unmeasured = None if measured is None else ~measured
        self.constraint = constraint
        lo, hi = bounds if bounds is not None else (None, None)
        self.lo = None if lo is None or lo <= 0 else float(lo)
        self.hi = None if hi is None else float(hi)
        peak = float(self.xp.max(amplitude)) if amplitude.size else 0.0
        # Floor for |F x| so a zero transform gives a zero (not NaN) projection.
        self.floor = max(peak * 1e-20, 1e-30)
        weights = amplitude * amplitude
        if measured is not None:
            weights = weights * measured
        self.energy = max(float(self.xp.sum(weights)), 1e-300)

    def modulus(self, x: Any) -> Any:
        """``P_M``: impose the measured modulus on measured pixels."""
        xp = self.xp
        spec = self.be.fft2(x)
        factor = xp.abs(spec)
        xp.maximum(factor, self.floor, out=factor)
        xp.divide(self.amplitude, factor, out=factor)
        if self.unmeasured is not None:
            xp.copyto(factor, 1.0, where=self.unmeasured)
        spec *= factor
        return self.be.ifft2(spec)

    def object(self, u: Any, support: Any) -> Any:
        """``P_S``: zero outside the support, nearest feasible value inside."""
        xp = self.xp
        if self.constraint == "complex":
            v = u
        elif self.constraint == "real":
            v = u.real
        else:
            v = xp.maximum(u.real, 0.0)
        if self.lo is not None or self.hi is not None:
            if self.constraint == "positive":
                v = xp.clip(v, self.lo, self.hi)
            else:
                mag = xp.abs(v)
                clipped = xp.clip(mag, self.lo, self.hi)
                if self.constraint == "real":
                    v = xp.where(v < 0, -clipped, clipped)
                else:
                    v = v * (clipped / xp.maximum(mag, self.floor))
        return xp.where(support, v, 0.0)

    def feasible(self, u: Any, support: Any) -> Any:
        """Pixels where ``u`` already satisfies the object constraints (for HIO)."""
        ok = support
        if self.constraint == "positive":
            ok = ok & (u.real >= 0.0)
        if self.lo is not None or self.hi is not None:
            mag = self.xp.abs(u) if self.constraint == "complex" else self.xp.abs(u.real)
            if self.lo is not None:
                ok = ok & (mag >= self.lo)
            if self.hi is not None:
                ok = ok & (mag <= self.hi)
        return ok

    def errors(self, u: Any, support: Any) -> tuple[Any, Any, Any]:
        """Estimate ``P_S u`` with its per-start modulus error and violation (device arrays)."""
        xp = self.xp
        est = self.object(u, support)
        diff = xp.abs(self.be.fft2(est)) - self.amplitude
        if self.measured is not None:
            diff = diff * self.measured
        mod = xp.sqrt(xp.sum(diff * diff, axis=(-2, -1)) / self.energy)
        resid = u - est
        num = xp.sum(resid.real**2 + resid.imag**2, axis=(-2, -1))
        den = xp.sum(u.real**2 + u.imag**2, axis=(-2, -1))
        viol = xp.sqrt(num / xp.maximum(den, 1e-300))
        return est, mod, viol


def _frequency_grid(be: Backend, shape: tuple[int, int]) -> tuple[Any, Any]:
    """Uncentred integer frequency indices ``(ky[:, None], kx[None, :])`` on the backend."""
    ny, nx = shape
    ky = np.fft.fftfreq(ny) * ny
    kx = np.fft.fftfreq(nx) * nx
    return be.asarray(ky[:, None], dtype="real"), be.asarray(kx[None, :], dtype="real")


def _gaussian_blur(be: Backend, image: Any, sigma: float, k2: Any) -> Any:
    """Circular Gaussian blur (std ``sigma`` pixels) of real images over the last two axes."""
    if sigma <= 0:
        return image
    # k2 holds (ky/ny)^2 + (kx/nx)^2 in cycles per pixel squared.
    transfer = be.xp.exp((-2.0 * math.pi**2 * sigma**2) * k2)
    return be.ifft2(be.fft2(image) * transfer).real


def _shrinkwrap(
    be: Backend, image: Any, sigma: float, threshold: float, k2: Any, bounds_mask: Any
) -> Any:
    xp = be.xp
    blurred = _gaussian_blur(be, xp.abs(image), sigma, k2)
    peak = xp.max(blurred, axis=(-2, -1), keepdims=True)
    support = blurred > threshold * peak
    if bounds_mask is not None:
        support &= bounds_mask
    return support


# ----------------------------------------------------------------------------- the solver
def cdi(
    magnitudes: Any,
    support: Any,
    *,
    schedule: ScheduleLike = "hio:500,er:100",
    starts: int = 1,
    constraint: str = "complex",
    bounds: tuple[float | None, float | None] | None = None,
    shrinkwrap: ShrinkwrapLike = None,
    measured: Any = None,
    intensity: bool = False,
    beta: float | None = None,
    initial: Any = None,
    seed: Any = None,
    device: BackendLike = "cpu",
    precision: str | None = None,
    check_every: int = 10,
    tol: float = 1e-6,
    oss_stages: int = 10,
    callback: Callable[[int, Any, float], bool | None] | None = None,
) -> CDIResult:
    """Reconstruct an object from its far-field modulus by iterative projections.

    Parameters
    ----------
    magnitudes:
        ``(ny, nx)`` measured far-field modulus ``|F o|``, centred (DC at
        ``(ny // 2, nx // 2)``, i.e. ``fftshift``-ed), any linear units.
        Non-finite values are treated as unmeasured.
    support:
        ``(ny, nx)`` boolean object support, centred like the object. Use
        :func:`autocorrelation_support` with ``shrinkwrap`` when it is
        unknown.
    schedule:
        Algorithms to run in order, e.g. ``"hio:400,er:100"`` or
        ``[("raar", 300, 0.8), ("er", 50)]``; see :func:`parse_schedule`.
    starts:
        Number of independent random starts run simultaneously as a batch.
        The start with the lowest final modulus error is returned.
    constraint:
        Object-domain constraint inside the support: ``"complex"`` (support
        only), ``"real"`` or ``"positive"`` (real and non-negative).
    bounds:
        Optional ``(min, max)`` bounds on the object modulus inside the
        support (either may be None), in object units.
    shrinkwrap:
        Support refinement: None/False (fixed support), True (defaults), a
        dict of :class:`ShrinkwrapConfig` fields, or a config. Refined
        supports stay inside the initial ``support`` (unless it is all True),
        so pass a generous one, e.g. :func:`autocorrelation_support`.
    measured:
        Optional ``(ny, nx)`` boolean mask, centred, False where the modulus
        is unknown (beamstop, detector gaps); the modulus is left free there.
    intensity:
        If True, ``magnitudes`` holds intensities ``|F o|^2`` (negative values
        are clipped to zero before the square root).
    beta:
        Feedback (HIO, HPR, OSS) / relaxation (DM, RAAR, RRR) parameter of
        stages that do not set their own. None uses the per-algorithm
        :data:`DEFAULT_BETA` (0.9; RAAR 0.98; RRR 0.5). ER and ASR ignore it.
    initial:
        Optional starting object, ``(ny, nx)`` (shared by all starts) or
        ``(starts, ny, nx)``, centred. Default: random Fourier phases with the
        measured modulus, restricted to the support.
    seed:
        Seed or :class:`numpy.random.Generator` for the random starts (drawn
        on the host, so CPU and GPU runs start identically).
    device, precision:
        Backend selection, see :func:`solvephase.get_backend`.
    check_every:
        Iterations between error evaluations (each one synchronizes the GPU).
    tol:
        Stop when the best start's modulus error is at or below ``tol``.
    oss_stages:
        Number of filter widths an OSS stage steps through (its iterations
        are split evenly); at the end of each the best iterate is kept.
    callback:
        ``callback(iteration, object, error)`` at every check with the
        centred estimate of the currently best start; return True to stop.

    Returns
    -------
    CDIResult
        The best start's object and support, error histories and per-start
        final errors.

    Notes
    -----
    Each iteration of ER, HIO, RAAR, RRR, ASR and HPR costs one batched FFT
    pair; DM and OSS cost two. An error check adds one forward FFT.
    """
    t0 = time.perf_counter()
    be = get_backend(device, precision)
    xp = be.xp
    stages = parse_schedule(schedule, beta)
    constraint = str(constraint).lower()
    if constraint not in _CONSTRAINTS:
        raise ValueError(f"constraint must be one of {_CONSTRAINTS}, got {constraint!r}")
    if int(starts) < 1:
        raise ValueError(f"starts must be >= 1, got {starts}")
    if int(check_every) < 1:
        raise ValueError(f"check_every must be >= 1, got {check_every}")
    if int(oss_stages) < 1:
        raise ValueError(f"oss_stages must be >= 1, got {oss_stages}")
    starts, check_every, oss_stages = int(starts), int(check_every), int(oss_stages)
    sw = _shrinkwrap_config(shrinkwrap)

    data = np.asarray(to_numpy(magnitudes), dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"magnitudes must be a 2-D (ny, nx) array, got shape {data.shape}")
    shape = (int(data.shape[0]), int(data.shape[1]))
    finite = np.isfinite(data)
    data = np.where(finite, data, 0.0)
    if intensity:
        data = np.sqrt(np.maximum(data, 0.0))
    elif np.any(data < 0):
        raise ValueError("magnitudes must be non-negative; pass intensity=True for intensities")
    meas_host = finite.copy()
    if measured is not None:
        m = np.asarray(to_numpy(measured), dtype=bool)
        if m.shape != shape:
            raise ValueError(f"measured has shape {m.shape}, expected {shape}")
        meas_host &= m
    sup_host = np.asarray(to_numpy(support), dtype=bool)
    if sup_host.shape != shape:
        raise ValueError(f"support has shape {sup_host.shape}, expected {shape}")
    if not sup_host.any():
        raise ValueError("the support is empty; pass a boolean mask that covers the object")

    # One shift into uncentred (FFT) order; everything below is uncentred.
    amp = be.asarray(np.fft.ifftshift(data), dtype="real")
    meas = None if meas_host.all() else be.asarray(np.fft.ifftshift(meas_host))
    if meas is not None:
        amp = amp * meas
    outer = be.asarray(np.fft.ifftshift(sup_host))
    sup = outer
    proj = _Projector(be, amp, meas, constraint, bounds)

    rng = be.random(seed)
    if initial is None:
        phases = be.asarray(rng.uniform(0.0, 2.0 * math.pi, size=(starts, *shape)), dtype="real")
        x = be.ifft2(amp * xp.exp(1j * phases).astype(be.complex_dtype, copy=False))
        x = xp.where(sup, x, 0.0)
    else:
        init = np.asarray(to_numpy(initial))
        if init.shape == shape:
            init = np.broadcast_to(init, (starts, *shape))
        if init.shape != (starts, *shape):
            raise ValueError(f"initial must have shape {shape} or {(starts, *shape)}")
        x = be.asarray(np.fft.ifftshift(init, axes=(-2, -1)), dtype="complex").copy()
    x = x.astype(be.complex_dtype, copy=False)

    ky, kx = _frequency_grid(be, shape)
    k2 = (ky / shape[0]) ** 2 + (kx / shape[1]) ** 2
    sw_sigma = sw.sigma if sw is not None else 0.0
    sw_bound = None if (sw is None or sup_host.all()) else outer

    history: list[float] = []
    viol_history: list[float] = []
    times: list[float] = []
    message, converged = "iteration limit reached", False
    it = 0
    stop = False

    def report(u: Any) -> tuple[np.ndarray, np.ndarray, Any]:
        est, mod, viol = proj.errors(u, sup)
        mod_h = np.asarray(be.to_numpy(mod), dtype=np.float64)
        viol_h = np.asarray(be.to_numpy(viol), dtype=np.float64)
        best = int(np.argmin(mod_h))
        history.append(float(mod_h[best]))
        viol_history.append(float(viol_h[best]))
        times.append(time.perf_counter() - t0)
        return mod_h, viol_h, est

    for name, n_stage, b in stages:
        if stop:
            break
        # OSS: the filter width alpha (frequency pixels) steps linearly from 2N to N/5, as
        # in the reference implementation of Rodriguez et al. (2013).
        segment_ends: set[int] = set()
        if name == "oss":
            n_big = float(max(shape))
            alphas = np.linspace(2.0 * n_big, 0.2 * n_big, oss_stages)
            edges = np.linspace(0, n_stage, oss_stages + 1).round().astype(int)
            segment_ends = {int(e) for e in edges[1:]}
            kk = ky**2 + kx**2
            best_x, top_x = x.copy(), x.copy()
            best_err: np.ndarray = np.full(starts, np.inf)
            top_err: np.ndarray = np.full(starts, np.inf)
            segment = 0
            window = xp.exp(-0.5 * kk / float(alphas[0]) ** 2)
        for local in range(1, n_stage + 1):
            it += 1
            check = it % check_every == 0 or local in segment_ends
            x_old = x
            y = proj.modulus(x)
            u = y
            if name == "er":
                x = proj.object(y, sup)
            elif name in ("hio", "oss"):
                x = xp.where(proj.feasible(y, sup), proj.object(y, sup), x - b * y)
                if name == "oss":
                    smooth = be.ifft2(be.fft2(x) * window)
                    x = xp.where(sup, x, smooth)
            elif name == "hpr":
                x = x - b * y + proj.object((1.0 + b) * y - x, sup)
            elif name == "asr":
                x = x - y + proj.object(2.0 * y - x, sup)
            elif name == "rrr":
                x = x + b * (proj.object(2.0 * y - x, sup) - y)
            elif name == "raar":
                x = b * x + (1.0 - 2.0 * b) * y + b * proj.object(2.0 * y - x, sup)
            else:  # dm, Elser 2003 with gamma_S = -1/beta, gamma_M = 1/beta
                g_s, g_m = -1.0 / b, 1.0 / b
                u = (1.0 + g_m) * y - g_m * x
                f_s = (1.0 + g_s) * proj.object(x, sup) - g_s * x
                x = x + b * (proj.object(u, sup) - proj.modulus(f_s))
            x = x.astype(be.complex_dtype, copy=False)

            if check:
                mod_h, _, est = report(u)
                if name == "oss":
                    # Keep each start's best iterate of the segment; restart the next
                    # segment from it, and end the stage on the best of all segments.
                    better = mod_h < best_err
                    if better.any():
                        best_x = xp.where(be.asarray(better)[:, None, None], x_old, best_x)
                        best_err = np.where(better, mod_h, best_err)
                    if local in segment_ends:
                        top = best_err < top_err
                        top_x = xp.where(be.asarray(top)[:, None, None], best_x, top_x)
                        top_err = np.where(top, best_err, top_err)
                        x = top_x.copy() if local == n_stage else best_x.copy()
                        best_err = np.full(starts, np.inf)
                        segment += 1
                        if segment < len(alphas):
                            window = xp.exp(-0.5 * kk / float(alphas[segment]) ** 2)
                if callback is not None:
                    best = int(np.argmin(mod_h))
                    centred = xp.fft.fftshift(est[best], axes=(-2, -1))
                    if callback(it, centred, float(mod_h[best])):
                        message, converged, stop = "stopped by callback", True, True
                if not stop and float(mod_h.min()) <= tol:
                    message, converged, stop = "modulus error reached tol", True, True
            if (
                sw is not None
                and it % sw.every == 0
                and it >= sw.start
                and (sw.stop is None or it <= sw.stop)
            ):
                sup = _shrinkwrap(be, y, sw_sigma, sw.threshold, k2, sw_bound)
                sw_sigma = max(sw.sigma_min, sw_sigma * sw.decay)
            if stop:
                break

    # Final estimate: the support projection of the data-consistent image of x
    # (for DM, of f_M(x)); it is also the last history entry.
    y = proj.modulus(x)
    if name == "dm":
        y = (1.0 + 1.0 / b) * y - (1.0 / b) * x
    mod_h, viol_h, est = report(y)
    best = int(np.argmin(mod_h))
    objects = xp.fft.fftshift(est.astype(be.complex_dtype, copy=False), axes=(-2, -1))
    sup_b = sup if sup.ndim == 2 else sup[best]
    be.synchronize()
    return CDIResult(
        object=objects[best],
        support=xp.fft.fftshift(sup_b, axes=(-2, -1)),
        objects=objects,
        history=history,
        support_history=viol_history,
        times=times,
        n_iter=it,
        converged=converged,
        message=message,
        elapsed=time.perf_counter() - t0,
        device=be.device,
        best_start=best,
        start_errors=mod_h,
        modulus_error=float(mod_h[best]),
        support_error=float(viol_h[best]),
        schedule=stages,
    )


# ----------------------------------------------------------------------------- helpers
def autocorrelation_support(
    intensity: Any,
    threshold: float = 0.04,
    *,
    measured: Any = None,
    device: BackendLike = None,
) -> Any:
    """Loose initial support from the thresholded autocorrelation.

    The inverse Fourier transform of the far-field intensity is the object's
    autocorrelation, whose support is twice the object's extent (the
    difference set ``S - S``); thresholded, it is the usual starting support
    for shrinkwrap.

    Parameters
    ----------
    intensity:
        ``(ny, nx)`` centred far-field intensity ``|F o|^2`` (square
        magnitudes first). Non-finite values count as unmeasured.
    threshold:
        Fraction of the autocorrelation maximum kept.
    measured:
        Optional centred boolean mask, False where the intensity is unknown
        (set to zero before transforming).
    device:
        Backend; by default the backend of ``intensity``.

    Returns
    -------
    array
        ``(ny, nx)`` centred boolean support on the backend.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")
    be = backend_of(intensity) if device is None else get_backend(device)
    xp = be.xp
    data = be.asarray(intensity, dtype="real")
    if data.ndim != 2:
        raise ValueError(f"intensity must be 2-D, got shape {tuple(data.shape)}")
    data = xp.where(xp.isfinite(data), data, 0.0)
    if measured is not None:
        data = data * be.asarray(measured, dtype=bool)
    auto = xp.abs(xp.fft.fftshift(be.ifft2(xp.fft.ifftshift(data))))
    return auto > threshold * xp.max(auto)


def _upsampled_correlation(
    xp: Any, product: Any, centre: tuple[float, float], upsample: int, half: int
) -> Any:
    """Cross-correlation sampled on ``centre + m / upsample``, ``|m| <= half``.

    ``product`` is ``F(ref) * conj(F(est))`` (uncentred); the correlation at
    shift ``s`` is ``sum_k product(k) exp(+2 pi i k.s / n)``
    (Guizar-Sicairos et al. 2008, matrix-multiply DFT).
    """
    ny, nx = product.shape
    offsets = xp.arange(-half, half + 1) / float(upsample)
    ky = xp.asarray(np.fft.fftfreq(ny) * ny)
    kx = xp.asarray(np.fft.fftfreq(nx) * nx)
    ey = xp.exp((2j * math.pi / ny) * xp.outer(centre[0] + offsets, ky))
    ex = xp.exp((2j * math.pi / nx) * xp.outer(kx, centre[1] + offsets))
    return ey @ product @ ex


def align_object(
    estimate: Any,
    reference: Any,
    *,
    subpixel: bool = True,
    upsample: int = 100,
    twin: bool = True,
    scale: bool = True,
) -> tuple[Any, float]:
    """Register a CDI reconstruction to a reference, removing the trivial ambiguities.

    A far-field modulus does not change under a global phase factor, a
    translation, or the twin image ``conj(o(-r))``. This finds the
    translation (integer cross-correlation peak refined to ``1/upsample``
    pixel by an upsampled matrix DFT, Guizar-Sicairos et al. 2008), the twin
    choice and the complex factor ``c`` that minimize
    ``||c * shift(est) - ref|| / ||ref||``.

    Parameters
    ----------
    estimate, reference:
        ``(ny, nx)`` complex or real images of the same shape.
    subpixel:
        Refine the translation below one pixel (applied as a Fourier phase
        ramp, i.e. a circular shift).
    upsample:
        Sub-pixel refinement factor.
    twin:
        Also try the twin ``conj(est[::-1, ::-1])`` and keep the better one.
    scale:
        If True ``c`` is any complex number (phase and scale); if False only
        a global phase ``|c| = 1`` is removed.

    Returns
    -------
    aligned:
        ``c * shift(est)`` (or its twin), on the backend of ``estimate``.
    error:
        ``||aligned - ref|| / ||ref||``.
    """
    be = backend_of(estimate)
    if be.precision == "single":
        be = get_backend(be.device, "double")
    xp = be.xp
    est = be.asarray(estimate, dtype="complex")
    ref = be.asarray(reference, dtype="complex")
    if est.ndim != 2 or est.shape != ref.shape:
        raise ValueError(
            f"estimate and reference must be 2-D with the same shape, got "
            f"{tuple(est.shape)} and {tuple(ref.shape)}"
        )
    ny, nx = est.shape
    f_ref = be.fft2(ref)
    ref_norm = math.sqrt(be.dot(ref, ref))
    if ref_norm == 0:
        raise ValueError("the reference is zero")
    ky, kx = _frequency_grid(be, (ny, nx))
    candidates = [est]
    if twin:
        candidates.append(xp.conj(est[::-1, ::-1]))
    best: tuple[Any, float] | None = None
    for cand in candidates:
        f_c = be.fft2(cand)
        product = f_ref * xp.conj(f_c)
        corr = xp.abs(be.ifft2(product))
        iy, ix = (int(v) for v in np.unravel_index(int(xp.argmax(corr)), corr.shape))
        sy = float(iy - ny if iy > ny // 2 else iy)
        sx = float(ix - nx if ix > nx // 2 else ix)
        if subpixel and upsample > 1:
            half = math.ceil(0.75 * upsample)
            local = xp.abs(_upsampled_correlation(xp, product, (sy, sx), int(upsample), half))
            my, mx = np.unravel_index(int(xp.argmax(local)), local.shape)
            sy += (int(my) - half) / float(upsample)
            sx += (int(mx) - half) / float(upsample)
        ramp = xp.exp((-2j * math.pi) * (ky * (sy / ny) + kx * (sx / nx)))
        shifted = be.ifft2(f_c * ramp)
        if sy == round(sy) and sx == round(sx):
            shifted = xp.roll(cand, (round(sy), round(sx)), axis=(0, 1))
        energy = be.dot(shifted, shifted)
        if energy == 0:
            c: complex = 0.0
        else:
            c = complex(xp.sum(xp.conj(shifted) * ref)) / energy
            if not scale:
                c = c / abs(c) if c != 0 else 1.0
        aligned = c * shifted
        resid = aligned - ref
        err = math.sqrt(be.dot(resid, resid)) / ref_norm
        if best is None or err < best[1]:
            best = (aligned, err)
    assert best is not None
    return best


@dataclass
class SimulatedCDI:
    """Synthetic CDI data from :func:`simulate_cdi` (centred backend arrays).

    Attributes
    ----------
    magnitudes:
        ``(ny, nx)`` far-field modulus: ``|F o|`` without noise, or the square
        root of the Poisson counts.
    intensity:
        ``(ny, nx)`` far-field intensity: ``|F o|^2`` or the Poisson counts.
    object:
        ``(ny, nx)`` zero-padded true object, scaled so that ``|F o|^2`` is the
        expected intensity.
    support:
        ``(ny, nx)`` tight support ``|o| > 0`` of the padded object.
    """

    magnitudes: Any
    intensity: Any
    object: Any
    support: Any


def simulate_cdi(
    obj: Any,
    oversampling: float = 2.0,
    *,
    shape: tuple[int, int] | None = None,
    photons: float | None = None,
    seed: Any = None,
    device: BackendLike = "cpu",
    precision: str | None = None,
) -> SimulatedCDI:
    """Zero-pad an object and compute its centred far-field diffraction data.

    Parameters
    ----------
    obj:
        ``(my, mx)`` complex or real object.
    oversampling:
        Linear oversampling per axis: the window is ``round(oversampling * m)``
        pixels along each axis. Unique recovery needs Miao's oversampling
        ratio (window area / object area) above 2, i.e. ``oversampling``
        above ``sqrt(2)`` for an object filling its box.
    shape:
        Explicit ``(ny, nx)`` window instead of ``oversampling``.
    photons:
        If given, the expected total photon count: the intensity is scaled
        to it and Poisson noise is drawn (on the host).
    seed:
        Seed or :class:`numpy.random.Generator` for the noise.
    device, precision:
        Backend of the returned arrays.

    Returns
    -------
    SimulatedCDI
    """
    be = get_backend(device, precision)
    xp = be.xp
    o = np.asarray(to_numpy(obj))
    if o.ndim != 2:
        raise ValueError(f"obj must be 2-D, got shape {o.shape}")
    my, mx = o.shape
    if shape is None:
        if oversampling < 1:
            raise ValueError(f"oversampling must be >= 1, got {oversampling}")
        shape = (round(oversampling * my), round(oversampling * mx))
    ny, nx = int(shape[0]), int(shape[1])
    if ny < my or nx < mx:
        raise ValueError(f"window {shape} is smaller than the object {o.shape}")
    padded = np.zeros((ny, nx), dtype=np.complex128)
    y0, x0 = (ny - my) // 2, (nx - mx) // 2
    padded[y0 : y0 + my, x0 : x0 + mx] = o
    field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded), norm="ortho"))
    inten = np.abs(field) ** 2
    if photons is not None:
        if photons <= 0:
            raise ValueError(f"photons must be positive, got {photons}")
        factor = float(photons) / float(inten.sum())
        padded *= math.sqrt(factor)
        expected = inten * factor
        inten = be.random(seed).poisson(expected).astype(np.float64)
        mags = np.sqrt(inten)
    else:
        mags = np.abs(field)
    obj_out = padded if np.iscomplexobj(o) else padded.real
    return SimulatedCDI(
        magnitudes=be.asarray(mags, dtype="real"),
        intensity=be.asarray(inten, dtype="real"),
        object=be.asarray(obj_out),
        support=xp.asarray(np.abs(padded) > 0),
    )
