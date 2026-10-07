"""Transport-of-intensity (TIE) phase retrieval from a defocus stack.

In the paraxial regime the axial change of intensity of a monochromatic beam
is fixed by its transverse phase (Teague, JOSA 73, 1434, 1983)::

    -k dI/dz = div(I grad(phi)),    k = 2 pi / lambda.

Given intensities ``I(z_j)`` near the plane ``z = 0`` the solver estimates the
axial derivative ``dI/dz`` and the in-focus intensity ``I0``, then inverts the
elliptic equation for ``phi`` (radians at ``z = 0``):

* **Axial derivative.** ``"central"`` uses the nearest planes on each side of
  ``z = 0`` (and ``z = 0`` itself when measured): a central difference for a
  symmetric pair, the three-point Lagrange derivative otherwise.
  ``"polyfit"`` fits a polynomial of degree ``order`` in ``z`` to every plane
  by least squares, pixel by pixel (Savitzky-Golay style), and takes its
  slope at ``z = 0``; with more planes than coefficients it averages noise,
  with ``order = N - 1`` it is the exact ``N``-point stencil whose truncation
  error falls as a higher power of the plane spacing (Waller et al., Opt.
  Express 18, 12552, 2010). The measured ``z = 0`` plane is the in-focus
  intensity when present; otherwise the interpolant (or fit) value at
  ``z = 0`` is used.
* **Spectral solvers.** ``"fft"`` assumes a periodic field (real FFTs);
  ``"dct"`` assumes zero normal slope at the window edge and expands the field
  in the cosine series of its mirror extension (type-II DCT; gradients are
  type-II DST series), which avoids the wrap-around coupling of opposite edges
  for non-periodic fields (Zuo, Chen & Asundi, Opt. Express 22, 9220, 2014).
  Both differentiate exactly and use either the uniform-intensity
  approximation ``laplacian(phi) = -k dI/dz / mean(I0)`` (``uniform=True``) or
  Teague's auxiliary function ``psi`` with ``grad(psi) = I grad(phi)``: two
  Poisson solves and a division by the clipped intensity (Gureyev & Nugent,
  JOSA A 13, 1670, 1996; Paganin & Nugent, PRL 80, 2586, 1998).
* **Masked exact solver.** Teague's construction assumes ``I grad(phi)`` is
  curl free. ``"pcg"`` drops that assumption and solves the five-point
  discretization of ``div(I grad(phi)) = -k dI/dz`` inside a support ``mask``
  (default: where ``I0`` exceeds ``intensity_floor`` of its maximum) by
  conjugate gradients preconditioned with the DCT Poisson solver. Edges that
  leave the mask carry no flux, so it is the right solver for an illuminated
  aperture with dark surroundings, where the edge signal carries the
  boundary slopes (curvature sensing, Roddier, Appl. Opt. 27, 1223, 1988).
* **Regularization.** The first Poisson solve of ``"fft"``/``"dct"`` uses the
  Tikhonov filter ``lam / (lam**2 + alpha)`` on the eigenvalues ``lam`` of
  ``-laplacian`` with ``alpha = regularization * lam_min**2``: the lowest
  non-zero spatial frequency of the grid is attenuated by exactly
  ``1 / (1 + regularization)``. Raise it to suppress the low-frequency
  "cloud" artefacts that noise produces (Zuo et al., Opt. Lasers Eng. 135,
  106187, 2020). Teague's second solve is bounded and left unregularized.

Piston is not measurable and is removed (mean over the mask, or over each
connected region of the mask for ``"pcg"``).
"""

from __future__ import annotations

import functools
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, fields, replace
from typing import Any, Literal

import numpy as np

from .. import _kernels
from ..backend import Backend, BackendLike, _cpu_workers, backend_of, get_backend, to_numpy
from ..propagation import AngularSpectrumPropagator

__all__ = ["TIEResult", "simulate_defocus_stack", "tie"]

TIEMethod = Literal["dct", "fft", "pcg"]
TIEDerivative = Literal["central", "polyfit"]


@dataclass
class TIEResult:
    """Outcome of a transport-of-intensity retrieval.

    Array fields are backend arrays (CuPy for GPU solves) until you call
    :meth:`to_numpy`.

    Attributes
    ----------
    phase:
        ``(ny, nx)`` phase at ``z = 0`` in radians, piston removed, zero
        outside ``mask``.
    opd:
        The same wavefront as optical path difference in metres
        (``phase * wavelength / (2 pi)``).
    intensity:
        ``(ny, nx)`` in-focus intensity ``I0`` used by the solver (measured or
        estimated), in data units.
    didz:
        ``(ny, nx)`` axial intensity derivative estimate, data units per metre.
    mask:
        Boolean support the phase is defined on.
    wavelength:
        Wavelength in metres.
    method:
        Solver: ``"fft"``, ``"dct"`` or ``"pcg"``.
    derivative:
        Axial derivative estimator: ``"central"`` or ``"polyfit"``.
    uniform:
        Whether the uniform-intensity approximation was used.
    history:
        Relative residual norm per convergence check (``"pcg"`` only).
    n_iter:
        Conjugate-gradient iterations (0 for the direct solvers).
    converged:
        Whether the solve finished (always true for direct solvers).
    message:
        Why it stopped.
    elapsed:
        Wall-clock seconds, including the derivative estimate.
    device:
        ``"cpu"`` or ``"gpu"``.
    """

    phase: Any
    opd: Any
    intensity: Any
    didz: Any
    mask: Any
    wavelength: float
    method: str
    derivative: str
    uniform: bool
    history: list[float] = field(default_factory=list)
    n_iter: int = 0
    converged: bool = True
    message: str = ""
    elapsed: float = 0.0
    device: str = "cpu"

    def to_numpy(self) -> TIEResult:
        """Copy with every array field on the host."""
        changes: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if hasattr(value, "shape") and not isinstance(value, np.ndarray):
                changes[f.name] = to_numpy(value)
        return replace(self, **changes)


# ----------------------------------------------------------- forward model
def simulate_defocus_stack(
    field: Any,
    distances: float | Sequence[float],
    pitch: float,
    wavelength: float,
    *,
    paraxial: bool = False,
    device: BackendLike = None,
    precision: str | None = None,
) -> Any:
    """Intensities of a complex field propagated to several distances.

    Uses :class:`~solvephase.propagation.AngularSpectrumPropagator` (periodic
    boundaries: pad the field if light must not wrap around).

    Parameters
    ----------
    field:
        ``(ny, nx)`` complex field at ``z = 0`` (amplitude, not intensity).
    distances:
        Propagation distances in metres (positive downstream); ``0`` returns
        ``|field|**2``.
    pitch:
        Sample pitch in metres.
    wavelength:
        Wavelength in metres.
    paraxial:
        Use the Fresnel transfer function instead of the exact angular
        spectrum.
    device, precision:
        Backend; defaults to the backend of ``field``.

    Returns
    -------
    ``(Z, ny, nx)`` intensity stack on the backend (``(ny, nx)`` for a scalar
    distance).
    """
    be = _resolve_backend(field, device, precision)
    u = be.asarray(field, dtype="complex")
    if u.ndim != 2:
        raise ValueError(f"field must be (ny, nx), got shape {tuple(u.shape)}")
    prop = AngularSpectrumPropagator(
        u.shape, pitch, wavelength, distances, paraxial=paraxial, backend=be
    )
    out = prop.forward(u)
    return (out.real**2 + out.imag**2).astype(be.real_dtype, copy=False)


# ------------------------------------------------------- axial derivative
def _poly_weights(z: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares polynomial weights for the value and slope at ``z = 0``."""
    if order < 1:
        raise ValueError(f"order must be at least 1, got {order}")
    if z.size < order + 1:
        raise ValueError(
            f"a degree-{order} fit needs at least {order + 1} planes, got {z.size}; "
            "lower order or add planes"
        )
    scale = float(np.max(np.abs(z)))
    vander = (z[:, None] / scale) ** np.arange(order + 1)[None, :]
    if np.linalg.matrix_rank(vander) < order + 1:
        raise ValueError(f"the plane distances {z.tolist()} cannot constrain a degree-{order} fit")
    pinv = np.linalg.pinv(vander)
    return pinv[0], pinv[1] / scale


def _axial_weights(
    z: np.ndarray, derivative: str, order: int | None
) -> tuple[np.ndarray, np.ndarray, int | None]:
    """Plane weights for ``dI/dz`` and ``I0`` at ``z = 0``, and the ``z = 0`` index."""
    n = z.size
    tiny = 1e-9 * float(np.max(np.abs(z)))
    zero = np.flatnonzero(np.abs(z) <= tiny)
    focus = int(zero[0]) if zero.size else None
    w_value = np.zeros(n)
    w_slope = np.zeros(n)
    if derivative == "central":
        if order is not None:
            raise ValueError("order applies to derivative='polyfit' only")
        below = np.flatnonzero(z < -tiny)
        above = np.flatnonzero(z > tiny)
        if below.size == 0 or above.size == 0:
            raise ValueError(
                "derivative='central' needs a plane on each side of z = 0; "
                "use derivative='polyfit' for a one-sided stack"
            )
        idx = [int(below[np.argmax(z[below])]), int(above[np.argmin(z[above])])]
        if focus is not None:
            idx.insert(1, focus)
        sel = np.array(idx)
        value, slope = _poly_weights(z[sel], sel.size - 1)
        w_value[sel] = value
        w_slope[sel] = slope
    elif derivative == "polyfit":
        if order is None:
            order = min(n - 1, 3)
        w_value, w_slope = _poly_weights(z, int(order))
    else:
        raise ValueError(f"derivative must be 'central' or 'polyfit', got {derivative!r}")
    return w_slope, w_value, focus


# ----------------------------------------------------------- Poisson kernels
def _inverse_filter(be: Backend, lam: np.ndarray, alpha: float) -> Any:
    """Tikhonov-regularized inverse ``lam / (lam**2 + alpha)``, zero mode removed."""
    safe = np.where(lam == 0, 1.0, lam)
    inv = np.where(lam == 0, 0.0, safe / (safe**2 + alpha))
    return be.xp.asarray(inv, dtype=be.real_dtype)


def _lam_min(lams: Sequence[np.ndarray]) -> float:
    """Smallest non-zero eigenvalue over the per-axis eigenvalue sets."""
    return min(float(lam[1]) for lam in lams if lam.size > 1)


@functools.lru_cache(maxsize=4)
def _fft_plan(
    be: Backend, shape: tuple[int, int], pitch: tuple[float, float], regularization: float
) -> tuple[Any, Any, Any, Any]:
    """Wavenumbers ``(qy, qx)`` on the ``rfft2`` grid; regularized and exact inverse ``-lap``."""
    ny, nx = shape
    qy = 2.0 * math.pi * np.fft.fftfreq(ny, d=pitch[0])
    qx = 2.0 * math.pi * np.fft.rfftfreq(nx, d=pitch[1])
    lam = qy[:, None] ** 2 + qx[None, :] ** 2
    alpha = regularization * _lam_min([qy**2, qx**2]) ** 2
    # The Nyquist derivative of a real signal is undefined; drop it.
    if ny % 2 == 0:
        qy[ny // 2] = 0.0
    if nx % 2 == 0:
        qx[-1] = 0.0
    xp, dt = be.xp, be.real_dtype
    qy_d, qx_d = xp.asarray(qy[:, None], dtype=dt), xp.asarray(qx[None, :], dtype=dt)
    return qy_d, qx_d, _inverse_filter(be, lam, alpha), _inverse_filter(be, lam, 0.0)


@functools.lru_cache(maxsize=4)
def _dct_plan(
    be: Backend, shape: tuple[int, int], pitch: tuple[float, float], regularization: float
) -> tuple[Any, Any, Any, Any]:
    """Cosine-series wavenumbers ``pi p / (n pitch)``; regularized and exact inverse ``-lap``."""
    wy = np.pi * np.arange(shape[0]) / (shape[0] * pitch[0])
    wx = np.pi * np.arange(shape[1]) / (shape[1] * pitch[1])
    alpha = regularization * _lam_min([wy**2, wx**2]) ** 2
    lam = wy[:, None] ** 2 + wx[None, :] ** 2
    xp, dt = be.xp, be.real_dtype
    wy_d, wx_d = xp.asarray(wy[:, None], dtype=dt), xp.asarray(wx[None, :], dtype=dt)
    return wy_d, wx_d, _inverse_filter(be, lam, alpha), _inverse_filter(be, lam, 0.0)


@functools.lru_cache(maxsize=4)
def _fd_inverse(be: Backend, shape: tuple[int, int], pitch: tuple[float, float]) -> Any:
    """Inverse eigenvalues of the Neumann five-point ``-laplacian`` (DCT-II basis)."""
    ey = (2.0 - 2.0 * np.cos(np.pi * np.arange(shape[0]) / shape[0])) / pitch[0] ** 2
    ex = (2.0 - 2.0 * np.cos(np.pi * np.arange(shape[1]) / shape[1])) / pitch[1] ** 2
    return _inverse_filter(be, ey[:, None] + ex[None, :], 0.0)


# ------------------------------------------------------------------ solvers
def _solve_fft(
    be: Backend,
    b: Any,
    intensity: Any,
    support: Any,
    pitch: tuple[float, float],
    regularization: float,
) -> Any:
    """Spectral periodic solve of ``-div(I grad phi) = b`` (Teague) or ``-lap phi = b``.

    ``intensity`` is the clipped in-focus intensity, or None for ``-lap phi = b``.
    """
    xp = be.xp
    shape = b.shape
    qy, qx, inv, inv_exact = _fft_plan(be, shape, pitch, regularization)
    psi_hat = be.rfft2(b) * inv
    if intensity is None:
        return be.irfft2(psi_hat, shape)
    inv_i = xp.where(support, 1.0 / intensity, 0.0)
    gy = be.irfft2(1j * qy * psi_hat, shape) * inv_i
    gx = be.irfft2(1j * qx * psi_hat, shape) * inv_i
    neg_div = -1j * (qy * be.rfft2(gy) + qx * be.rfft2(gx))
    # The second inversion is bounded (order zero overall): no extra regularization.
    return be.irfft2(neg_div * inv_exact, shape)


def _trig(be: Backend, kind: str, a: Any, axis: int | None, inverse: bool = False) -> Any:
    """Orthonormal type-II DCT (``kind="c"``) or DST (``"s"``) along one axis (None: both)."""
    name = ("i" if inverse else "") + ("dct" if kind == "c" else "dst")
    if be.is_gpu:
        from cupyx.scipy import fft as cfft

        if axis is None:
            return getattr(cfft, name + "n")(a, type=2, norm="ortho")
        return getattr(cfft, name)(a, type=2, norm="ortho", axis=axis)
    from scipy import fft

    # Threads only pay off on large grids; on small ones they stall on a busy machine.
    workers = _cpu_workers(a.size)
    if axis is None:
        return getattr(fft, name + "n")(a, type=2, norm="ortho", workers=workers)
    return getattr(fft, name)(a, type=2, norm="ortho", axis=axis, workers=workers)


def _solve_dct(
    be: Backend,
    b: Any,
    intensity: Any,
    support: Any,
    pitch: tuple[float, float],
    regularization: float,
) -> Any:
    """Spectral Neumann solve of ``-div(I grad phi) = b`` (Teague) or ``-lap phi = b``.

    The field is expanded in the cosine series of its even extension (type-II
    DCT), whose derivatives are sine series (type-II DST): differentiation is
    exact for that extension and every gradient vanishes on the boundary.
    """
    xp = be.xp
    wy, wx, inv, inv_exact = _dct_plan(be, b.shape, pitch, regularization)
    psi_hat = _trig(be, "c", b, None) * inv
    if intensity is None:
        return _trig(be, "c", psi_hat, None, True)
    inv_i = xp.where(support, 1.0 / intensity, 0.0)
    # grad: cosine coefficient p -> sine coefficient p - 1, times -w_p.
    sy = xp.zeros_like(psi_hat)
    sy[:-1, :] = -wy[1:] * psi_hat[1:, :]
    sx = xp.zeros_like(psi_hat)
    sx[:, :-1] = -wx[:, 1:] * psi_hat[:, 1:]
    gy = _trig(be, "c", _trig(be, "s", sy, 0, True), 1, True) * inv_i
    gx = _trig(be, "s", _trig(be, "c", sx, 0, True), 1, True) * inv_i
    # div: sine coefficient p - 1 -> cosine coefficient p, times +w_p.
    hy = _trig(be, "c", _trig(be, "s", gy, 0), 1)
    hx = _trig(be, "s", _trig(be, "c", gx, 0), 1)
    div = xp.zeros_like(psi_hat)
    div[1:, :] += wy[1:] * hy[:-1, :]
    div[:, 1:] += wx[:, 1:] * hx[:, :-1]
    return _trig(be, "c", -div * inv_exact, None, True)


def _fd_operator(be: Backend, wy: Any, wx: Any) -> Any:
    """``phi -> D^T diag(w) D phi`` for forward differences ``D`` and edge weights ``w``.

    ``wy``/``wx`` already include the ``1 / pitch**2`` factors.
    """

    def apply(phi: Any) -> Any:
        fy = phi[1:, :] - phi[:-1, :]
        fy *= wy
        fx = phi[:, 1:] - phi[:, :-1]
        fx *= wx
        out = be.zeros(phi.shape)
        out[:-1, :] -= fy
        out[1:, :] += fy
        out[:, :-1] -= fx
        out[:, 1:] += fx
        return out

    return apply


def _solve_pcg(
    be: Backend,
    b: Any,
    intensity: Any,
    mask: Any,
    pitch: tuple[float, float],
    tol: float,
    max_iter: int,
    check_every: int,
    project: Any,
) -> tuple[Any, list[float], int, bool, str]:
    """Masked conjugate gradients for ``D^T (I D phi) = b``, DCT-preconditioned.

    ``D`` is the forward difference; an edge between two support pixels
    carries the mean of their intensities, any other edge carries nothing.
    ``project`` removes the per-region mean (the operator's null space).
    """
    xp = be.xp
    mi = xp.where(mask, intensity, 0.0).astype(be.real_dtype)
    wy = (0.5 / pitch[0] ** 2) * (mi[1:, :] + mi[:-1, :]) * (mask[1:, :] & mask[:-1, :])
    wx = (0.5 / pitch[1] ** 2) * (mi[:, 1:] + mi[:, :-1]) * (mask[:, 1:] & mask[:, :-1])
    i_mean = float(xp.sum(mi)) / max(float(xp.sum(mask)), 1.0)
    inv = _fd_inverse(be, b.shape, pitch) / i_mean
    apply_a = _fd_operator(be, wy.astype(be.real_dtype), wx.astype(be.real_dtype))

    def precond(r: Any) -> Any:
        # Projecting out the null space keeps single precision stable.
        return project(_trig(be, "c", _trig(be, "c", r, None) * inv, None, True))

    phi = be.zeros(b.shape)
    history: list[float] = []
    b_norm = math.sqrt(_kernels.dot(be, b, b))
    if b_norm == 0.0:
        return phi, history, 0, True, "zero right-hand side"
    r = b.copy()
    z = precond(r)
    p = z.copy()
    rz = _kernels.dot(be, r, z)
    converged = False
    message = f"reached max_iter={max_iter}"
    n_iter = 0
    while n_iter < max_iter:
        n_iter += 1
        ap = apply_a(p)
        pap = _kernels.dot(be, p, ap)
        if not pap > 0:
            # Only round-off is left (the operator is positive semi-definite).
            rel = math.sqrt(_kernels.dot(be, r, r)) / b_norm
            history.append(rel)
            converged = rel <= tol
            message = f"stagnated at relative residual {rel:.2e} (round-off)"
            break
        alpha = rz / pap
        phi += alpha * p
        r -= alpha * ap
        if n_iter % check_every == 0 or n_iter == max_iter:
            rel = math.sqrt(_kernels.dot(be, r, r)) / b_norm
            history.append(rel)
            if rel <= tol:
                converged = True
                message = f"relative residual {rel:.2e} <= tol"
                break
        z = precond(r)
        rz_new = _kernels.dot(be, r, z)
        if not rz_new > 0:
            rel = math.sqrt(_kernels.dot(be, r, r)) / b_norm
            history.append(rel)
            converged = rel <= tol
            message = f"preconditioned residual vanished at relative residual {rel:.2e}"
            break
        p *= rz_new / rz
        p += z
        rz = rz_new
    return phi, history, n_iter, converged, message


# -------------------------------------------------------------------- helpers
def _resolve_backend(array: Any, device: BackendLike, precision: str | None) -> Backend:
    if device is None:
        return backend_of(array, precision)
    return get_backend(device, precision)


def _pitch_pair(pitch: Any) -> tuple[float, float]:
    values = np.broadcast_to(np.asarray(pitch, dtype=np.float64), (2,))
    if not np.all(values > 0) or not np.all(np.isfinite(values)):
        raise ValueError(f"pitch must be positive and finite (metres), got {pitch!r}")
    return float(values[0]), float(values[1])


def _region_labels(be: Backend, mask: Any, per_region: bool) -> tuple[Any, int]:
    """Flattened region labels of ``mask`` (0 outside) and the number of regions."""
    if not per_region:
        return mask.ravel().astype(np.int64), 1
    from scipy import ndimage

    labels, n_regions = ndimage.label(be.to_numpy(mask))
    return be.xp.asarray(labels.ravel().astype(np.int64)), int(n_regions)


def _remove_region_means(be: Backend, a: Any, labels: Any, n_regions: int) -> Any:
    """Subtract from ``a`` its mean over each labelled region; zero outside them."""
    xp = be.xp
    if n_regions == 1:
        inside = labels.reshape(a.shape) > 0
        mean = xp.sum(xp.where(inside, a, 0.0)) / xp.maximum(xp.sum(inside), 1)
        return xp.where(inside, a - mean, 0.0).astype(a.dtype, copy=False)
    flat = a.ravel()
    sums = xp.bincount(labels, weights=flat, minlength=n_regions + 1)
    counts = xp.bincount(labels, minlength=n_regions + 1)
    means = sums / xp.maximum(counts, 1)
    out = xp.where(labels > 0, flat - means[labels].astype(a.dtype), 0.0)
    return out.reshape(a.shape).astype(a.dtype, copy=False)


# ------------------------------------------------------------------ public
def tie(
    intensities: Any,
    distances: Sequence[float] | Any,
    *,
    pitch: float | tuple[float, float],
    wavelength: float,
    method: TIEMethod = "dct",
    uniform: bool = False,
    derivative: TIEDerivative = "central",
    order: int | None = None,
    in_focus: Any = None,
    regularization: float = 1e-6,
    intensity_floor: float = 1e-3,
    mask: Any = None,
    tol: float = 1e-6,
    max_iter: int = 2000,
    check_every: int = 10,
    device: BackendLike = None,
    precision: str | None = None,
) -> TIEResult:
    """Retrieve the phase at ``z = 0`` from intensities at known defocus.

    Solves ``-k dI/dz = div(I grad phi)``; see the module notes and the
    transport-of-intensity guide page.

    Parameters
    ----------
    intensities:
        ``(N, ny, nx)`` intensity images (NumPy or CuPy), any consistent
        units, ``N >= 2``.
    distances:
        ``(N,)`` defocus of each image in metres, relative to the plane where
        the phase is wanted (positive downstream). May include ``0``.
    pitch:
        Pixel pitch in metres, scalar or ``(y, x)``.
    wavelength:
        Wavelength in metres.
    method:
        ``"dct"`` (Neumann boundaries, default), ``"fft"`` (periodic) or
        ``"pcg"`` (exact non-uniform solve inside ``mask``, Neumann).
    uniform:
        Use the uniform-intensity approximation ``I = mean(I0)`` (one Poisson
        solve) instead of the non-uniform solution.
    derivative:
        ``"central"`` (nearest planes about ``z = 0``) or ``"polyfit"``
        (least-squares polynomial over all planes).
    order:
        Polynomial degree for ``"polyfit"``; default ``min(N - 1, 3)``.
    in_focus:
        Optional ``(ny, nx)`` in-focus intensity; overrides the measured
        ``z = 0`` plane and the estimate.
    regularization:
        Dimensionless Tikhonov weight for ``"fft"``/``"dct"``: the lowest
        non-zero frequency is attenuated by ``1 / (1 + regularization)``.
        Ignored by ``"pcg"``.
    intensity_floor:
        Fraction of ``max(I0)`` below which the intensity is clipped: when
        dividing by it (``"fft"``/``"dct"``) and in the ``"pcg"`` weights,
        whose condition number it bounds. The default ``"pcg"`` mask is where
        ``I0`` exceeds it.
    mask:
        Optional ``(ny, nx)`` boolean support. The phase is zero outside it
        and its piston is removed over it. For ``"pcg"`` the equation is
        solved only inside it.
    tol, max_iter, check_every:
        ``"pcg"`` stopping rule: relative residual, iteration limit, and how
        often the residual is checked (each check syncs a GPU).
    device, precision:
        Backend; defaults to the backend of ``intensities``.

    Returns
    -------
    TIEResult
        Phase (radians) and OPD (metres) on the backend.
    """
    t0 = time.perf_counter()
    be = _resolve_backend(intensities, device, precision)
    xp = be.xp
    stack = be.asarray(intensities, dtype="real")
    if stack.ndim != 3 or stack.shape[0] < 2:
        raise ValueError(
            f"intensities must be an (N, ny, nx) stack with N >= 2, got shape {tuple(stack.shape)}"
        )
    z = np.asarray(distances, dtype=np.float64).ravel()
    if z.size != stack.shape[0]:
        raise ValueError(f"got {z.size} distances for {stack.shape[0]} images")
    if not np.all(np.isfinite(z)) or np.unique(z).size != z.size:
        raise ValueError("distances must be finite and distinct")
    if wavelength <= 0:
        raise ValueError("wavelength must be positive (metres)")
    pitch2 = _pitch_pair(pitch)
    method_name = str(method).lower()
    if method_name not in ("dct", "fft", "pcg"):
        raise ValueError(f"method must be 'dct', 'fft' or 'pcg', got {method!r}")
    if regularization < 0:
        raise ValueError("regularization must be non-negative")
    if not 0 < intensity_floor < 1:
        raise ValueError("intensity_floor must lie in (0, 1)")

    w_slope, w_value, focus = _axial_weights(z, derivative, order)
    rdt = be.real_dtype
    didz = xp.tensordot(xp.asarray(w_slope, dtype=rdt), stack, axes=1)
    if in_focus is not None:
        i0 = be.asarray(in_focus, dtype="real")
        if i0.shape != stack.shape[1:]:
            raise ValueError(f"in_focus must have shape {stack.shape[1:]}, got {i0.shape}")
    elif focus is not None:
        i0 = stack[focus]
    else:
        i0 = xp.tensordot(xp.asarray(w_value, dtype=rdt), stack, axes=1)
    i_max = float(xp.max(i0))
    if not i_max > 0:
        raise ValueError("the in-focus intensity is not positive anywhere")
    floor = intensity_floor * i_max

    if mask is not None:
        support = be.asarray(mask).astype(bool)
        if support.shape != i0.shape:
            raise ValueError(f"mask must have shape {i0.shape}, got {support.shape}")
    elif method_name == "pcg":
        support = i0 > floor
    else:
        support = xp.ones(i0.shape, dtype=bool)
    n_support = float(xp.sum(support))
    if n_support == 0:
        raise ValueError("the mask is empty")
    i_mean = float(xp.sum(xp.where(support, i0, 0.0))) / n_support

    k = 2.0 * math.pi / wavelength
    b = (k * didz).astype(rdt, copy=False)
    history: list[float] = []
    n_iter, converged, message = 0, True, "direct solve"
    labels, n_regions = _region_labels(be, support, method_name == "pcg")
    if method_name == "pcg":
        # Clipping bounds the condition number by 1 / intensity_floor.
        weight = xp.full(i0.shape, i_mean, dtype=rdt) if uniform else xp.maximum(i0, floor)
        # The support's net outflow is unobservable: keep the system consistent.
        b = _remove_region_means(be, b, labels, n_regions)
        phi, history, n_iter, converged, message = _solve_pcg(
            be,
            b,
            weight,
            support,
            pitch2,
            tol,
            int(max_iter),
            max(1, int(check_every)),
            lambda a: _remove_region_means(be, a, labels, n_regions),
        )
    else:
        if uniform:
            rhs, clipped = b / i_mean, None
        else:
            rhs, clipped = b, xp.maximum(i0, floor).astype(rdt, copy=False)
        solver = _solve_fft if method_name == "fft" else _solve_dct
        phi = solver(be, rhs, clipped, support, pitch2, float(regularization))
    phi = _remove_region_means(be, phi.astype(rdt, copy=False), labels, n_regions)
    be.synchronize()
    return TIEResult(
        phase=phi,
        opd=phi * (wavelength / (2.0 * math.pi)),
        intensity=i0,
        didz=didz,
        mask=support,
        wavelength=float(wavelength),
        method=method_name,
        derivative=str(derivative),
        uniform=bool(uniform),
        history=history,
        n_iter=n_iter,
        converged=converged,
        message=message,
        elapsed=time.perf_counter() - t0,
        device=be.device,
    )
