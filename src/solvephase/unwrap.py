"""Least-squares phase unwrapping over arbitrary (obscured, segmented) pupils.

Iterative-transform algorithms return phase wrapped into ``(-pi, pi]``. The
weighted least-squares unwrapper of Ghiglia & Romero (JOSA A 11, 107, 1994)
finds the smooth phase whose finite differences best match the wrapped
differences of the input, using only pixel pairs that are both inside the
mask. It is solved by conjugate gradients preconditioned with the unweighted
Neumann Poisson solver (a DCT), on CPU or GPU.

Least-squares unwrapping is exact when the true phase changes by less than pi
between neighbouring pixels. Spiders and segment gaps split a pupil into
disconnected regions whose relative 2 pi multiples the data cannot fix. With
``bridge=True`` (default) each region's multiple of 2 pi is chosen so that the
regions join a smooth low-order polynomial surface; that assumes the true
wavefront is continuous across the gap, which is right for spider vanes but
is a guess for independently pistoned mirror segments.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .backend import BackendLike, backend_of, get_backend

__all__ = ["unwrap_phase", "wrap"]


def wrap(phase: Any) -> Any:
    """Wrap phase into ``(-pi, pi]``."""
    xp = _xp(phase)
    return phase - 2.0 * math.pi * xp.round(phase / (2.0 * math.pi))


def _xp(array: Any) -> Any:
    return backend_of(array).xp


def _dct2(be: Any, a: Any, inverse: bool = False) -> Any:
    if be.is_gpu:
        from cupyx.scipy import fft as cfft

        return (cfft.idctn if inverse else cfft.dctn)(a, type=2, norm="ortho")
    from scipy import fft

    return (fft.idctn if inverse else fft.dctn)(a, type=2, norm="ortho")


def unwrap_phase(
    wrapped: Any,
    mask: Any = None,
    *,
    weights: Any = None,
    tol: float = 1e-8,
    max_iter: int = 500,
    bridge: bool = True,
    bridge_order: int = 4,
    device: BackendLike = None,
) -> Any:
    """Weighted least-squares phase unwrapping.

    Parameters
    ----------
    wrapped:
        ``(ny, nx)`` wrapped phase in radians (NumPy or CuPy).
    mask:
        Boolean support; pixels outside are ignored and returned as 0.
        Defaults to everything.
    weights:
        Optional per-pixel reliability in ``[0, 1]`` (e.g. pupil amplitude);
        a pair is weighted by the smaller of its two pixel weights.
    tol:
        Relative residual for the conjugate-gradient stop.
    max_iter:
        Conjugate-gradient iteration limit.
    bridge:
        Choose the relative 2 pi multiples of disconnected regions so they
        join a smooth surface (see module notes).
    bridge_order:
        Total degree of the polynomial surface used for bridging.
    device:
        Backend override; defaults to the backend of ``wrapped``.

    Returns
    -------
    Unwrapped phase on the same backend, zero outside ``mask``. Within each
    connected region its mean matches the wrapped input's circular mean.
    """
    be = backend_of(wrapped) if device is None else get_backend(device)
    xp = be.xp
    psi = be.asarray(wrapped, dtype=np.float64)
    ny, nx = psi.shape
    m = xp.ones((ny, nx), dtype=bool) if mask is None else be.asarray(mask).astype(bool)
    w = m.astype(np.float64) if weights is None else be.asarray(weights, dtype=np.float64) * m
    wx = xp.minimum(w[:, 1:], w[:, :-1])
    wy = xp.minimum(w[1:, :], w[:-1, :])
    dx = wrap(psi[:, 1:] - psi[:, :-1]) * wx
    dy = wrap(psi[1:, :] - psi[:-1, :]) * wy

    def div_t(gx: Any, gy: Any) -> Any:
        # Adjoint of the forward difference operator: D^T g.
        out = xp.zeros((ny, nx))
        out[:, :-1] -= gx
        out[:, 1:] += gx
        out[:-1, :] -= gy
        out[1:, :] += gy
        return out

    def apply_a(phi: Any) -> Any:
        gx = (phi[:, 1:] - phi[:, :-1]) * wx
        gy = (phi[1:, :] - phi[:-1, :]) * wy
        return div_t(gx, gy)

    p_idx = xp.arange(ny, dtype=np.float64)[:, None]
    q_idx = xp.arange(nx, dtype=np.float64)[None, :]
    eig = (2.0 - 2.0 * xp.cos(math.pi * p_idx / ny)) + (2.0 - 2.0 * xp.cos(math.pi * q_idx / nx))
    eig[0, 0] = 1.0

    def precond(r: Any) -> Any:
        spec = _dct2(be, r) / eig
        spec[0, 0] = 0.0
        return _dct2(be, spec, inverse=True)

    b = div_t(dx, dy)
    phi = xp.zeros((ny, nx))
    r = b - apply_a(phi)
    z = precond(r)
    p = z.copy()
    rz = be.dot(r, z)
    b_norm = math.sqrt(be.dot(b, b))
    if b_norm == 0.0:
        out = xp.where(m, 0.0, 0.0)
    else:
        for _ in range(max_iter):
            ap = apply_a(p)
            pap = be.dot(p, ap)
            if pap <= 0:
                break
            alpha = rz / pap
            phi = phi + alpha * p
            r = r - alpha * ap
            if math.sqrt(be.dot(r, r)) <= tol * b_norm:
                break
            z = precond(r)
            rz_new = be.dot(r, z)
            p = z + (rz_new / rz) * p
            rz = rz_new
        out = phi
    # Restore each connected region's level from the wrapped data.
    host_mask = be.to_numpy(m)
    from scipy import ndimage

    labels, n_regions = ndimage.label(host_mask)
    result = be.to_numpy(out).copy()
    host_psi = be.to_numpy(psi)
    for region in range(1, n_regions + 1):
        sel = labels == region
        offset = np.angle(np.mean(np.exp(1j * (host_psi[sel] - result[sel]))))
        result[sel] += offset
    if bridge and n_regions > 1:
        result = _bridge_regions(result, labels, n_regions, bridge_order)
    result[~host_mask] = 0.0
    return be.asarray(result, dtype=be.real_dtype if be.precision == "single" else np.float64)


def _bridge_regions(
    phase: np.ndarray, labels: np.ndarray, n_regions: int, order: int
) -> np.ndarray:
    """Shift disconnected regions by multiples of 2 pi to join a smooth surface."""
    sel = labels > 0
    ny, nx = phase.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    scale = max(ny, nx) / 2.0
    y = (yy[sel] - (ny - 1) / 2.0) / scale
    x = (xx[sel] - (nx - 1) / 2.0) / scale
    lab = labels[sel]
    # Ignore tiny regions (isolated edge pixels) when fitting; shift them all the same.
    sizes = np.bincount(lab, minlength=n_regions + 1)
    cols = [x**i * y**j for i in range(order + 1) for j in range(order + 1 - i)]
    region_cols = [(lab == r).astype(float) for r in range(2, n_regions + 1)]
    a = np.column_stack(cols + region_cols)
    out = phase.copy()
    for _ in range(3):
        coef, *_ = np.linalg.lstsq(a, out[sel], rcond=None)
        pistons = coef[len(cols) :]
        shifts = np.round(pistons / (2.0 * np.pi))
        if not np.any(shifts):
            break
        for r, n in zip(range(2, n_regions + 1), shifts):
            if n and sizes[r] > 0:
                out[labels == r] -= 2.0 * np.pi * n
    return out
