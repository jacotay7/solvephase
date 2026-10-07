"""Fused elementwise kernels for the GPU hot paths.

On small problems a GPU solve is bound by the host, not the device: every CuPy
operation costs about 15-30 us of Python and launch overhead (more on slow
host cores), while the arithmetic on a 128x128 array takes a few
microseconds. Each kernel here replaces a chain of 3-20 CuPy operations
(and their temporaries) with one launch.

The kernels evaluate the same expressions as the NumPy code paths they
replace, in the same order and precision, and are compiled with
``--fmad=false`` so that no multiply-add is contracted. They differ from the
unfused CuPy chain at most by rounding in the last bit; kernels with products
of two complex arrays are compiled like CuPy's own multiply kernel to round
the same way. On CuPy 14 / CUDA 12 every kernel reproduces the unfused chain
bit for bit.

The kernels need CuPy: callers use them on the GPU only and keep their NumPy
path on the CPU (:func:`dot` handles both).
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np

_OPTIONS = ("--fmad=false",)

# NaN-propagating maximum, as cupy.maximum / numpy.maximum.
_PREAMBLE = r"""
#ifndef NAN
#define NAN __int_as_float(0x7fffffff)
#endif
template <typename T> __device__ __forceinline__ T sp_max(T a, T b) {
    return (isnan(a) | isnan(b)) ? T(NAN) : max(a, b);
}
// Subtraction that is never contracted into a multiply-add.
__device__ __forceinline__ float sp_sub(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ double sp_sub(double a, double b) { return __dsub_rn(a, b); }
"""


@functools.cache
def _elementwise(name: str) -> Any:  # pragma: no cover - GPU only
    import cupy

    in_params, out_params, body = _ELEMENTWISE[name]
    # Products of two complex arrays (and complex abs) are written with thrust
    # complex operations and compiled with CuPy's default flags, like CuPy's own
    # multiply and abs kernels, so they round exactly as the unfused expression.
    options = () if name in _CUPY_FLAGS else _OPTIONS
    return cupy.ElementwiseKernel(
        in_params, out_params, body, f"solvephase_{name}", preamble=_PREAMBLE, options=options
    )


@functools.cache
def _loss_floor_kernel() -> Any:  # pragma: no cover - GPU only
    import cupy

    # A maximum does not depend on the reduction order, so this matches
    # ``1e-6 * max(abs(maximum(data + shift, 0))) + 1e-30`` (losses._floor) exactly.
    return cupy.ReductionKernel(
        "float64 data, float64 shift",
        "float64 out",
        "abs(sp_max(data + shift, 0.0))",
        "sp_max(a, b)",
        "out = 1e-6 * a + 1e-30",
        "0",
        "solvephase_loss_floor",
        preamble=_PREAMBLE,
        options=_OPTIONS,
    )


_ELEMENTWISE: dict[str, tuple[str, str, str]] = {
    # FocalPlaneModel.forward: phi = (phase + div) * ratio, exp(i phi), amp * exp(i phi).
    "pupil_field": (
        "T phase, T div, T ratio, T amp",
        "C phasor, C u",
        "T p = (phase + div) * ratio; phasor = exp(C(0, p)); u = amp * phasor;",
    ),
    # Spectrally weighted intensity w |e|^2 / s (s = norm for one wavelength, else 1).
    "intensity": (
        "C e, T w, T s",
        "T out",
        "out = (w * (e.real() * e.real() + e.imag() * e.imag())) / s;",
    ),
    # FocalPlaneModel.vjp: g * E * (2 w / norm).
    "scale_field": ("T g, C e, T s", "C out", "out = C((g * e.real()) * s, (g * e.imag()) * s);"),
    # Im(conj(u) v) * ratio.
    "cross_imag": ("C u, C v, T r", "T out", "out = (conj(u) * v).imag() * r;"),
    # FocalPlaneModel.jvp: (i d ratio) u.
    "direction_field": (
        "T d, T r, C u",
        "C out",
        "T s = d * r; out = C(-(s * u.imag()), s * u.real());",
    ),
    # w Re(conj(e) de), scaled by s (s = 2 / norm for one wavelength).
    "cross_real": ("C e, C de, T w, T s", "T out", "out = (w * (conj(e) * de).real()) * s;"),
    # Losses: model arrives in working precision and is evaluated in float64
    # (``m`` below); the gradient is stored back in working precision.
    "poisson": (
        "M model, float64 data, float64 w, float64 shift, float64 m0",
        "float64 term, G grad, float64 curv",
        """
        double dd = sp_max(data + shift, 0.0);
        double t = (double)model + shift;
        double m = sp_max(t, m0);
        double ratio = dd > 0 ? dd / m : 1.0;
        double f = (m - dd) + dd * log(ratio);
        double g = 1.0 - dd / m;
        double h2 = dd / (m * m);
        double delta = t < m0 ? t - m0 : 0.0;
        term = w * (f + delta * (g + 0.5 * h2 * delta));
        grad = (G)(w * (g + h2 * delta));
        curv = w / m;
        """,
    ),
    "amplitude": (
        "M model, float64 data, float64 w, float64 m0",
        "float64 term, G grad, float64 curv",
        """
        double mm = (double)model;
        double dd = sp_max(data, 0.0);
        double m = sp_max(mm, m0);
        double sm = sqrt(m);
        double sd = sqrt(dd);
        double resid = sm - sd;
        double f = 0.5 * resid * resid;
        double g = resid / (2.0 * sm);
        double h2 = sd / (4.0 * m * sm);
        double delta = mm < m0 ? mm - m0 : 0.0;
        term = w * (f + delta * (g + 0.5 * h2 * delta));
        grad = (G)(w * (g + h2 * delta));
        curv = w / (4.0 * m);
        """,
    ),
    "gaussian": (
        "M model, float64 data, float64 w",
        "float64 term, G grad",
        "double resid = (double)model - data; double wr = w * resid; "
        "term = wr * resid; grad = (G)wr;",
    ),
    # |a|^2 (the real working type is fixed by the output passed).
    "abs2": ("C a", "T out", "out = a.real() * a.real() + a.imag() * a.imag();"),
    # Phase diversity (Paxman et al. 1992): the residual d - o s; the metric terms
    # w (t + gamma |o|^2); the gradient spectrum c (m conj(o)) r. (a * conj(b) is
    # left to CuPy: inlining the conjugate changes how the product rounds.)
    "pd_resid": (
        "C d, C o, C s",
        "C out",
        "C os = o * s; out = C(sp_sub(d.real(), os.real()), sp_sub(d.imag(), os.imag()));",
    ),
    "pd_terms": (
        "T terms, T w, T gamma, C o",
        "T out",
        "out = w * (terms + gamma * (o.real() * o.real() + o.imag() * o.imag()));",
    ),
    "pd_grad": (
        "T c, T m, C o, C r",
        "C out",
        "C t = C(c, 0) * (C(m, 0) * conj(o)); out = t * r;",
    ),
    # x + alpha * y (optimizer updates).
    "axpy": ("T x, T alpha, T y", "T out", "out = x + alpha * y;"),
    # Gerchberg-Saxton modulus projection: where(measured, target * win / max(|win|, 1e-30), win)
    # with target = sqrt_signal * scale.
    "gs_modulus": (
        "bool measured, T sqrt_signal, T scale, C win",
        "C out",
        "T target = sqrt_signal * scale; "
        "out = measured ? (C(target, 0) * win) / C(sp_max(abs(win), (T)1e-30), 0) : win;",
    ),
    # Gerchberg-Saxton measured power |win|^2 * measured (summed by CuPy: a custom
    # reduction would sum in another order).
    "gs_power": ("C win, bool measured", "T out", "out = (abs(win) * abs(win)) * (T)measured;"),
    # Fast & Furious (Korkiakoski et al. 2014) energy normalization and Strehl compensation,
    # given total = sum(p) and peak = max(p) of the raw image.
    "ff_normalize": (
        "T p, T total, T peak, T sum_a2, T max_a2, T a2",
        "T out",
        "T c = sum_a2 / (total > 0 ? total : (T)1); "
        "out = p * c + ((T)1 - (peak * c) / max_a2) * a2;",
    ),
    "ff_normalize_plain": (
        "T p, T total, T sum_a2",
        "T out",
        "out = p * (sum_a2 / (total > 0 ? total : (T)1));",
    ),
    # Pupil field of a DM change, amp * (-k) * change, as a complex array.
    "ff_change": ("T amp, T neg_k, T change", "C out", "out = C((amp * neg_k) * change, 0);"),
    # Even/odd image parts (a 180-degree rotation reverses the flattened array),
    # the odd field y and |v| (Eqs. 14-17).
    "ff_parts": (
        "raw T p, T a, T a2, T y_den",
        "T p_even, T y, T v_abs",
        "T pf = p[_ind.size() - 1 - i]; T pv = p[i]; "
        "p_even = (T)0.5 * (pv + pf); T p_odd = (T)0.5 * (pv - pf); "
        "y = ((-a) * p_odd) / y_den; "
        "v_abs = sqrt(abs((p_even - a2) - y * y));",
    ),
    # Focal field (v + i y) * window, v = v_abs * sign.
    "ff_field": (
        "T v_abs, T sign, T y, T window",
        "C out",
        "T v = v_abs * sign; out = C(v * window, y * window);",
    ),
    # Even-part sign from the previous frame and the DM change (Eqs. 18-21).
    "ff_field_signed": (
        "T v_abs, T y, T window, T prev_even, T p_even, C div",
        "C out",
        "T vd = div.real(); T yd = div.imag(); "
        "T diff = (((prev_even - p_even) - vd * vd) - yd * yd) - ((T)2 * y) * yd; "
        "T dv = diff * vd; T sg = dv > 0 ? (T)1 : (dv < 0 ? (T)-1 : dv); "
        "T v = v_abs * sg; out = C(v * window, y * window);",
    ),
    # OPD = Re(A phi) / A and its odd and even parts.
    "ff_opd": (
        "raw C a_phi, raw T inv_amp",
        "T opd, T odd, T even",
        "size_t j = _ind.size() - 1 - i; "
        "T o = a_phi[i].real() * inv_amp[i]; T of = a_phi[j].real() * inv_amp[j]; "
        "opd = o; odd = (T)0.5 * (o - of); even = (T)0.5 * (o + of);",
    ),
}

_CUPY_FLAGS = frozenset(
    {"cross_imag", "cross_real", "gs_modulus", "gs_power", "pd_resid", "pd_grad"}
)


def pupil_field(phase: Any, div: Any, ratio: Any, amp: Any, cdtype: Any) -> tuple[Any, Any]:
    """``(phasor, u)`` from ``phase (K|1, 1, ny, nx)``, ``div (K, 1, ny, nx)``, ``ratio (L, 1, 1)``.

    Both outputs have the broadcast shape ``(K, L, ny, nx)``.
    """
    import cupy

    shape = np.broadcast_shapes(phase.shape, div.shape, ratio.shape, amp.shape)
    phasor = cupy.empty(shape, dtype=cdtype)
    u = cupy.empty(shape, dtype=cdtype)
    _elementwise("pupil_field")(phase, div, ratio, amp, phasor, u)
    return phasor, u


def abs2(a: Any) -> Any:  # pragma: no cover - GPU only
    """``a.real**2 + a.imag**2`` of a complex array in one launch."""
    import cupy

    out = cupy.empty(a.shape, dtype=a.real.dtype)
    _elementwise("abs2")(a, out)
    return out


def call(name: str, *args: Any) -> Any:  # pragma: no cover - GPU only
    """Run the fused elementwise kernel ``name``.

    Outputs are allocated by CuPy (broadcast shape) unless they are passed as
    the trailing arguments; outputs typed by a placeholder no input binds
    must be passed.
    """
    return _elementwise(name)(*args)


def loss_floor(data: Any, shift: float) -> Any:
    """0-d device array ``1e-6 max|max(data + shift, 0)| + 1e-30`` in one launch."""
    return _loss_floor_kernel()(data, np.float64(shift))


def dot(backend: Any, a: Any, b: Any) -> float:
    """``backend.dot(a, b)``: ``Re(sum(conj(a) b))`` as a Python float.

    For real arrays on the GPU it is evaluated as ``(a * b).sum()``, which
    rounds exactly as ``cupy.vdot`` does (checked on random vectors of both
    precisions) with about 30 us less host overhead per call.
    """
    if backend.is_gpu and a.dtype.kind == b.dtype.kind == "f":  # pragma: no cover - GPU only
        return float((a.reshape(-1) * b.reshape(-1)).sum())
    return float(backend.dot(a, b))
