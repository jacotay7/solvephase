"""Device-resident optimizers.

Parameters stay on the backend (CPU or GPU) for the whole solve; only scalars
(objective values and directional derivatives) cross to the host for line
search decisions. The objective is a callable ``fun(x) -> (f, g)`` returning
a Python float and a backend gradient array.

* :func:`lbfgs` - limited-memory BFGS with a strong-Wolfe line search
  (Nocedal & Wright, *Numerical Optimization*, 2nd ed., Alg. 7.4, 3.5, 3.6).
* :func:`levenberg_marquardt` - damped Gauss-Newton for problems with few
  parameters, with Nielsen's damping update.
* :func:`adam` - first-order Adam (Kingma & Ba 2015), for very large or
  noisy objectives.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import _kernels
from .backend import Backend

__all__ = ["OptimizeResult", "adam", "lbfgs", "levenberg_marquardt"]

Objective = Callable[[Any], tuple[float, Any]]
Callback = Callable[[int, Any, float], bool | None]


@dataclass
class OptimizeResult:
    """Outcome of an optimizer run.

    Attributes
    ----------
    x:
        Final parameter vector (backend array).
    fun:
        Final objective value.
    n_iter:
        Iterations taken.
    n_eval:
        Objective evaluations.
    converged:
        Whether a convergence test (rather than the iteration limit) stopped
        the run.
    message:
        Why the run stopped.
    history:
        Objective value after each iteration (the first entry is the start).
    times:
        Wall-clock seconds since the start at each history entry.
    """

    x: Any
    fun: float
    n_iter: int
    n_eval: int
    converged: bool
    message: str
    history: list[float] = field(default_factory=list)
    times: list[float] = field(default_factory=list)


def _dot(backend: Backend, a: Any, b: Any) -> float:
    return _kernels.dot(backend, a, b)


def _axpy(backend: Backend, x: Any, alpha: float, y: Any) -> Any:
    """``x + alpha * y`` (one fused kernel on the GPU, same rounding)."""
    if backend.is_gpu and x.dtype == y.dtype and x.dtype.kind == "f":
        return _kernels.call("axpy", x, alpha, y)
    return x + alpha * y


def _cubic_min(x1: float, f1: float, g1: float, x2: float, f2: float, g2: float) -> float:
    """Minimizer of the cubic interpolating two points with slopes, clamped to the interval."""
    lo, hi = (x1, x2) if x1 <= x2 else (x2, x1)
    d1 = g1 + g2 - 3.0 * (f1 - f2) / (x1 - x2)
    disc = d1 * d1 - g1 * g2
    if disc >= 0.0:
        d2 = math.copysign(math.sqrt(disc), x2 - x1)
        denom = g2 - g1 + 2.0 * d2
        if denom != 0.0:
            t = x2 - (x2 - x1) * ((g2 + d2 - d1) / denom)
            if math.isfinite(t):
                return min(max(t, lo), hi)
    return 0.5 * (lo + hi)


def _strong_wolfe(
    fun: Objective,
    backend: Backend,
    x: Any,
    f0: float,
    g0: Any,
    d: Any,
    gtd0: float,
    step: float,
    *,
    c1: float = 1e-4,
    c2: float = 0.9,
    max_eval: int = 25,
) -> tuple[float, float, Any, int]:
    """Strong-Wolfe line search. Returns ``(step, f, g, n_evals)``."""
    n_eval = 0
    f_prev, g_prev, gtd_prev, t_prev = f0, g0, gtd0, 0.0
    f_new, g_new = fun(_axpy(backend, x, step, d))
    n_eval += 1
    gtd_new = _dot(backend, g_new, d)
    bracket: list[tuple[float, float, Any, float]] | None = None
    while n_eval < max_eval:
        if (
            not math.isfinite(f_new)
            or f_new > f0 + c1 * step * gtd0
            or (n_eval > 1 and f_new >= f_prev)
        ):
            bracket = [(t_prev, f_prev, g_prev, gtd_prev), (step, f_new, g_new, gtd_new)]
            break
        if abs(gtd_new) <= -c2 * gtd0:
            return step, f_new, g_new, n_eval
        if gtd_new >= 0:
            bracket = [(step, f_new, g_new, gtd_new), (t_prev, f_prev, g_prev, gtd_prev)]
            break
        t_next = _cubic_min(t_prev, f_prev, gtd_prev, step, f_new, gtd_new)
        t_next = min(max(t_next, step * 1.1), step * 10.0) if math.isfinite(f_new) else step * 0.5
        t_prev, f_prev, g_prev, gtd_prev = step, f_new, g_new, gtd_new
        step = t_next
        f_new, g_new = fun(_axpy(backend, x, step, d))
        n_eval += 1
        gtd_new = _dot(backend, g_new, d)
    if bracket is None:
        return step, f_new, g_new, n_eval

    # Zoom: bracket[0] is the low (best) end.
    lo, hi = bracket
    if lo[1] > hi[1] or not math.isfinite(lo[1]):
        lo, hi = hi, lo
    d_norm = max(1.0, float(np.sqrt(abs(_dot(backend, d, d)))))  # d is fixed: one dot
    while n_eval < max_eval:
        if abs(hi[0] - lo[0]) * d_norm < 1e-12:
            break
        if math.isfinite(hi[1]):
            t = _cubic_min(lo[0], lo[1], lo[3], hi[0], hi[1], hi[3])
        else:
            t = 0.5 * (lo[0] + hi[0])
        width = abs(hi[0] - lo[0])
        a, b = min(lo[0], hi[0]), max(lo[0], hi[0])
        if min(t - a, b - t) < 0.1 * width:  # keep away from the ends
            t = 0.5 * (a + b)
        f_t, g_t = fun(_axpy(backend, x, t, d))
        n_eval += 1
        gtd_t = _dot(backend, g_t, d)
        if not math.isfinite(f_t) or f_t > f0 + c1 * t * gtd0 or f_t >= lo[1]:
            hi = (t, f_t, g_t, gtd_t)
        else:
            if abs(gtd_t) <= -c2 * gtd0:
                return t, f_t, g_t, n_eval
            if gtd_t * (hi[0] - lo[0]) >= 0:
                hi = lo
            lo = (t, f_t, g_t, gtd_t)
    return lo[0], lo[1], lo[2], n_eval


def lbfgs(
    fun: Objective,
    x0: Any,
    backend: Backend,
    *,
    max_iter: int = 200,
    memory: int = 10,
    ftol: float = 1e-10,
    gtol: float = 1e-10,
    max_eval: int | None = None,
    callback: Callback | None = None,
) -> OptimizeResult:
    """Minimize with limited-memory BFGS and a strong-Wolfe line search.

    Parameters
    ----------
    fun:
        ``fun(x) -> (f, g)`` with ``f`` a Python float and ``g`` a backend array.
    x0:
        Starting point (backend array, any shape; treated as a flat vector).
    max_iter:
        Iteration limit.
    memory:
        Number of correction pairs kept.
    ftol:
        Stop when the relative decrease of ``f`` over an iteration is below
        ``ftol``.
    gtol:
        Stop when ``max |g|`` falls below ``gtol`` times its initial value.
    max_eval:
        Limit on objective evaluations (default ``10 * max_iter``).
    callback:
        ``callback(iteration, x, f)``; returning ``True`` stops the run.
    """
    t0 = time.perf_counter()
    xp = backend.xp
    x = x0.copy()
    f, g = fun(x)
    n_eval = 1
    history, times = [f], [time.perf_counter() - t0]
    g0_max = float(xp.max(xp.abs(g))) if g.size else 0.0
    if g0_max == 0.0:
        return OptimizeResult(x, f, 0, n_eval, True, "zero gradient at start", history, times)
    s_hist: list[Any] = []
    y_hist: list[Any] = []
    rho_hist: list[float] = []
    sy_yy: tuple[float, float] = (1.0, 1.0)  # s.y and y.y of the newest pair
    max_eval = max_eval or 10 * max_iter
    message, converged = "iteration limit reached", False
    it = 0
    for it in range(1, max_iter + 1):
        # Two-loop recursion.
        q = -g
        alphas = []
        for s, y, rho in zip(reversed(s_hist), reversed(y_hist), reversed(rho_hist)):
            a = rho * _dot(backend, s, q)
            alphas.append(a)
            q = _axpy(backend, q, -a, y)
        if s_hist:
            # s.y and y.y of the newest pair were computed when it was stored.
            gamma = sy_yy[0] / sy_yy[1]
            q = q * gamma
        for (s, y, rho), a in zip(zip(s_hist, y_hist, rho_hist), reversed(alphas)):
            b = rho * _dot(backend, y, q)
            q = _axpy(backend, q, a - b, s)
        d = q
        gtd = _dot(backend, g, d)
        if gtd > -1e-30:
            # Not a descent direction: reset the memory and use steepest descent.
            s_hist.clear()
            y_hist.clear()
            rho_hist.clear()
            d = -g
            gtd = _dot(backend, g, d)
        step = 1.0 if s_hist else min(1.0, 1.0 / max(float(xp.sum(xp.abs(g))), 1e-30))
        step, f_new, g_new, n = _strong_wolfe(fun, backend, x, f, g, d, gtd, step)
        n_eval += n
        if not math.isfinite(f_new) or f_new > f:
            message = "line search failed to decrease the objective"
            converged = True
            break
        s = step * d
        x = x + s
        y = g_new - g
        sy = _dot(backend, s, y)
        yy = _dot(backend, y, y)
        if sy > 1e-12 * math.sqrt(_dot(backend, s, s) * yy):
            s_hist.append(s)
            y_hist.append(y)
            rho_hist.append(1.0 / sy)
            sy_yy = (sy, yy)
            if len(s_hist) > memory:
                s_hist.pop(0), y_hist.pop(0), rho_hist.pop(0)
        f_old, f, g = f, f_new, g_new
        history.append(f)
        times.append(time.perf_counter() - t0)
        if callback is not None and callback(it, x, f):
            message, converged = "stopped by callback", True
            break
        if abs(f_old - f) <= ftol * max(abs(f_old), abs(f), 1e-300):
            message, converged = "relative objective change below ftol", True
            break
        if float(xp.max(xp.abs(g))) <= gtol * g0_max:
            message, converged = "gradient below gtol", True
            break
        if n_eval >= max_eval:
            message = "evaluation limit reached"
            break
    return OptimizeResult(x, f, it, n_eval, converged, message, history, times)


def levenberg_marquardt(
    system: Callable[[Any], tuple[float, Any, Any]],
    value: Callable[[Any], float],
    x0: Any,
    backend: Backend,
    *,
    max_iter: int = 50,
    ftol: float = 1e-10,
    xtol: float = 1e-10,
    damping: float = 1e-3,
    callback: Callback | None = None,
) -> OptimizeResult:
    """Damped Gauss-Newton (Levenberg-Marquardt) for small parameter counts.

    Parameters
    ----------
    system:
        ``system(x) -> (f, g, H)``: objective, gradient and a positive
        semi-definite Hessian approximation (Gauss-Newton/Fisher), as backend
        arrays except ``f``.
    value:
        ``value(x) -> f`` for trial steps.
    damping:
        Initial damping relative to the largest diagonal of ``H``.

    Notes
    -----
    The step solves ``(H + mu diag(H)) dx = -g`` (Marquardt scaling) on the
    host in double precision; ``mu`` follows Nielsen's update.
    """
    t0 = time.perf_counter()
    x = x0.copy()
    f, g, h = system(x)
    n_eval = 1
    history, times = [f], [time.perf_counter() - t0]
    nu = 2.0
    mu: float | None = None
    message, converged = "iteration limit reached", False
    it = 0
    for it in range(1, max_iter + 1):
        g_h = np.asarray(backend.to_numpy(g), dtype=np.float64)
        h_h = np.asarray(backend.to_numpy(h), dtype=np.float64)
        diag = np.maximum(np.diag(h_h), 1e-12 * max(np.max(np.diag(h_h)), 1e-300))
        if mu is None:
            mu = damping
        accepted = False
        while not accepted:
            a = h_h + mu * np.diag(diag)
            try:
                dx_h = -np.linalg.solve(a, g_h)
            except np.linalg.LinAlgError:
                dx_h = -np.linalg.lstsq(a, g_h, rcond=None)[0]
            predicted = -(g_h @ dx_h + 0.5 * dx_h @ h_h @ dx_h)
            dx = backend.asarray(dx_h, dtype=x.dtype)
            f_try = value(x + dx)
            n_eval += 1
            actual = f - f_try
            if math.isfinite(f_try) and actual > 0:
                ratio = actual / predicted if predicted > 0 else 1.0
                mu *= max(1.0 / 3.0, 1.0 - (2.0 * float(ratio) - 1.0) ** 3)
                nu = 2.0
                accepted = True
            else:
                mu *= nu
                nu *= 2.0
                if mu > 1e16:
                    break
        if not accepted:
            message, converged = "no decreasing step found (converged to noise level)", True
            break
        step_norm = float(np.linalg.norm(dx_h))
        x_norm = float(np.linalg.norm(backend.to_numpy(x)))
        x = x + dx
        f_old = f
        f, g, h = system(x)
        n_eval += 1
        history.append(f)
        times.append(time.perf_counter() - t0)
        if callback is not None and callback(it, x, f):
            message, converged = "stopped by callback", True
            break
        if abs(f_old - f) <= ftol * max(abs(f_old), abs(f), 1e-300):
            message, converged = "relative objective change below ftol", True
            break
        if step_norm <= xtol * (x_norm + xtol):
            message, converged = "step below xtol", True
            break
    return OptimizeResult(x, f, it, n_eval, converged, message, history, times)


def adam(
    fun: Objective,
    x0: Any,
    backend: Backend,
    *,
    max_iter: int = 500,
    learning_rate: float = 0.05,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    ftol: float = 0.0,
    callback: Callback | None = None,
) -> OptimizeResult:
    """First-order Adam with bias correction.

    ``learning_rate`` is in parameter units per step (internal parameters are
    radians of phase, so ~0.01-0.1 is typical).
    """
    t0 = time.perf_counter()
    xp = backend.xp
    x = x0.copy()
    m = xp.zeros_like(x)
    v = xp.zeros_like(x)
    f, g = fun(x)
    history, times = [f], [time.perf_counter() - t0]
    best_x, best_f = x.copy(), f
    message, converged = "iteration limit reached", False
    it = 0
    for it in range(1, max_iter + 1):
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        mhat = m / (1 - beta1**it)
        vhat = v / (1 - beta2**it)
        x = x - learning_rate * mhat / (xp.sqrt(vhat) + eps)
        f_old = f
        f, g = fun(x)
        history.append(f)
        times.append(time.perf_counter() - t0)
        if f < best_f:
            best_x, best_f = x.copy(), f
        if callback is not None and callback(it, x, f):
            message, converged = "stopped by callback", True
            break
        if ftol > 0 and abs(f_old - f) <= ftol * max(abs(f_old), abs(f), 1e-300):
            message, converged = "relative objective change below ftol", True
            break
    return OptimizeResult(best_x, best_f, it, it + 1, converged, message, history, times)
