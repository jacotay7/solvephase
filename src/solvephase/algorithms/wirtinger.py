"""Phase retrieval for generic linear measurement models: the Wirtinger-flow family.

Recover ``x`` (up to a global phase) from ``y = |A x|^2`` for a known linear
operator ``A`` (:mod:`solvephase.operators`). The algorithms are the gradient
methods catalogued by PhasePack (Chandra et al., Asilomar 2017), each started
from a spectral-type initialization:

Initializations
    * spectral - leading eigenvector of ``A^H diag(y) A`` (Netrapalli, Jain &
      Sanghavi, NeurIPS 2013; Candès, Li & Soltanolkotabi, IEEE Trans. Inf.
      Theory 61, 1985, 2015);
    * truncated spectral - drops measurements above ``alpha_y^2 mean(y)``
      (Chen & Candès, Comm. Pure Appl. Math. 70, 822, 2017);
    * orthogonality-promoting - the ``m/6`` largest amplitudes, unweighted
      (Wang, Giannakis & Eldar, IEEE Trans. Inf. Theory 64, 773, 2018);
    * weighted maximal correlation - the ``3m/13`` largest amplitudes weighted
      by ``psi^(1/2)`` (Wang, Giannakis, Saad & Chen, IEEE Trans. Signal
      Process. 66, 2818, 2018);
    * optimal preprocessing - ``T(y) = (y - 1) / (y + sqrt(delta) - 1)`` with
      ``y`` normalized to unit mean and ``delta = m / n`` (Luo, Alghamdi & Lu,
      IEEE Trans. Signal Process. 67, 2347, 2019);
    * random.

Algorithms
    * ``"wf"`` - Wirtinger Flow on the intensity loss with the step schedule
      ``mu_t = min(1 - exp(-t / tau0), mu_max)`` (Candès et al. 2015;
      ``tau0 = 330``; ``mu_max`` defaults to 0.1 because the paper's 0.2
      diverges or stalls on many Gaussian instances with ``n >= 128``);
    * ``"twf"`` - Truncated Wirtinger Flow on the Poisson likelihood (Chen &
      Candès 2017);
    * ``"taf"`` - Truncated Amplitude Flow (Wang, Giannakis & Eldar 2018);
    * ``"raf"`` - Reweighted Amplitude Flow (Wang, Giannakis, Saad & Chen 2018);
    * ``"lbfgs"`` - L-BFGS (:func:`solvephase.optimize.lbfgs`) on the amplitude
      or intensity least-squares loss, using a real view of the complex vector.

Scaling. The literature states every rule for the unit-variance complex
Gaussian model (``E |A_kj|^2 = 1``). Internally the operator is rescaled to
that convention, ``A -> sqrt(s) A`` and ``y -> s y`` with
``s = m n / ||A||_F^2``, so the published step sizes and truncation thresholds
apply unchanged to any operator (for coded diffraction with a unitary FFT this
reproduces the unnormalized-FFT model of the CDP papers). Measurement vectors
are assumed to have nearly equal norms, which holds for the Gaussian and CDP
models; the row-norm normalizations of the TAF/RAF initializations are
therefore dropped.

Whitening. When the operator reports ``G = diag(A^H A)``
(:meth:`~solvephase.operators.LinearOperator.gram_diagonal`; every built-in
operator does), the power method runs on ``G^{-1/2} Y G^{-1/2}`` and maps the
eigenvector back with ``G^{1/2}``. For the Gaussian model ``G ~ m I`` and
nothing changes; for coded diffraction with octanary masks it removes
eigenvectors localized on pixels where most masks are strong, which otherwise
beat the signal eigenvector for images of about ``10^4`` pixels or more.

All iterations stay on the backend device; the residual is transferred to the
host only every ``check_every`` iterations.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ..backend import Backend, backend_of, to_numpy
from ..operators import LinearOperator, _norm
from ..optimize import lbfgs

__all__ = [
    "GenericResult",
    "init_optimal_spectral",
    "init_orthogonality_promoting",
    "init_random",
    "init_spectral",
    "init_truncated_spectral",
    "init_weighted_correlation",
    "relative_error",
    "wirtinger",
]

Callback = Callable[[int, Any, float], "bool | None"]

METHODS = ("wf", "twf", "taf", "raf", "lbfgs")
INITS = ("spectral", "truncated", "orthogonal", "weighted", "optimal", "random")

#: Initialization used by ``init="auto"``: each algorithm's own paper, except WF, whose
#: plain spectral start is replaced by the truncated one (identical on Gaussian data, and
#: the plain one is dominated by the heavy tail of ``y`` for coded diffraction).
DEFAULT_INIT = {
    "wf": "truncated",
    "twf": "truncated",
    "taf": "orthogonal",
    "raf": "weighted",
    "lbfgs": "optimal",
}

#: Default parameters per method; ``step`` is ``mu`` (``mu_max`` for WF).
DEFAULT_OPTIONS: dict[str, dict[str, float]] = {
    "wf": {"step": 0.1, "tau0": 330.0},
    "twf": {"step": 0.2, "alpha_lb": 0.3, "alpha_ub": 5.0, "alpha_h": 5.0},
    "taf": {"step": 1.0, "gamma": 0.7},
    "raf": {"step": 4.0, "beta": 5.0},
    "lbfgs": {"memory": 10},
}


# --------------------------------------------------------------------- result
@dataclass
class GenericResult:
    """Outcome of :func:`wirtinger`.

    Attributes
    ----------
    x:
        Recovered vector, shape ``operator.in_shape`` (backend array), in the
        units of the true signal (the global phase is arbitrary).
    history:
        Relative amplitude residual ``|| |A x| - sqrt(y) || / || sqrt(y) ||``
        at the start and at every check.
    times:
        Wall-clock seconds since the call started, for each history entry
        (the first entry includes the initialization).
    n_iter:
        Iterations performed.
    converged:
        Whether a tolerance (rather than the iteration limit) stopped the run.
    message:
        Why the run stopped.
    elapsed:
        Total wall-clock seconds, including the initialization.
    device:
        ``"cpu"`` or ``"gpu"``.
    method:
        Algorithm used.
    init:
        Initialization used (``"given"`` for a user-supplied start).
    options:
        Method parameters used (``step`` and the method's options).
    """

    x: Any
    history: list[float]
    times: list[float]
    n_iter: int
    converged: bool
    message: str
    elapsed: float
    device: str
    method: str
    init: str
    options: dict[str, float] = field(default_factory=dict)

    @property
    def residual(self) -> float:
        """Final relative amplitude residual."""
        return self.history[-1]

    def to_numpy(self) -> GenericResult:
        """A copy with :attr:`x` on the host as a NumPy array."""
        return replace(self, x=to_numpy(self.x))


# ---------------------------------------------------------------- utilities
def relative_error(x: Any, truth: Any) -> float:
    """Distance to the truth modulo the global phase.

    ``min_phi ||x - exp(i phi) truth|| / ||truth||``, with the optimal phase
    ``exp(i phi) = truth^H x / |truth^H x|``. Works for NumPy or CuPy input
    (computed on the device of ``x``).
    """
    be = backend_of(x)
    xp = be.xp
    a = be.asarray(x, dtype="complex")
    t = be.asarray(truth, dtype="complex")
    c = complex(xp.sum(xp.conj(t) * a))
    phase = c / abs(c) if c != 0 else 1.0
    d = a - phase * t
    return math.sqrt(be.dot(d, d) / max(be.dot(t, t), 1e-300))


def _abs2(u: Any) -> Any:
    return u.real * u.real + u.imag * u.imag


class _Model:
    """The operator and data rescaled to the unit-variance Gaussian convention."""

    def __init__(self, operator: LinearOperator, measurements: Any, magnitudes: bool) -> None:
        if not isinstance(operator, LinearOperator):
            raise TypeError(
                f"operator must be a solvephase.operators.LinearOperator, got "
                f"{type(operator).__name__}; wrap a dense matrix in MatrixOperator"
            )
        be = operator.backend
        xp = be.xp
        self.operator = operator
        self.backend: Backend = be
        if getattr(measurements, "dtype", np.dtype(float)).kind == "c":
            raise ValueError(
                "measurements must be real intensities (or magnitudes with magnitudes=True), "
                "got a complex array"
            )
        data = be.asarray(measurements, dtype="real")
        if tuple(data.shape) != tuple(operator.out_shape):
            raise ValueError(
                f"measurements have shape {tuple(data.shape)} but the operator produces "
                f"{tuple(operator.out_shape)}"
            )
        if magnitudes:
            psi = xp.abs(data)
            y = psi * psi
        else:
            # Noise can make intensities negative; they carry no amplitude.
            y = xp.maximum(data, 0)
            psi = xp.sqrt(y)
        self.m = operator.out_size
        self.n = operator.in_size
        gain = operator.frobenius_norm_squared() / self.n
        if not gain > 0:
            raise ValueError("the operator has zero Frobenius norm")
        self.scale = self.m / gain
        self.root = math.sqrt(self.scale)
        self.y = y * self.scale
        self.psi = psi * self.root
        self.psi_norm = math.sqrt(be.dot(self.psi, self.psi))
        if not self.psi_norm > 0:
            raise ValueError("all measurements are zero: the signal is zero or unobservable")
        # ||x||^2 estimate: E y_k = ||x||^2 in the Gaussian convention.
        self.norm0 = self.psi_norm / math.sqrt(self.m)
        self.tiny = float(np.finfo(be.real_dtype).tiny)
        # Diagonal whitening G^{-1/2}, G = diag(A^H A)/m in the rescaled units (mean ~1).
        self.whiten: Any = None
        gram = operator.gram_diagonal()
        if gram is not None:
            gram = be.asarray(gram, dtype="real") * (self.scale / self.m)
            if be.scalar(xp.min(gram)) > 0:
                self.whiten = 1.0 / xp.sqrt(gram)

    def forward(self, z: Any) -> Any:
        return self.operator.forward(z) * self.root

    def adjoint(self, v: Any) -> Any:
        return self.operator.adjoint(v) * self.root

    def residual_norm(self, u: Any) -> Any:
        """``|| |u| - psi ||`` (device 0-d array on the GPU, float on the CPU)."""
        return _norm(self.backend, self.backend.xp.abs(u) - self.psi)

    def random_vector(self, seed: Any) -> Any:
        rng = self.backend.random(seed)
        g = rng.standard_normal((2, *self.operator.in_shape))
        v = self.backend.asarray(g[0] + 1j * g[1], dtype="complex")
        return v / _norm(self.backend, v)


# --------------------------------------------------------- initializations
def _power(
    model: _Model,
    weights: Any,
    *,
    shift: float = 0.0,
    iterations: int,
    tol: float,
    seed: Any,
    check_every: int = 10,
) -> tuple[Any, float]:
    """Power method for ``M = (1/m) W A^H diag(weights) A W + shift I``.

    ``W = G^{-1/2}`` is the model's diagonal whitening (identity when the
    operator does not report ``diag(A^H A)``). Returns the unit eigenvector
    estimate (whitened coordinates) and its Rayleigh quotient.
    """
    be = model.backend
    wh = model.whiten
    v = model.random_vector(seed)
    inv_m = 1.0 / model.m
    previous: float | None = None
    value = 0.0
    for it in range(1, iterations + 1):
        w = model.adjoint(weights * model.forward(v if wh is None else wh * v)) * inv_m
        if wh is not None:
            w = wh * w
        if shift:
            w = w + shift * v
        if it % check_every == 0 or it == iterations:
            value = be.dot(v, w)
            if previous is not None and abs(value - previous) <= tol * abs(value):
                return w / _norm(be, w), value
            previous = value
        v = w / _norm(be, w)
    return v, value


def _leading_eigenvector(
    model: _Model,
    weights: Any,
    *,
    shift: float = 0.0,
    iterations: int,
    tol: float,
    seed: Any,
) -> Any:
    """Unit-norm leading eigenvector, mapped back from whitened coordinates."""
    v, _ = _power(model, weights, shift=shift, iterations=iterations, tol=tol, seed=seed)
    if model.whiten is not None:
        v = v / model.whiten
        v = v / _norm(model.backend, v)
    return v


def _spectral(model: _Model, iterations: int, tol: float, seed: Any) -> Any:
    v = _leading_eigenvector(model, model.y, iterations=iterations, tol=tol, seed=seed)
    return v * model.norm0


def _truncated(model: _Model, iterations: int, tol: float, seed: Any, alpha_y: float = 3.0) -> Any:
    y = model.y
    weights = y * (y <= alpha_y**2 * model.norm0**2)
    v = _leading_eigenvector(model, weights, iterations=iterations, tol=tol, seed=seed)
    return v * model.norm0


def _top_fraction(model: _Model, fraction: float) -> Any:
    """Boolean mask of the ``ceil(fraction m)`` largest amplitudes."""
    xp = model.backend.xp
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    k = max(1, min(model.m, math.ceil(fraction * model.m)))
    flat = model.psi.ravel()
    threshold = xp.partition(flat, model.m - k)[model.m - k]
    return model.psi >= threshold


def _orthogonal(
    model: _Model, iterations: int, tol: float, seed: Any, fraction: float = 1.0 / 6.0
) -> Any:
    weights = _top_fraction(model, fraction).astype(model.y.dtype)
    v = _leading_eigenvector(model, weights, iterations=iterations, tol=tol, seed=seed)
    return v * model.norm0


def _weighted(
    model: _Model,
    iterations: int,
    tol: float,
    seed: Any,
    fraction: float = 3.0 / 13.0,
    exponent: float = 0.5,
) -> Any:
    weights = _top_fraction(model, fraction) * model.psi**exponent
    v = _leading_eigenvector(model, weights, iterations=iterations, tol=tol, seed=seed)
    return v * model.norm0


def _optimal(model: _Model, iterations: int, tol: float, seed: Any) -> Any:
    delta = model.m / model.n
    if delta <= 1.0:
        raise ValueError(
            f"the optimal-preprocessing initialization needs more measurements than "
            f"unknowns (m/n = {delta:.3g} <= 1); use init='spectral' or 'truncated'"
        )
    root_delta = math.sqrt(delta)
    ybar = model.y / model.norm0**2
    weights = (ybar - 1.0) / (ybar + (root_delta - 1.0))
    # T >= -1/(sqrt(delta) - 1); shift by that times ||A W||^2/m so the matrix is PSD and
    # the power method finds the largest algebraic eigenvalue.
    _, gram_norm = _power(model, 1.0, iterations=100, tol=1e-4, seed=seed)
    shift = 1.05 * gram_norm / (root_delta - 1.0)
    v = _leading_eigenvector(model, weights, shift=shift, iterations=iterations, tol=tol, seed=seed)
    return v * model.norm0


def _random(model: _Model, iterations: int, tol: float, seed: Any) -> Any:
    return model.random_vector(seed) * model.norm0


_INIT_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "spectral": _spectral,
    "truncated": _truncated,
    "orthogonal": _orthogonal,
    "weighted": _weighted,
    "optimal": _optimal,
    "random": _random,
}


def init_spectral(
    operator: LinearOperator,
    measurements: Any,
    *,
    magnitudes: bool = False,
    iterations: int = 100,
    tol: float = 1e-6,
    seed: Any = None,
) -> Any:
    """Spectral initialization (Netrapalli et al. 2013; Candès et al. 2015).

    Leading eigenvector of ``Y = (1/m) A^H diag(y) A`` by the power method,
    scaled to the norm estimate ``||x||^2 ~ mean(y)`` (in the unit-variance
    convention; see the module docstring).

    Parameters
    ----------
    operator:
        The measurement operator ``A``.
    measurements:
        Intensities ``y = |A x|^2`` (or magnitudes ``|A x|`` with
        ``magnitudes=True``), shape ``operator.out_shape``.
    magnitudes:
        Whether ``measurements`` are magnitudes rather than intensities.
    iterations:
        Maximum power iterations.
    tol:
        Relative change of the Rayleigh quotient that stops the power method.
    seed:
        Seed of the random power-method start (host RNG).

    Returns
    -------
    array
        Initial estimate of shape ``operator.in_shape`` on the operator's
        backend.
    """
    model = _Model(operator, measurements, magnitudes)
    return _spectral(model, iterations, tol, seed)


def init_truncated_spectral(
    operator: LinearOperator,
    measurements: Any,
    *,
    magnitudes: bool = False,
    alpha_y: float = 3.0,
    iterations: int = 100,
    tol: float = 1e-6,
    seed: Any = None,
) -> Any:
    """Truncated spectral initialization (Chen & Candès 2017).

    As :func:`init_spectral` but keeping only ``y_k <= alpha_y^2 mean(y)``,
    which removes the heavy tail that biases the plain spectral estimate.
    Parameters are those of :func:`init_spectral` plus ``alpha_y``
    (dimensionless, default 3 as in the paper).
    """
    model = _Model(operator, measurements, magnitudes)
    return _truncated(model, iterations, tol, seed, alpha_y=alpha_y)


def init_orthogonality_promoting(
    operator: LinearOperator,
    measurements: Any,
    *,
    magnitudes: bool = False,
    fraction: float = 1.0 / 6.0,
    iterations: int = 100,
    tol: float = 1e-6,
    seed: Any = None,
) -> Any:
    """Orthogonality-promoting initialization (Wang, Giannakis & Eldar 2018).

    Leading eigenvector of ``sum_{k in S} a_k a_k^*`` over the
    ``ceil(fraction m)`` largest amplitudes (the measurement vectors most
    aligned with ``x``), scaled to ``sqrt(mean(y))``. Parameters are those of
    :func:`init_spectral` plus ``fraction`` (default 1/6, as in TAF).
    """
    model = _Model(operator, measurements, magnitudes)
    return _orthogonal(model, iterations, tol, seed, fraction=fraction)


def init_weighted_correlation(
    operator: LinearOperator,
    measurements: Any,
    *,
    magnitudes: bool = False,
    fraction: float = 3.0 / 13.0,
    exponent: float = 0.5,
    iterations: int = 100,
    tol: float = 1e-6,
    seed: Any = None,
) -> Any:
    """Weighted maximal correlation initialization (Wang, Giannakis, Saad & Chen 2018).

    Leading eigenvector of ``sum_{k in S} psi_k^exponent a_k a_k^*`` over the
    ``ceil(fraction m)`` largest amplitudes ``psi = sqrt(y)``, scaled to
    ``sqrt(mean(y))``. Defaults (``3/13`` and ``1/2``) follow the RAF paper.
    """
    model = _Model(operator, measurements, magnitudes)
    return _weighted(model, iterations, tol, seed, fraction=fraction, exponent=exponent)


def init_optimal_spectral(
    operator: LinearOperator,
    measurements: Any,
    *,
    magnitudes: bool = False,
    iterations: int = 200,
    tol: float = 1e-6,
    seed: Any = None,
) -> Any:
    """Spectral initialization with optimal preprocessing (Luo, Alghamdi & Lu 2019).

    Leading (largest algebraic) eigenvector of ``(1/m) A^H diag(T(y)) A`` with
    ``T(y) = (y - 1) / (y + sqrt(delta) - 1)``, ``y`` normalized to unit mean
    and ``delta = m / n > 1``. ``T`` is negative for small ``y``, so the power
    method runs on the matrix shifted by ``||A||^2 / (m (sqrt(delta) - 1))``
    (``||A||`` from :meth:`~solvephase.operators.LinearOperator.norm_estimate`),
    which slows it; the default ``iterations`` is therefore larger. Asymptotically
    this weighting minimizes the number of measurements needed for a
    non-trivial correlation with ``x``.
    """
    model = _Model(operator, measurements, magnitudes)
    return _optimal(model, iterations, tol, seed)


def init_random(
    operator: LinearOperator,
    measurements: Any,
    *,
    magnitudes: bool = False,
    seed: Any = None,
) -> Any:
    """Random complex Gaussian start scaled to the norm estimate ``sqrt(mean(y))``."""
    model = _Model(operator, measurements, magnitudes)
    return _random(model, 0, 0.0, seed)


# --------------------------------------------------------------- iterations
def _wf_step(model: _Model, opts: dict[str, float], norm0_sq: float) -> Callable[..., Any]:
    mu_max, tau0 = opts["step"], opts["tau0"]
    scale = 1.0 / (model.m * norm0_sq)

    def step(it: int, z: Any, u: Any) -> Any:
        mu = min(1.0 - math.exp(-it / tau0), mu_max)
        return z - (mu * scale) * model.adjoint((_abs2(u) - model.y) * u)

    return step


def _twf_step(model: _Model, opts: dict[str, float]) -> Callable[..., Any]:
    xp = model.backend.xp
    mu, lb, ub, ah = opts["step"], opts["alpha_lb"], opts["alpha_ub"], opts["alpha_h"]
    factor = 2.0 * mu / model.m

    def step(it: int, z: Any, u: Any) -> Any:
        au2 = _abs2(u)
        au = xp.sqrt(au2)
        r = model.y - au2
        ar = xp.abs(r)
        ratio = au / _norm(model.backend, z)
        keep = (ratio >= lb) & (ratio <= ub) & (ar <= (ah * xp.mean(ar)) * ratio)
        g = xp.where(keep, r / xp.maximum(au2, model.tiny), 0) * u
        return z + factor * model.adjoint(g)

    return step


def _amplitude_residual(model: _Model, u: Any) -> tuple[Any, Any]:
    """``(|u|, u - psi u / |u|)``."""
    xp = model.backend.xp
    au = xp.abs(u)
    return au, u - (model.psi / xp.maximum(au, model.tiny)) * u


def _taf_step(model: _Model, opts: dict[str, float]) -> Callable[..., Any]:
    xp = model.backend.xp
    factor = opts["step"] / model.m
    gamma1 = 1.0 + opts["gamma"]

    def step(it: int, z: Any, u: Any) -> Any:
        au, v = _amplitude_residual(model, u)
        return z - factor * model.adjoint(xp.where(au * gamma1 >= model.psi, v, 0))

    return step


def _raf_step(model: _Model, opts: dict[str, float]) -> Callable[..., Any]:
    xp = model.backend.xp
    factor = opts["step"] / model.m
    beta = opts["beta"]

    def step(it: int, z: Any, u: Any) -> Any:
        au, v = _amplitude_residual(model, u)
        w = au / xp.maximum(au + beta * model.psi, model.tiny)
        return z - factor * model.adjoint(w * v)

    return step


class _Monitor:
    """Shared stopping rules, history and callback handling.

    Two tests run at every check: the relative amplitude residual ``r`` below
    ``tol`` (an exact fit), and its relative change since the previous check
    below ``tol`` (a stationary point: near a minimizer with residual ``r*``
    the change of ``r`` is quadratic in the distance to it, so this stops only
    once the remaining optimization error is far below the noise-induced one,
    and never during geometric convergence to an exact fit).
    """

    def __init__(
        self, model: _Model, tol: float, callback: Callback | None, t0: float, u0: Any
    ) -> None:
        self.model = model
        self.tol = max(tol, 8.0 * model.backend.eps)
        self.callback = callback
        self.t0 = t0
        self.history = [float(model.residual_norm(u0)) / model.psi_norm]
        self.times = [time.perf_counter() - t0]
        self.message = "iteration limit reached"
        self.converged = False

    def check(self, it: int, z: Any, residual_norm: Any, *, final: bool = False) -> bool:
        """Record a check; return True to stop.

        ``final`` records the end state of a run that stopped by itself and
        applies only the exact-fit test.
        """
        r = float(residual_norm) / self.model.psi_norm
        previous = self.history[-1]
        self.history.append(r)
        self.times.append(time.perf_counter() - self.t0)
        if not math.isfinite(r):
            self.message = "diverged (non-finite residual); reduce the step size"
            return True
        if not final and self.callback is not None and self.callback(it, z, r):
            self.message, self.converged = "stopped by callback", True
            return True
        if r <= self.tol:
            self.message, self.converged = "relative residual below tol", True
            return True
        if not final and abs(previous - r) <= self.tol * previous:
            self.message, self.converged = "residual stagnated (stationary point)", True
            return True
        return False


def _iterate(
    model: _Model,
    z: Any,
    step: Callable[..., Any],
    monitor: _Monitor,
    iterations: int,
    check_every: int,
) -> tuple[Any, int]:
    u = model.forward(z)
    it = 0
    for it in range(1, iterations + 1):
        z = step(it, z, u)
        u = model.forward(z)
        if (it % check_every == 0 or it == iterations) and monitor.check(
            it, z, model.residual_norm(u)
        ):
            break
    return z, it


def _run_lbfgs(
    model: _Model,
    z0: Any,
    monitor: _Monitor,
    iterations: int,
    check_every: int,
    loss: str,
    memory: int,
) -> tuple[Any, int]:
    be = model.backend
    xp = be.xp
    cdt, rdt = be.complex_dtype, be.real_dtype
    inv_m = 1.0 / model.m
    amplitude = loss == "amplitude"

    def fun(xr: Any) -> tuple[float, Any]:
        z = xr.view(cdt)
        u = model.forward(z)
        if amplitude:
            au, v = _amplitude_residual(model, u)
            d = au - model.psi
            f = (0.5 * inv_m) * be.dot(d, d)
            g = model.adjoint(v) * inv_m
        else:
            d = _abs2(u) - model.y
            f = (0.25 * inv_m) * be.dot(d, d)
            g = model.adjoint(d * u) * inv_m
        return f, xp.ascontiguousarray(g).view(rdt)

    def check(it: int, xr: Any, f: float) -> bool:
        if it % check_every:
            return False
        z = xr.view(cdt)
        if amplitude:
            res = math.sqrt(max(2.0 * model.m * f, 0.0))
        else:
            res = model.residual_norm(model.forward(z))
        return monitor.check(it, z, res)

    x0 = xp.ascontiguousarray(z0).view(rdt)
    out = lbfgs(
        fun,
        x0,
        be,
        max_iter=iterations,
        memory=memory,
        ftol=be.eps,
        gtol=0.0,
        callback=check,
    )
    z = out.x.view(cdt)
    if out.message != "stopped by callback":
        # The optimizer stopped by itself; record the final state.
        monitor.check(out.n_iter, z, model.residual_norm(model.forward(z)), final=True)
        if out.message != "iteration limit reached" and not monitor.converged:
            monitor.message, monitor.converged = out.message, out.converged
    return z, out.n_iter


def _options(method: str, options: dict[str, float] | None, step: float | None) -> dict[str, float]:
    opts = dict(DEFAULT_OPTIONS[method])
    for key, value in (options or {}).items():
        if key not in opts:
            raise ValueError(
                f"unknown option {key!r} for method {method!r}; valid options: {sorted(opts)}"
            )
        opts[key] = float(value)
    if step is not None:
        if method == "lbfgs":
            raise ValueError("method='lbfgs' chooses its steps by line search; omit step")
        opts["step"] = float(step)
    return opts


def wirtinger(
    operator: LinearOperator,
    measurements: Any,
    *,
    method: str = "raf",
    init: Any = "auto",
    magnitudes: bool = False,
    iterations: int = 1000,
    tol: float = 1e-7,
    check_every: int = 10,
    step: float | None = None,
    loss: str = "amplitude",
    options: dict[str, float] | None = None,
    power_iterations: int | None = None,
    seed: Any = None,
    callback: Callback | None = None,
) -> GenericResult:
    """Recover ``x`` from ``y = |A x|^2`` with a Wirtinger-flow-family solver.

    Parameters
    ----------
    operator:
        Measurement operator ``A`` (:mod:`solvephase.operators`); its backend
        sets the device and precision.
    measurements:
        Intensities ``y = |A x|^2``, shape ``operator.out_shape`` (host or
        device array). Negative values (noise) are clipped to zero.
    method:
        ``"raf"`` (Reweighted Amplitude Flow, default), ``"taf"`` (Truncated
        Amplitude Flow), ``"twf"`` (Truncated Wirtinger Flow), ``"wf"``
        (Wirtinger Flow) or ``"lbfgs"`` (L-BFGS on the ``loss``).
    init:
        ``"auto"`` (the method's own initialization, see
        :data:`DEFAULT_INIT`), ``"spectral"``, ``"truncated"``,
        ``"orthogonal"``, ``"weighted"``, ``"optimal"``, ``"random"``, or an
        array of shape ``operator.in_shape`` to start from.
    magnitudes:
        Whether ``measurements`` are magnitudes ``|A x|`` instead of
        intensities.
    iterations:
        Maximum iterations (each costs one ``forward`` and one ``adjoint``;
        an L-BFGS iteration may cost more through its line search).
    tol:
        Stop when the relative amplitude residual falls below ``tol`` (exact
        fit), or changes by less than ``tol`` (relative) between checks
        (stationary point, e.g. with noisy data). It is raised to ``8 eps``
        of the working precision (about ``1e-6`` in single precision).
    check_every:
        Iterations between convergence checks (the only host syncs).
    step:
        Step size ``mu`` overriding the method default (``mu_max`` for WF).
    loss:
        ``"amplitude"`` or ``"intensity"`` least squares, for
        ``method="lbfgs"``.
    options:
        Method parameters overriding :data:`DEFAULT_OPTIONS`: ``tau0`` (WF);
        ``alpha_lb``, ``alpha_ub``, ``alpha_h`` (TWF); ``gamma`` (TAF);
        ``beta`` (RAF); ``memory`` (L-BFGS).
    power_iterations:
        Power-method iterations of the initialization (default 100, 200 for
        ``"optimal"``).
    seed:
        Seed for the initialization (host RNG; CPU and GPU agree).
    callback:
        ``callback(iteration, x, residual)`` at each check; returning
        ``True`` stops the run.

    Returns
    -------
    GenericResult
        The estimate (global phase arbitrary; compare with
        :func:`relative_error`) and the residual history.
    """
    t0 = time.perf_counter()
    method = str(method).lower()
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if loss not in ("amplitude", "intensity"):
        raise ValueError(f"loss must be 'amplitude' or 'intensity', got {loss!r}")
    if iterations < 0 or check_every < 1:
        raise ValueError("iterations must be >= 0 and check_every >= 1")
    opts = _options(method, options, step)
    model = _Model(operator, measurements, magnitudes)
    be = model.backend
    xp = be.xp

    if isinstance(init, str):
        name = DEFAULT_INIT[method] if init == "auto" else init
        if name not in _INIT_FUNCTIONS:
            raise ValueError(f"init must be 'auto', one of {INITS}, or an array; got {init!r}")
        n_power = power_iterations or (200 if name == "optimal" else 100)
        z = _INIT_FUNCTIONS[name](model, n_power, 1e-6, seed)
    else:
        name = "given"
        z = be.asarray(init, dtype="complex")
        if tuple(z.shape) != tuple(operator.in_shape):
            raise ValueError(
                f"init has shape {tuple(z.shape)} but the operator expects "
                f"{tuple(operator.in_shape)}"
            )
    z = xp.ascontiguousarray(z)

    monitor = _Monitor(model, tol, callback, t0, model.forward(z))
    if method == "lbfgs":
        z, n_iter = _run_lbfgs(
            model, z, monitor, iterations, check_every, loss, int(opts["memory"])
        )
    else:
        if method == "wf":
            norm0_sq = be.dot(z, z) or model.norm0**2
            stepper = _wf_step(model, opts, norm0_sq)
        elif method == "twf":
            stepper = _twf_step(model, opts)
        elif method == "taf":
            stepper = _taf_step(model, opts)
        else:
            stepper = _raf_step(model, opts)
        z, n_iter = _iterate(model, z, stepper, monitor, iterations, check_every)
    be.synchronize()
    return GenericResult(
        x=z,
        history=monitor.history,
        times=monitor.times,
        n_iter=n_iter,
        converged=monitor.converged,
        message=monitor.message,
        elapsed=time.perf_counter() - t0,
        device=be.device,
        method=method,
        init=name,
        options=opts,
    )
