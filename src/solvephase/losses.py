"""Data-fidelity terms comparing model images to measured images.

Each loss returns its value, its gradient with respect to the model, and a
non-negative per-pixel *curvature* (the expected Hessian, i.e. Fisher
information) used by Gauss-Newton/Levenberg-Marquardt. All operations run on
the backend of the input arrays; values are 0-d backend arrays so the caller
decides when to synchronize.

``weights`` (same shape as the data, or broadcastable) scales each pixel's
contribution; zero excludes a pixel (bad pixels, saturation, beyond a field
stop).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .backend import Backend

__all__ = ["AmplitudeLoss", "GaussianLoss", "Loss", "PoissonLoss", "make_loss"]


class Loss(ABC):
    """A data-fidelity term ``L(model, data)``."""

    name: str = "loss"

    @abstractmethod
    def evaluate(
        self, backend: Backend, model: Any, data: Any, weights: Any
    ) -> tuple[Any, Any, Any]:
        """Return ``(value, d value / d model, curvature)`` on the backend."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class GaussianLoss(Loss):
    """Weighted least squares ``0.5 * sum w (model - data)^2``.

    With ``weights = 1 / variance`` this is the Gaussian negative
    log-likelihood. :func:`noise_weights` gives a photon + read-noise variance.
    """

    name = "gaussian"

    def evaluate(
        self, backend: Backend, model: Any, data: Any, weights: Any
    ) -> tuple[Any, Any, Any]:
        xp = backend.xp
        resid = model - data
        wr = weights * resid
        value = 0.5 * xp.sum(wr * resid, dtype=np.float64)
        return value, wr, xp.broadcast_to(weights, model.shape)


def _floor(xp: Any, data: Any, floor: float | None) -> Any:
    """Model floor below which losses are extended quadratically."""
    if floor is not None:
        return floor
    return 1e-6 * xp.max(xp.abs(data)) + 1e-30


class PoissonLoss(Loss):
    """Poisson negative log-likelihood (Paxman, Schulz & Fienup 1992).

    Evaluated as the deviance ``sum w [ (m' - d') + d' log(d' / m') ]`` with
    ``m' = m + s``, ``d' = d + s`` and ``s = read_noise^2`` (in the data's
    units squared), the shifted-Poisson approximation that also covers
    Gaussian read noise. The deviance differs from the negative
    log-likelihood by a data-only constant, is non-negative, and is about
    ``chi^2 / 2`` at a good fit.

    Below a small positive floor ``m0`` (default ``1e-6 max|d|``) the loss
    continues as its second-order Taylor expansion about ``m0``, so value and
    gradient stay finite and consistent when a line search probes negative
    model values.

    Parameters
    ----------
    read_noise:
        Read-noise standard deviation in data units (e.g. photo-electrons).
    floor:
        Explicit model floor ``m0`` in data units.
    """

    name = "poisson"

    def __init__(self, read_noise: float = 0.0, floor: float | None = None) -> None:
        if read_noise < 0 or (floor is not None and floor <= 0):
            raise ValueError("read_noise must be >= 0 and floor > 0")
        self.read_noise = float(read_noise)
        self.floor = floor

    def evaluate(
        self, backend: Backend, model: Any, data: Any, weights: Any
    ) -> tuple[Any, Any, Any]:
        xp = backend.xp
        shift = self.read_noise**2
        d = xp.maximum(data + shift, 0.0)
        t = model + shift
        m0 = _floor(xp, d, self.floor)
        m = xp.maximum(t, m0)
        # d log(d / m) rather than d log d - d log m: the two large logarithms
        # cancel catastrophically in single precision at high photon counts.
        ratio = xp.where(d > 0, d / m, 1.0)
        f = (m - d) + d * xp.log(ratio)
        g = 1.0 - d / m
        below = t < m0
        h2 = d / (m * m)
        delta = xp.where(below, t - m0, 0.0)
        value = xp.sum(weights * (f + delta * (g + 0.5 * h2 * delta)), dtype=np.float64)
        grad = weights * (g + h2 * delta)
        return value, grad, weights / m

    def __repr__(self) -> str:
        return f"PoissonLoss(read_noise={self.read_noise:g})"


class AmplitudeLoss(Loss):
    """Least squares on square-root intensities, ``0.5 * sum w (sqrt m - sqrt d)^2``.

    The amplitude metric of Fienup's iterative transform algorithms; it
    variance-stabilizes photon noise and converges fast from poor starts.
    Below a small positive floor (default ``1e-6 max|d|``) it continues as its
    second-order Taylor expansion, like :class:`PoissonLoss`.
    """

    name = "amplitude"

    def __init__(self, floor: float | None = None) -> None:
        if floor is not None and floor <= 0:
            raise ValueError("floor must be > 0")
        self.floor = floor

    def evaluate(
        self, backend: Backend, model: Any, data: Any, weights: Any
    ) -> tuple[Any, Any, Any]:
        xp = backend.xp
        d = xp.maximum(data, 0.0)
        m0 = _floor(xp, d, self.floor)
        m = xp.maximum(model, m0)
        sm = xp.sqrt(m)
        sd = xp.sqrt(d)
        resid = sm - sd
        f = 0.5 * resid * resid
        g = resid / (2.0 * sm)
        h2 = sd / (4.0 * m * sm)
        delta = xp.where(model < m0, model - m0, 0.0)
        value = xp.sum(weights * (f + delta * (g + 0.5 * h2 * delta)), dtype=np.float64)
        grad = weights * (g + h2 * delta)
        return value, grad, weights / (4.0 * m)


def make_loss(loss: str | Loss, **kwargs: Any) -> Loss:
    """Return a :class:`Loss` from a name (``"gaussian"``, ``"poisson"``, ``"amplitude"``)."""
    if isinstance(loss, Loss):
        return loss
    name = str(loss).lower()
    if name in ("gaussian", "ls", "least_squares", "l2"):
        return GaussianLoss()
    if name in ("poisson", "ml", "poisson_ml"):
        return PoissonLoss(**kwargs)
    if name in ("amplitude", "sqrt"):
        return AmplitudeLoss(**kwargs)
    raise ValueError(f"unknown loss {loss!r}; use 'gaussian', 'poisson' or 'amplitude'")
