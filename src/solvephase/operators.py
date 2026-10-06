"""Generic linear measurement operators with exact adjoints.

The generic phase-retrieval solvers (:mod:`solvephase.algorithms.wirtinger`)
recover ``x`` from ``y = |A x|^2`` for any linear ``A``. This module supplies
the operators:

* :class:`MatrixOperator` - a dense complex matrix, with
  :meth:`MatrixOperator.gaussian` for the i.i.d. complex Gaussian model used in
  the Wirtinger-flow literature (Candès, Li & Soltanolkotabi, IEEE Trans. Inf.
  Theory 61, 1985, 2015);
* :class:`CodedDiffractionOperator` - coded diffraction patterns (CDP): ``L``
  random modulation masks followed by a unitary FFT (Candès, Li &
  Soltanolkotabi, Appl. Comput. Harmon. Anal. 39, 277, 2015);
* :class:`OversampledFourierOperator` - zero padding followed by a unitary FFT
  (the classical Fourier phase-retrieval model).

Every operator maps ``(..., *in_shape)`` to ``(..., *out_shape)`` (leading
batch axes are allowed), works in the precision of its :class:`Backend`, and
ships :meth:`~LinearOperator.adjoint`, its exact conjugate transpose.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .backend import Backend, BackendLike, get_backend

__all__ = [
    "CodedDiffractionOperator",
    "LinearOperator",
    "MatrixOperator",
    "OversampledFourierOperator",
]


def _norm(backend: Backend, a: Any) -> Any:
    """``||a||_2`` without a host sync on the GPU (device 0-d array there).

    On the CPU it is a Python float from :meth:`Backend.dot`, which avoids the
    threaded BLAS level-1 calls that stall when the cores are busy.
    """
    if backend.is_gpu:
        return backend.xp.linalg.norm(a)
    return math.sqrt(backend.dot(a, a))


def _shape(value: Any, name: str, ndim: int | None = None) -> tuple[int, ...]:
    if np.ndim(value) == 0:
        value = (value,) * (ndim or 1)
    items = tuple(int(v) for v in value)
    if not items or min(items) < 1 or (ndim is not None and len(items) != ndim):
        expected = f"{ndim} positive ints" if ndim else "positive ints"
        raise ValueError(f"{name} must be {expected}, got {value!r}")
    return items


class LinearOperator(ABC):
    """A linear map ``A`` between complex arrays, with its exact adjoint.

    Subclasses set :attr:`backend`, :attr:`in_shape` and :attr:`out_shape` and
    implement :meth:`forward` and :meth:`adjoint`.
    """

    backend: Backend
    in_shape: tuple[int, ...]
    out_shape: tuple[int, ...]

    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Apply ``A``: ``(..., *in_shape)`` to ``(..., *out_shape)``."""

    @abstractmethod
    def adjoint(self, y: Any) -> Any:
        """Apply ``A^H``: ``(..., *out_shape)`` to ``(..., *in_shape)``."""

    def __call__(self, x: Any) -> Any:
        return self.forward(x)

    @property
    def in_size(self) -> int:
        """Number of unknowns ``n`` (complex entries of the input)."""
        return math.prod(self.in_shape)

    @property
    def out_size(self) -> int:
        """Number of measurements ``m`` (entries of the output)."""
        return math.prod(self.out_shape)

    def frobenius_norm_squared(self, *, probes: int = 32, seed: Any = 0) -> float:
        """``trace(A^H A) = ||A||_F^2``.

        ``||A||_F^2 / n`` is the mean energy gain ``E ||A x||^2 / ||x||^2`` for
        a random ``x``; the solvers use it to put every operator on the scale of
        the unit-variance Gaussian model. The base implementation is a
        Hutchinson estimate from ``probes`` random phase vectors (relative
        error about ``1 / sqrt(probes)`` at worst); concrete operators override
        it with the exact value.
        """
        be = self.backend
        rng = be.random(seed)
        phases = rng.uniform(0.0, 2.0 * np.pi, size=(probes, *self.in_shape))
        z = be.asarray(np.exp(1j * phases), dtype="complex")
        az = self.forward(z)
        return be.dot(az, az) / probes

    def gram_diagonal(self) -> Any:
        """``diag(A^H A)`` as a real array of shape :attr:`in_shape`, or ``None``.

        The squared norm of each column of ``A``. Spectral initializations use
        it to whiten the power method (see
        :mod:`solvephase.algorithms.wirtinger`). The base class returns ``None``
        (unknown); every built-in operator returns the exact value.
        """
        return None

    def norm_estimate(
        self, *, iterations: int = 100, tol: float = 1e-6, seed: Any = 0, check_every: int = 5
    ) -> float:
        """Spectral norm ``||A||_2`` by power iteration on ``A^H A``.

        Parameters
        ----------
        iterations:
            Maximum power iterations.
        tol:
            Stop when the estimate changes by less than ``tol`` (relative)
            between checks.
        seed:
            Seed of the random start (host RNG, so CPU and GPU agree).
        check_every:
            Iterations between scalar convergence checks (device syncs).

        Returns
        -------
        float
            The estimate, which approaches ``||A||_2`` from below.
        """
        be = self.backend
        rng = be.random(seed)
        v0 = rng.standard_normal((2, *self.in_shape))
        v = be.asarray(v0[0] + 1j * v0[1], dtype="complex")
        v = v / _norm(be, v)
        estimate = 0.0
        for it in range(1, iterations + 1):
            w = self.adjoint(self.forward(v))
            norm = _norm(be, w)
            v = w / norm
            if it % check_every == 0 or it == iterations:
                value = math.sqrt(be.scalar(norm))
                if abs(value - estimate) <= tol * value:
                    return value
                estimate = value
        return estimate


# ------------------------------------------------------------------- matrix
#: Below this many matrix entries (times the batch size), CPU products use einsum.
_EINSUM_MAX_ELEMENTS = 1 << 18


class MatrixOperator(LinearOperator):
    """A dense complex matrix ``A`` of shape ``(m, n)``: ``y = A x``.

    Row ``k`` of ``A`` is the conjugated measurement vector ``a_k^*``, so
    ``y_k = |a_k^* x|^2``. ``forward`` accepts ``(..., n)`` and returns
    ``(..., m)``.

    Parameters
    ----------
    matrix:
        ``(m, n)`` array (host or device); stored in the backend's complex
        precision.
    backend:
        Target backend (default CPU, double precision).
    """

    def __init__(self, matrix: Any, *, backend: BackendLike = None) -> None:
        self.backend = get_backend(backend)
        a = self.backend.asarray(matrix, dtype="complex")
        if a.ndim != 2:
            raise ValueError(f"matrix must be two-dimensional (m, n), got shape {a.shape}")
        xp = self.backend.xp
        self.matrix = a
        self._at = xp.ascontiguousarray(a.T)
        self._ac = xp.conj(a)
        self.in_shape = (int(a.shape[1]),)
        self.out_shape = (int(a.shape[0]),)

    @classmethod
    def gaussian(
        cls, m: int, n: int, *, seed: Any = None, backend: BackendLike = None
    ) -> MatrixOperator:
        """i.i.d. circularly-symmetric complex Gaussian matrix, ``E |A_kj|^2 = 1``.

        Entries are ``(g1 + i g2) / sqrt(2)`` with ``g1, g2 ~ N(0, 1)``, drawn on
        the host from ``seed`` (Candès et al. 2015: ``a_k ~ N(0, I/2) + i N(0,
        I/2)``, so ``E y_k = ||x||^2``).
        """
        if m < 1 or n < 1:
            raise ValueError(f"m and n must be positive, got m={m}, n={n}")
        be = get_backend(backend)
        rng = be.random(seed)
        g = rng.standard_normal((2, m, n))
        return cls((g[0] + 1j * g[1]) / math.sqrt(2.0), backend=be)

    def _use_blas(self, v: Any) -> bool:
        # Threaded OpenBLAS gemv/gemm stall for milliseconds when the cores are busy;
        # numpy's own einsum loops are slower per flop but never stall, and win below
        # a few hundred thousand multiply-adds.
        if self.backend.is_gpu:
            return True
        n_vectors = max(v.size // max(v.shape[-1], 1), 1)
        return bool(self.matrix.size * n_vectors > _EINSUM_MAX_ELEMENTS)

    def forward(self, x: Any) -> Any:
        if self._use_blas(x):
            return self.backend.xp.matmul(x, self._at)
        return np.einsum("...j,kj->...k", x, self.matrix)

    def adjoint(self, y: Any) -> Any:
        if self._use_blas(y):
            return self.backend.xp.matmul(y, self._ac)
        return np.einsum("...k,kj->...j", y, self._ac)

    def frobenius_norm_squared(self, *, probes: int = 32, seed: Any = 0) -> float:
        """Exact ``||A||_F^2`` (``probes`` and ``seed`` are ignored)."""
        return self.backend.dot(self.matrix, self.matrix)

    def gram_diagonal(self) -> Any:
        """Squared column norms ``sum_k |A_kj|^2``."""
        a = self.matrix
        return (a.real * a.real + a.imag * a.imag).sum(axis=0)


# ------------------------------------------------------- coded diffraction
class CodedDiffractionOperator(LinearOperator):
    """Coded diffraction patterns: ``y_l = |F(d_l * x)|^2`` for ``l < L``.

    Each of the ``L`` masks ``d_l`` multiplies the ``(ny, nx)`` image before a
    unitary 2-D FFT ``F`` (batched over the masks). ``forward`` maps
    ``(..., ny, nx)`` to ``(..., L, ny, nx)``; the adjoint is
    ``sum_l conj(d_l) * F^H y_l``. Candès, Li & Soltanolkotabi (2015) prove
    recovery from ``L = O(log^4 n)`` octanary masks; ``L = 6-8`` suffices in
    practice.

    Parameters
    ----------
    masks:
        ``(L, ny, nx)`` complex mask array.
    backend:
        Target backend.
    """

    def __init__(self, masks: Any, *, backend: BackendLike = None) -> None:
        self.backend = get_backend(backend)
        d = self.backend.asarray(masks, dtype="complex")
        if d.ndim != 3:
            raise ValueError(f"masks must be (L, ny, nx), got shape {d.shape}")
        self.masks = d
        self._masks_conj = self.backend.xp.conj(d)
        self.n_masks = int(d.shape[0])
        self.in_shape = (int(d.shape[1]), int(d.shape[2]))
        self.out_shape = (self.n_masks, *self.in_shape)

    @classmethod
    def random(
        cls,
        shape: Any,
        n_masks: int,
        *,
        kind: str = "octanary",
        seed: Any = None,
        backend: BackendLike = None,
    ) -> CodedDiffractionOperator:
        """Random masks drawn on the host from ``seed``.

        Parameters
        ----------
        shape:
            Image shape ``(ny, nx)`` (or an int for a square image).
        n_masks:
            Number of masks ``L``.
        kind:
            ``"octanary"`` - ``d = b1 * b2`` with ``b1`` uniform on
            ``{1, -1, i, -i}`` and ``b2 = 1/sqrt(2)`` with probability 4/5,
            ``sqrt(3)`` with probability 1/5 (Candès et al. 2015, eq. 1.9;
            ``E |d|^2 = 1``); ``"uniform"`` - unit-modulus masks with phases
            uniform on ``[0, 2 pi)``.
        """
        ny, nx = _shape(shape, "shape", 2)
        if n_masks < 1:
            raise ValueError(f"n_masks must be positive, got {n_masks}")
        rng = get_backend(backend).random(seed)
        size = (n_masks, ny, nx)
        if kind == "octanary":
            b1 = np.array([1, -1, 1j, -1j])[rng.integers(0, 4, size=size)]
            b2 = np.where(rng.random(size) < 0.8, 1.0 / math.sqrt(2.0), math.sqrt(3.0))
            masks = b1 * b2
        elif kind == "uniform":
            masks = np.exp(2j * np.pi * rng.random(size))
        else:
            raise ValueError(f"kind must be 'octanary' or 'uniform', got {kind!r}")
        return cls(masks, backend=backend)

    def forward(self, x: Any) -> Any:
        return self.backend.fft2(x[..., None, :, :] * self.masks)

    def adjoint(self, y: Any) -> Any:
        return (self.backend.ifft2(y) * self._masks_conj).sum(axis=-3)

    def frobenius_norm_squared(self, *, probes: int = 32, seed: Any = 0) -> float:
        """Exact ``||A||_F^2 = sum |d|^2`` (``probes`` and ``seed`` are ignored)."""
        return self.backend.dot(self.masks, self.masks)

    def gram_diagonal(self) -> Any:
        """``sum_l |d_l|^2``: ``A^H A`` is diagonal for a unitary FFT."""
        d = self.masks
        return (d.real * d.real + d.imag * d.imag).sum(axis=0)


# ------------------------------------------------------ oversampled Fourier
class OversampledFourierOperator(LinearOperator):
    """Zero padding followed by a unitary 2-D FFT.

    The ``in_shape`` image is placed in the corner of an ``out_shape`` grid and
    transformed, so ``A^H A = I`` (an isometry) and ``A^H`` crops the inverse
    FFT. ``forward`` maps ``(..., *in_shape)`` to ``(..., *out_shape)``.

    Fourier magnitudes alone do not determine an image up to a global phase:
    shifts and the conjugate flip ``x(-r)*`` give the same data. The
    Wirtinger-flow solvers therefore rarely succeed with this operator by
    itself; use it with support/positivity solvers or add masks
    (:class:`CodedDiffractionOperator`).

    Parameters
    ----------
    in_shape:
        Image shape ``(ny, nx)``.
    oversampling:
        Output size per input size (scalar or ``(y, x)`` pair, ``>= 1``);
        ignored when ``out_shape`` is given.
    out_shape:
        Explicit FFT grid shape, at least ``in_shape``.
    backend:
        Target backend.
    """

    def __init__(
        self,
        in_shape: Any,
        oversampling: Any = 2.0,
        *,
        out_shape: Any = None,
        backend: BackendLike = None,
    ) -> None:
        self.backend = get_backend(backend)
        self.in_shape = _shape(in_shape, "in_shape", 2)
        if out_shape is None:
            factors = np.broadcast_to(np.asarray(oversampling, dtype=np.float64), (2,))
            if np.any(factors < 1.0):
                raise ValueError(f"oversampling must be >= 1, got {oversampling!r}")
            out_shape = tuple(
                math.ceil(n * float(f) - 1e-9) for n, f in zip(self.in_shape, factors)
            )
        self.out_shape = _shape(out_shape, "out_shape", 2)
        if any(o < i for o, i in zip(self.out_shape, self.in_shape)):
            raise ValueError(
                f"out_shape {self.out_shape} must be at least in_shape {self.in_shape}"
            )

    def forward(self, x: Any) -> Any:
        ny, nx = self.in_shape
        padded = self.backend.zeros((*x.shape[:-2], *self.out_shape), dtype="complex")
        padded[..., :ny, :nx] = x
        return self.backend.fft2(padded)

    def adjoint(self, y: Any) -> Any:
        ny, nx = self.in_shape
        return self.backend.ifft2(y)[..., :ny, :nx]

    def frobenius_norm_squared(self, *, probes: int = 32, seed: Any = 0) -> float:
        """Exact ``||A||_F^2 = n`` (an isometry; arguments are ignored)."""
        return float(self.in_size)

    def gram_diagonal(self) -> Any:
        """All ones (``A^H A = I``)."""
        return self.backend.xp.ones(self.in_shape, dtype=self.backend.real_dtype)
