"""Array backends: NumPy/SciPy on the CPU and CuPy on CUDA GPUs.

Every numerical routine in solvephase takes its array namespace from a
:class:`Backend`, so the same algorithm runs on either device. A backend fixes
three things:

* ``device`` — ``"cpu"`` (NumPy arrays, :mod:`scipy.fft` with all cores) or
  ``"gpu"`` (CuPy arrays, cuFFT/cuBLAS);
* ``precision`` — ``"double"`` (float64/complex128) or ``"single"``
  (float32/complex64). CPU defaults to double, GPU to single;
* the unitary FFT used by every propagator.

Arrays returned by a GPU solver stay on the GPU. :func:`to_numpy` is the one
explicit host boundary.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import Any, Literal, Union

import numpy as np

DeviceName = Literal["cpu", "gpu", "auto"]
PrecisionName = Literal["single", "double"]

__all__ = [
    "Backend",
    "BackendLike",
    "backend_of",
    "get_backend",
    "gpu_available",
    "to_numpy",
]


@functools.lru_cache(maxsize=1)
def _cupy() -> Any:
    try:
        import cupy
    except ImportError:  # pragma: no cover - depends on the environment
        return None
    return cupy


@functools.lru_cache(maxsize=1)
def gpu_available() -> bool:
    """Return whether CuPy is importable and sees at least one CUDA device."""
    cupy = _cupy()
    if cupy is None:  # pragma: no cover - depends on the environment
        return False
    try:
        return bool(cupy.cuda.runtime.getDeviceCount() > 0)
    except Exception:  # pragma: no cover - driver/runtime missing
        return False


def _cpu_workers(size: int = 1 << 30) -> int:
    """SciPy FFT threads for a transform of ``size`` total elements.

    One thread per ~16k elements, capped at the core count: small transforms
    lose more to thread start-up and contention than they gain (a 48x48 FFT
    is ~10x slower on 16 threads than on one when the cores are busy).
    ``SOLVEPHASE_FFT_WORKERS`` fixes the count.
    """
    env = os.environ.get("SOLVEPHASE_FFT_WORKERS")
    if env:
        return max(1, int(env))
    return max(1, min(os.cpu_count() or 1, size >> 14))


@dataclass(frozen=True)
class Backend:
    """One array namespace, device and floating-point precision.

    Use :func:`get_backend` rather than constructing this directly.

    Attributes
    ----------
    device:
        ``"cpu"`` or ``"gpu"``.
    precision:
        ``"double"`` or ``"single"``.
    """

    device: str
    precision: str

    # ------------------------------------------------------------------ basics
    @property
    def xp(self) -> Any:
        """The array namespace: :mod:`numpy` or :mod:`cupy`."""
        return _cupy() if self.device == "gpu" else np

    @property
    def is_gpu(self) -> bool:
        """Whether arrays live on a CUDA device."""
        return self.device == "gpu"

    @property
    def real_dtype(self) -> np.dtype[Any]:
        """float64 for double precision, float32 for single."""
        return np.dtype(np.float64 if self.precision == "double" else np.float32)

    @property
    def complex_dtype(self) -> np.dtype[Any]:
        """complex128 for double precision, complex64 for single."""
        return np.dtype(np.complex128 if self.precision == "double" else np.complex64)

    @property
    def eps(self) -> float:
        """Machine epsilon of :attr:`real_dtype`."""
        return float(np.finfo(self.real_dtype).eps)

    def __repr__(self) -> str:
        return f"Backend(device={self.device!r}, precision={self.precision!r})"

    # ---------------------------------------------------------------- creation
    def asarray(self, value: Any, dtype: Any = None) -> Any:
        """Move ``value`` onto this backend.

        ``dtype`` may be a NumPy dtype, ``"real"`` or ``"complex"`` (this
        backend's working precision), or ``None`` to promote real input to
        :attr:`real_dtype` and complex input to :attr:`complex_dtype`.
        Integer and boolean arrays keep their type when ``dtype`` is None.
        """
        if dtype == "real":
            dtype = self.real_dtype
        elif dtype == "complex":
            dtype = self.complex_dtype
        array = self.xp.asarray(value) if self.is_gpu else np.asarray(to_numpy(value))
        if dtype is None:
            kind = array.dtype.kind
            if kind == "c":
                dtype = self.complex_dtype
            elif kind == "f":
                dtype = self.real_dtype
            else:
                return array
        return array.astype(dtype, copy=False)

    def zeros(self, shape: Any, dtype: Any = "real") -> Any:
        """Zero-filled array in working precision (``"real"``/``"complex"``)."""
        return self.xp.zeros(shape, dtype=self._dtype(dtype))

    def empty(self, shape: Any, dtype: Any = "real") -> Any:
        """Uninitialized array in working precision (``"real"``/``"complex"``)."""
        return self.xp.empty(shape, dtype=self._dtype(dtype))

    def _dtype(self, dtype: Any) -> Any:
        if dtype == "real":
            return self.real_dtype
        if dtype == "complex":
            return self.complex_dtype
        return dtype

    # -------------------------------------------------------------------- FFTs
    def fft2(self, array: Any, *, axes: tuple[int, int] = (-2, -1)) -> Any:
        """Unitary (``norm="ortho"``) two-dimensional FFT, uncentred."""
        if self.is_gpu:
            return self.xp.fft.fft2(array, axes=axes, norm="ortho")
        from scipy import fft

        return fft.fft2(array, axes=axes, norm="ortho", workers=_cpu_workers(array.size))

    def ifft2(self, array: Any, *, axes: tuple[int, int] = (-2, -1)) -> Any:
        """Unitary inverse of :meth:`fft2` (also its adjoint)."""
        if self.is_gpu:
            return self.xp.fft.ifft2(array, axes=axes, norm="ortho")
        from scipy import fft

        return fft.ifft2(array, axes=axes, norm="ortho", workers=_cpu_workers(array.size))

    def rfft2(self, array: Any, *, axes: tuple[int, int] = (-2, -1)) -> Any:
        """Unnormalized real-input 2-D FFT (used for convolutions)."""
        if self.is_gpu:
            return self.xp.fft.rfft2(array, axes=axes)
        from scipy import fft

        return fft.rfft2(array, axes=axes, workers=_cpu_workers(array.size))

    def irfft2(
        self, array: Any, shape: tuple[int, int], *, axes: tuple[int, int] = (-2, -1)
    ) -> Any:
        """Inverse of :meth:`rfft2` for an output of ``shape``."""
        if self.is_gpu:
            return self.xp.fft.irfft2(array, s=shape, axes=axes)
        from scipy import fft

        return fft.irfft2(array, s=shape, axes=axes, workers=_cpu_workers(array.size))

    # -------------------------------------------------------------- reductions
    def dot(self, a: Any, b: Any) -> float:
        """``Re(sum(conj(a) * b))`` over all elements, as a Python float.

        On the CPU this avoids BLAS level-1 calls: OpenBLAS threads ``ddot``
        above ~10k elements, and those threads stall (~1 ms per call) whenever
        the cores are busy. Use it for every scalar product in iterative loops.
        """
        a, b = a.reshape(-1), b.reshape(-1)
        if self.is_gpu:
            return float(self.xp.vdot(a, b).real)
        if a.dtype.kind == "c" or b.dtype.kind == "c":
            return float(np.einsum("i,i->", np.conj(a), b).real)
        return float(np.einsum("i,i->", a, b))

    # ---------------------------------------------------------- host boundary
    def to_numpy(self, array: Any) -> Any:
        """Copy an array to host NumPy (no-op on the CPU)."""
        return to_numpy(array)

    def scalar(self, value: Any) -> float:
        """Convert a 0-d array to a Python float (a device sync on GPU)."""
        return float(value)

    def synchronize(self) -> None:
        """Block until queued device work finishes (no-op on the CPU)."""
        if self.is_gpu:
            self.xp.cuda.get_current_stream().synchronize()

    def random(self, seed: Any) -> np.random.Generator:
        """A host :class:`numpy.random.Generator`.

        Random draws (initial guesses, noise) are made on the host and moved
        to the device so a seed gives the same numbers on CPU and GPU.
        """
        if isinstance(seed, np.random.Generator):
            return seed
        return np.random.default_rng(seed)


BackendLike = Union[Backend, str, None]  # noqa: UP007 (runtime alias on Python 3.10)


@functools.cache
def _make_backend(device: str, precision: str) -> Backend:
    return Backend(device=device, precision=precision)


def get_backend(device: BackendLike = "cpu", precision: str | None = None) -> Backend:
    """Return the :class:`Backend` for a device and precision.

    Parameters
    ----------
    device:
        ``"cpu"``, ``"gpu"`` (CuPy; raises if unavailable), ``"auto"`` (GPU when
        available, else CPU), or an existing :class:`Backend`, which is returned
        unchanged unless ``precision`` overrides it.
    precision:
        ``"double"`` or ``"single"``. Defaults to double on CPU, single on GPU.
    """
    if isinstance(device, Backend):
        if precision is None or precision == device.precision:
            return device
        return _make_backend(device.device, _check_precision(precision))
    name = "cpu" if device is None else str(device).lower()
    if name == "cuda":
        name = "gpu"
    if name == "auto":
        name = "gpu" if gpu_available() else "cpu"
    if name not in ("cpu", "gpu"):
        raise ValueError(f"device must be 'cpu', 'gpu' or 'auto', got {device!r}")
    if name == "gpu" and not gpu_available():
        raise RuntimeError(
            "device='gpu' needs CuPy and a CUDA device. Install the extra matching your "
            "driver (pip install 'solvephase[cuda12]' or 'solvephase[cuda13]'), or use "
            "device='auto' to fall back to the CPU."
        )
    if precision is None:
        precision = "single" if name == "gpu" else "double"
    return _make_backend(name, _check_precision(precision))


def _check_precision(precision: str) -> str:
    value = str(precision).lower()
    aliases = {"float32": "single", "float64": "double", "fp32": "single", "fp64": "double"}
    value = aliases.get(value, value)
    if value not in ("single", "double"):
        raise ValueError(f"precision must be 'single' or 'double', got {precision!r}")
    return value


def to_numpy(array: Any) -> Any:
    """Return ``array`` as a host NumPy array (copying from the GPU if needed).

    Non-array values (lists, scalars) go through :func:`numpy.asarray`.
    """
    cupy = _cupy()
    if cupy is not None and isinstance(array, cupy.ndarray):
        return cupy.asnumpy(array)
    return np.asarray(array)


def backend_of(array: Any, precision: str | None = None) -> Backend:
    """Infer a backend from an array (CuPy arrays give the GPU backend).

    Without ``precision``, single precision is chosen for float32/complex64
    input and double otherwise.
    """
    cupy = _cupy()
    device = "gpu" if cupy is not None and isinstance(array, cupy.ndarray) else "cpu"
    if precision is None:
        dtype = getattr(array, "dtype", None)
        single = dtype is not None and np.dtype(dtype) in (np.float32, np.complex64)
        precision = "single" if single else "double"
    return get_backend(device, precision)
