"""Linear optical propagators with exact adjoints.

All propagators map a complex field of shape ``(..., ny, nx)`` to
``(..., my, mx)`` and provide :meth:`~Propagator.adjoint`, the exact conjugate
transpose of :meth:`~Propagator.forward`. The adjoint is what makes analytic
gradients (and therefore fast optimization) possible.

Conventions
-----------
* Arrays are ``(y, x)``. Coordinates sit on pixel centres and are centred:
  pixel ``i`` of ``n`` is at ``(i - (n - 1) / 2) * pitch``.
* Fraunhofer propagation uses the kernel ``exp(-2 pi i x . alpha / lambda)``,
  so a pupil OPD ramp ``OPD = a * x`` moves the image to angle ``+a`` (towards
  increasing column index). See :doc:`conventions`.
* Normalization is unitary: when the output window captures all the light,
  ``sum |E|^2 == sum |u|^2`` (Parseval). A finite window loses light; it is
  never renormalized.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np

from .backend import Backend, BackendLike, get_backend

__all__ = [
    "AngularSpectrumPropagator",
    "FFTPropagator",
    "FocalPlanePropagator",
    "MFTPropagator",
    "Propagator",
    "centered_coordinates",
]


def centered_coordinates(n: int, pitch: float = 1.0, offset: float = 0.0) -> np.ndarray:
    """Pixel-centre coordinates ``(i - (n - 1) / 2 + offset) * pitch`` for ``i < n``."""
    return (np.arange(n, dtype=np.float64) - (n - 1) / 2.0 + offset) * pitch


def _pair(value: Any, name: str) -> tuple[float, float]:
    if np.ndim(value) == 0:
        return float(value), float(value)
    items = tuple(float(v) for v in value)
    if len(items) != 2:
        raise ValueError(f"{name} must be a scalar or a (y, x) pair, got {value!r}")
    return items[0], items[1]


def _shape(value: Any, name: str) -> tuple[int, int]:
    if np.ndim(value) == 0:
        value = (value, value)
    items = tuple(int(v) for v in value)
    if len(items) != 2 or min(items) < 1:
        raise ValueError(f"{name} must be a positive int or (ny, nx) pair, got {value!r}")
    return items[0], items[1]


class Propagator(ABC):
    """A linear map between two sampled planes with an exact adjoint."""

    backend: Backend
    in_shape: tuple[int, int]
    out_shape: tuple[int, int]

    @abstractmethod
    def forward(self, field: Any) -> Any:
        """Propagate ``(..., *in_shape)`` to ``(..., *out_shape)``."""

    @abstractmethod
    def adjoint(self, field: Any) -> Any:
        """Apply the conjugate transpose: ``(..., *out_shape)`` to ``(..., *in_shape)``."""

    def __call__(self, field: Any) -> Any:
        return self.forward(field)


# --------------------------------------------------------------------- FFT
class FFTPropagator(Propagator):
    """Zero-padded FFT between centred grids.

    Equivalent to :class:`MFTPropagator` with ``samples = n_fft`` but costs one
    ``n_fft``-sized FFT. The output window of ``out_shape`` pixels is centred on
    the optical axis (shifted by ``offset`` output pixels); fractional offsets
    are exact, applied as an input phase ramp.

    Parameters
    ----------
    in_shape:
        Input (pupil) grid shape.
    out_shape:
        Output (image) window shape; each axis must be ``<= n_fft``.
    n_fft:
        FFT size per axis, ``>=`` the input size. One output pixel is
        ``1 / n_fft`` cycles per input pixel.
    offset:
        Output-window centre offset in output pixels, ``(dy, dx)``.
    """

    def __init__(
        self,
        in_shape: Any,
        out_shape: Any,
        n_fft: Any,
        *,
        offset: Any = 0.0,
        backend: BackendLike = None,
    ) -> None:
        self.backend = get_backend(backend)
        self.in_shape = _shape(in_shape, "in_shape")
        self.out_shape = _shape(out_shape, "out_shape")
        self.n_fft = _shape(n_fft, "n_fft")
        self.offset = _pair(offset, "offset")
        for axis in range(2):
            if self.n_fft[axis] < self.in_shape[axis]:
                raise ValueError(
                    f"n_fft {self.n_fft} must be at least the input shape {self.in_shape}; "
                    "use MFTPropagator for an undersampled transform"
                )
            if self.n_fft[axis] < self.out_shape[axis]:
                raise ValueError(
                    f"output window {self.out_shape} is larger than n_fft {self.n_fft}: the "
                    "FFT field would repeat; use MFTPropagator instead"
                )
        xp = self.backend.xp
        in_mods, out_mods = [], []
        for axis in range(2):
            n, m, big = self.in_shape[axis], self.out_shape[axis], self.n_fft[axis]
            c_in = (n - 1) / 2.0
            c_out = (m - 1) / 2.0 - self.offset[axis]
            # Output pixel j has frequency (j - c_out) / big. Folding the whole shift
            # into the input ramp puts the window on FFT bins 0..m-1, so the output
            # is a slice of the spectrum rather than a gather.
            i = np.arange(n, dtype=np.float64)
            freq = np.arange(m, dtype=np.float64) - c_out
            in_mods.append(np.exp(2j * np.pi * i * c_out / big))
            out_mods.append(np.exp(2j * np.pi * c_in * freq / big))
        cdt = self.backend.complex_dtype
        self._in_mod = xp.asarray(np.outer(in_mods[0], in_mods[1]), dtype=cdt)
        self._out_mod = xp.asarray(np.outer(out_mods[0], out_mods[1]), dtype=cdt)
        self._in_mod_conj = xp.conj(self._in_mod)
        self._out_mod_conj = xp.conj(self._out_mod)

    def forward(self, field: Any) -> Any:
        xp = self.backend.xp
        ny, nx = self.in_shape
        big_y, big_x = self.n_fft
        lead = field.shape[:-2]
        padded = xp.zeros((*lead, big_y, big_x), dtype=self.backend.complex_dtype)
        padded[..., :ny, :nx] = field * self._in_mod
        spectrum = self.backend.fft2(padded)
        my, mx = self.out_shape
        return spectrum[..., :my, :mx] * self._out_mod

    def adjoint(self, field: Any) -> Any:
        xp = self.backend.xp
        ny, nx = self.in_shape
        big_y, big_x = self.n_fft
        lead = field.shape[:-2]
        spectrum = xp.zeros((*lead, big_y, big_x), dtype=self.backend.complex_dtype)
        my, mx = self.out_shape
        spectrum[..., :my, :mx] = field * self._out_mod_conj
        padded = self.backend.ifft2(spectrum)
        return padded[..., :ny, :nx] * self._in_mod_conj


# --------------------------------------------------------------------- MFT
class MFTPropagator(Propagator):
    """Matrix Fourier transform with arbitrary sampling (Soummer et al. 2007).

    ``E = A_y u A_x^T`` with ``A[j, i] = exp(-2 pi i (i - c_in)(j - c_out) / N)
    / sqrt(N)``, where ``N`` (``samples``) is the equivalent FFT size and need
    not be an integer. Several ``N`` values (one per wavelength) give a stacked
    propagator mapping ``(..., L, ny, nx)`` to ``(..., L, my, mx)``.

    Parameters
    ----------
    in_shape, out_shape:
        Input and output grid shapes.
    samples:
        Equivalent FFT size ``N = 1 / (input pitch x output pitch)`` in
        dimensionless units: a scalar or ``(y, x)`` pair, or with ``stack=True``
        an ``(L,)`` or ``(L, 2)`` array, one entry per wavelength.
    offset:
        Output-window centre offset in output pixels, ``(dy, dx)``.
    """

    def __init__(
        self,
        in_shape: Any,
        out_shape: Any,
        samples: Any,
        *,
        offset: Any = 0.0,
        stack: bool = False,
        backend: BackendLike = None,
    ) -> None:
        self.backend = get_backend(backend)
        self.in_shape = _shape(in_shape, "in_shape")
        self.out_shape = _shape(out_shape, "out_shape")
        self.offset = _pair(offset, "offset")
        self.stacked = bool(stack)
        arr = np.asarray(samples, dtype=np.float64)
        if not self.stacked:
            if arr.ndim == 0:
                pairs = [(float(arr), float(arr))]
            elif arr.shape == (2,):
                pairs = [(float(arr[0]), float(arr[1]))]
            else:
                raise ValueError(f"samples must be a scalar or (y, x) pair, got {samples!r}")
        elif arr.ndim == 1:
            pairs = [(float(v), float(v)) for v in arr]
        elif arr.ndim == 2 and arr.shape[1] == 2:
            pairs = [(float(a), float(b)) for a, b in arr]
        else:
            raise ValueError(f"stacked samples must be (L,) or (L, 2), got shape {arr.shape}")
        if any(v <= 0 or not math.isfinite(v) for p in pairs for v in p):
            raise ValueError("samples must be positive and finite")
        self.samples = pairs
        xp = self.backend.xp
        cdt = self.backend.complex_dtype
        ay = np.stack([self._matrix(0, p[0]) for p in pairs])
        ax = np.stack([self._matrix(1, p[1]) for p in pairs])
        if not self.stacked:
            ay, ax = ay[0], ax[0]
        self._ay = xp.asarray(ay, dtype=cdt)
        self._axt = xp.asarray(np.swapaxes(ax, -1, -2), dtype=cdt)
        self._ayh = xp.asarray(np.conj(np.swapaxes(ay, -1, -2)), dtype=cdt)
        self._axc = xp.asarray(np.conj(ax), dtype=cdt)

    def _matrix(self, axis: int, big: float) -> np.ndarray:
        n, m = self.in_shape[axis], self.out_shape[axis]
        x = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
        f = np.arange(m, dtype=np.float64) - (m - 1) / 2.0 + self.offset[axis]
        return np.exp(-2j * np.pi * np.outer(f, x) / big) / math.sqrt(big)

    def _work(self, field: Any) -> float:
        my, mx = self.out_shape
        ny, nx = self.in_shape
        return float(field.size // (ny * nx)) * (my * ny * nx + my * nx * mx)

    def forward(self, field: Any) -> Any:
        xp = self.backend.xp
        with self.backend.blas_limit(self._work(field)):
            return xp.matmul(xp.matmul(self._ay, field), self._axt)

    def adjoint(self, field: Any) -> Any:
        xp = self.backend.xp
        my, mx = self.out_shape
        ny, nx = self.in_shape
        work = float(field.size // (my * mx)) * (ny * my * mx + ny * mx * nx)
        with self.backend.blas_limit(work):
            return xp.matmul(xp.matmul(self._ayh, field), self._axc)


# ----------------------------------------------------------- focal plane
class FocalPlanePropagator(Propagator):
    """Fraunhofer propagation from a pupil to a detector, at one or more wavelengths.

    Maps a pupil field ``(..., L, ny, nx)`` (one slice per wavelength) to an
    image field ``(..., L, my, mx)``. With a single wavelength the wavelength
    axis is still present (``L = 1``).

    The engine is chosen per call to :meth:`build`: a zero-padded FFT when the
    equivalent FFT size ``N = lambda / (pupil pitch x pixel scale)`` is an
    integer at every wavelength and the FFT is cheaper, otherwise a stacked
    matrix Fourier transform, which handles any sampling (including
    undersampled detectors) and broadband light exactly.

    Parameters
    ----------
    pupil_shape:
        ``(ny, nx)`` of the pupil grid.
    pupil_pitch:
        Pupil pixel pitch in metres.
    wavelengths:
        Wavelength(s) in metres.
    pixel_scale:
        Detector (or model) pixel scale in radians.
    out_shape:
        ``(my, mx)`` of the image.
    offset:
        Image-window centre offset in pixels ``(dy, dx)``.
    method:
        ``"auto"``, ``"fft"`` or ``"mft"``.
    """

    def __init__(
        self,
        pupil_shape: Any,
        pupil_pitch: float,
        wavelengths: Any,
        pixel_scale: float,
        out_shape: Any,
        *,
        offset: Any = 0.0,
        method: str = "auto",
        backend: BackendLike = None,
    ) -> None:
        self.backend = get_backend(backend)
        self.in_shape = _shape(pupil_shape, "pupil_shape")
        self.out_shape = _shape(out_shape, "out_shape")
        self.wavelengths = np.atleast_1d(np.asarray(wavelengths, dtype=np.float64))
        if self.wavelengths.ndim != 1 or np.any(self.wavelengths <= 0):
            raise ValueError("wavelengths must be positive")
        if pupil_pitch <= 0 or pixel_scale <= 0:
            raise ValueError("pupil_pitch and pixel_scale must be positive")
        self.pupil_pitch = float(pupil_pitch)
        self.pixel_scale = float(pixel_scale)
        self.offset = _pair(offset, "offset")
        self.samples = self.wavelengths / (self.pupil_pitch * self.pixel_scale)
        method = method.lower()
        if method not in ("auto", "fft", "mft"):
            raise ValueError("method must be 'auto', 'fft' or 'mft'")
        fft_ok = self._fft_feasible()
        if method == "fft" and not fft_ok:
            raise ValueError(
                "an FFT engine needs an integer equivalent FFT size "
                f"(lambda / (pitch x pixel_scale) = {self.samples}) at least as large as the "
                "pupil and image grids; use method='mft'"
            )
        if method == "auto":
            method = "fft" if fft_ok and self._fft_cheaper() else "mft"
        self.method = method
        if method == "fft":
            self._engines: list[Propagator] = [
                FFTPropagator(
                    self.in_shape,
                    self.out_shape,
                    round(n),
                    offset=self.offset,
                    backend=self.backend,
                )
                for n in self.samples
            ]
            self._mft: MFTPropagator | None = None
        else:
            self._engines = []
            self._mft = MFTPropagator(
                self.in_shape,
                self.out_shape,
                [float(n) for n in self.samples],
                offset=self.offset,
                stack=True,
                backend=self.backend,
            )

    @property
    def n_wavelengths(self) -> int:
        """Number of wavelength slices ``L``."""
        return int(self.wavelengths.size)

    def _fft_feasible(self) -> bool:
        for n in self.samples:
            rounded = round(n)
            if abs(n - rounded) > 1e-9 * max(1.0, n):
                return False
            if rounded < max(self.in_shape) or rounded < max(self.out_shape):
                return False
        return True

    def _fft_cheaper(self) -> bool:
        ny, nx = self.in_shape
        my, mx = self.out_shape
        fft_cost = 0.0
        for n in self.samples:
            big = float(round(n))
            fft_cost += 5.0 * big * big * math.log2(big * big) + 6.0 * big * big
        mft_cost = self.n_wavelengths * 8.0 * (my * ny * nx + my * nx * mx)
        if self.backend.is_gpu:
            mft_cost *= 0.35  # dense complex matmul runs closer to peak than cuFFT
        return fft_cost <= mft_cost

    def forward(self, field: Any) -> Any:
        if self._mft is not None:
            return self._mft.forward(field)
        xp = self.backend.xp
        return xp.stack(
            [eng.forward(field[..., i, :, :]) for i, eng in enumerate(self._engines)], axis=-3
        )

    def adjoint(self, field: Any) -> Any:
        if self._mft is not None:
            return self._mft.adjoint(field)
        xp = self.backend.xp
        return xp.stack(
            [eng.adjoint(field[..., i, :, :]) for i, eng in enumerate(self._engines)], axis=-3
        )


# ---------------------------------------------------------- near field
class AngularSpectrumPropagator(Propagator):
    """Scalar angular-spectrum (Fresnel/near-field) propagation over a distance.

    ``E_z = IFFT[ FFT[u] * H ]`` with ``H = exp(i 2 pi z sqrt(1/lambda^2 - f^2))``
    and evanescent components removed. Shapes are unchanged. Several distances
    give a stack ``(..., Z, ny, nx)`` from a field broadcast against it.

    Parameters
    ----------
    shape:
        ``(ny, nx)`` of the field.
    pitch:
        Sample pitch in metres.
    wavelength:
        Wavelength in metres.
    distances:
        Propagation distance(s) in metres (positive downstream).
    paraxial:
        Use the Fresnel (paraxial) transfer function
        ``exp(-i pi lambda z f^2)`` (global phase dropped).
    """

    def __init__(
        self,
        shape: Any,
        pitch: float,
        wavelength: float,
        distances: float | Sequence[float],
        *,
        paraxial: bool = False,
        backend: BackendLike = None,
    ) -> None:
        self.backend = get_backend(backend)
        self.in_shape = _shape(shape, "shape")
        self.out_shape = self.in_shape
        self.pitch = float(pitch)
        self.wavelength = float(wavelength)
        dist = np.atleast_1d(np.asarray(distances, dtype=np.float64))
        self.stacked = np.ndim(distances) > 0
        self.distances = dist
        ny, nx = self.in_shape
        fy = np.fft.fftfreq(ny, d=self.pitch)[:, None]
        fx = np.fft.fftfreq(nx, d=self.pitch)[None, :]
        f2 = fy**2 + fx**2
        if paraxial:
            h = np.exp(-1j * np.pi * self.wavelength * dist[:, None, None] * f2[None])
        else:
            arg = 1.0 / self.wavelength**2 - f2
            kz = np.sqrt(np.maximum(arg, 0.0))
            h = np.where(arg[None] > 0, np.exp(2j * np.pi * dist[:, None, None] * kz[None]), 0.0)
            # Drop the plane-wave (piston) term common to every frequency.
            h = h * np.exp(-2j * np.pi * dist[:, None, None] / self.wavelength)
        if not self.stacked:
            h = h[0]
        xp = self.backend.xp
        self._h = xp.asarray(h, dtype=self.backend.complex_dtype)
        self._hc = xp.conj(self._h)

    def forward(self, field: Any) -> Any:
        if self.stacked:
            field = field[..., None, :, :]
        return self.backend.ifft2(self.backend.fft2(field) * self._h)

    def adjoint(self, field: Any) -> Any:
        out = self.backend.ifft2(self.backend.fft2(field) * self._hc)
        if self.stacked:
            out = out.sum(axis=-3)
        return out
