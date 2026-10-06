"""Synthetic data for tests, tutorials and benchmarks.

For physically detailed data, see :mod:`solvephase.interop`: atmospheric OPD
from ``pyturb`` and detector frames from ``getframes``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .backend import to_numpy
from .basis import Basis
from .focal import FocalPlaneModel
from .pupil import Pupil

__all__ = ["random_aberration", "simulate_images"]


def random_aberration(
    pupil: Pupil,
    rms: float,
    *,
    n_modes: int = 36,
    power: float = 1.0,
    start: int = 2,
    seed: Any = None,
) -> np.ndarray:
    """A random Zernike aberration with a power-law spectrum.

    Coefficient ``j`` (counting from ``start`` in Noll order) has standard
    deviation proportional to ``j^-power``; the map is scaled to exactly ``rms``
    metres RMS over the pupil (piston removed).

    Parameters
    ----------
    pupil:
        The pupil.
    rms:
        Target RMS OPD in metres.
    n_modes:
        Number of Zernike modes.
    power:
        Spectral slope (0 = white, ~1 typical of polished optics).
    start:
        First Noll index (2 includes tip/tilt, 4 starts at defocus).
    seed:
        Seed or :class:`numpy.random.Generator`.

    Returns
    -------
    ``(ny, nx)`` OPD map in metres.
    """
    rng = np.random.default_rng(seed)
    basis = Basis.zernike(pupil, n_modes, start=start)
    coeffs = rng.standard_normal(n_modes) / np.arange(1, n_modes + 1, dtype=float) ** power
    opd = basis.synthesize(coeffs)
    w = pupil.amplitude**2
    mean = np.sum(w * opd) / np.sum(w)
    opd = np.where(pupil.mask, opd - mean, 0.0)
    current = np.sqrt(np.sum(w * opd**2) / np.sum(w))
    return opd * (rms / current) if current > 0 else opd


def simulate_images(
    model: FocalPlaneModel,
    opd: Any = None,
    *,
    photons: float | Any = 1e6,
    background: float | Any = 0.0,
    read_noise: float = 0.0,
    noise: bool = True,
    amplitude: Any = None,
    seed: Any = None,
) -> np.ndarray:
    """Simulate photon-counting focal-plane images for a model.

    Parameters
    ----------
    model:
        The forward model.
    opd:
        Pupil OPD in metres (default flat).
    photons:
        Photons collected per channel from the whole PSF (scalar or ``(K,)``).
    background:
        Background photons per pixel per channel.
    read_noise:
        Gaussian read noise standard deviation in electrons.
    noise:
        Add Poisson and read noise; ``False`` returns the expectation.
    amplitude:
        Optional pupil amplitude override.
    seed:
        Seed or :class:`numpy.random.Generator`.

    Returns
    -------
    ``(K, my, mx)`` host array in photo-electrons (quantum efficiency 1).
    """
    rng = np.random.default_rng(seed)
    psf = np.asarray(to_numpy(model.images(opd, amplitude=amplitude)), dtype=np.float64)
    k = psf.shape[0]
    flux = np.broadcast_to(np.asarray(photons, dtype=np.float64), (k,))
    bg = np.broadcast_to(np.asarray(background, dtype=np.float64), (k,))
    expected = flux[:, None, None] * psf + bg[:, None, None]
    if not noise:
        return expected
    images = rng.poisson(np.maximum(expected, 0.0)).astype(np.float64)
    if read_noise > 0:
        images += rng.normal(0.0, read_noise, size=images.shape)
    return images
