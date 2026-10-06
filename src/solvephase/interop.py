"""Bridges to the rest of the AO simulation family.

* :func:`turbulence_opd` - Kolmogorov / von Karman OPD over a pupil from
  `pyturb <https://github.com/jacotay7/pyturb>`_;
* :func:`expose` - turn model images into realistic detector frames with
  `getframes <https://github.com/jacotay7/getframes>`_ (shot, read and dark
  noise, gain, bias, digitization), and back into photo-electrons for the
  Poisson likelihood.

Both packages are optional (``pip install 'solvephase[interop]'``); modal
bases from `aobasis <https://github.com/jacotay7/aobasis>`_ are built in
(:class:`solvephase.Basis`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .backend import to_numpy
from .pupil import Pupil

__all__ = ["DetectorImages", "expose", "turbulence_opd"]


def _require(name: str) -> Any:
    try:
        return __import__(name)
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            f"this function needs {name}; install it with pip install 'solvephase[interop]'"
        ) from exc


def turbulence_opd(
    pupil: Pupil,
    r0: float,
    *,
    outer_scale: float = 25.0,
    r0_wavelength: float = 500e-9,
    count: int | None = None,
    seed: Any = None,
) -> np.ndarray:
    """Atmospheric OPD over a pupil from :class:`pyturb.PhaseScreen`.

    Parameters
    ----------
    pupil:
        The pupil (its pitch sets the screen pixel scale).
    r0:
        Fried parameter in metres at ``r0_wavelength``.
    outer_scale:
        Outer scale in metres (``numpy.inf`` for Kolmogorov).
    r0_wavelength:
        Wavelength at which ``r0`` is given.
    count:
        Number of independent screens (``None`` for a single map).
    seed:
        Seed passed to pyturb.

    Returns
    -------
    ``(ny, nx)`` (or ``(count, ny, nx)``) OPD in metres, zero outside the
    pupil. OPD is achromatic, so the result is valid at any wavelength.
    """
    pyturb = _require("pyturb")
    ny, nx = pupil.shape
    n = max(ny, nx)
    gen = pyturb.PhaseScreen(
        n=n, pixel_scale=pupil.pitch, r0=r0, L0=outer_scale, seed=seed, dtype="float64"
    )
    phase = np.asarray(to_numpy(gen.generate(count)), dtype=np.float64)
    oy, ox = (n - ny) // 2, (n - nx) // 2
    phase = phase[..., oy : oy + ny, ox : ox + nx]
    opd = phase * r0_wavelength / (2.0 * np.pi)
    return np.where(pupil.mask, opd, 0.0)


@dataclass
class DetectorImages:
    """Detector frames and their conversion to photo-electrons.

    Attributes
    ----------
    adu:
        ``(K, my, mx)`` raw frames in ADU.
    electrons:
        ``(K, my, mx)`` bias-subtracted, gain-corrected frames in
        photo-electrons, for the Poisson likelihood.
    read_noise:
        Read noise in electrons (pass to ``read_noise=`` of the solver).
    truth:
        ``(K, my, mx)`` noise-free expected photo-electrons.
    weights:
        ``(K, my, mx)`` pixel weights for the solvers: 0 where the pixel is
        saturated (digitizer ceiling or full well), else 1.
    """

    adu: np.ndarray
    electrons: np.ndarray
    read_noise: float
    truth: np.ndarray
    weights: np.ndarray

    @property
    def n_saturated(self) -> int:
        """Number of saturated (masked) pixels."""
        return int(np.count_nonzero(self.weights == 0))


def expose(
    images: Any,
    camera: Any = "andor_ikon_m934",
    *,
    exposure: float = 1.0,
    photons: float | Any = 1e6,
    background: float = 0.0,
    seed: int | None = None,
) -> DetectorImages:
    """Expose normalized model images on a :class:`getframes.Camera`.

    Parameters
    ----------
    images:
        ``(K, my, mx)`` normalized model images (each sums to about 1, as
        returned by :meth:`FocalPlaneModel.images`).
    camera:
        A getframes preset name or a :class:`getframes.Camera`; its resolution
        is set to the image shape.
    exposure:
        Exposure time in seconds.
    photons:
        Photons per channel arriving at the detector during the exposure.
    background:
        Background photons per pixel during the exposure.
    seed:
        Base seed; channel ``k`` uses ``seed + k``.
    """
    gf = _require("getframes")
    data = np.asarray(to_numpy(images), dtype=np.float64)
    if data.ndim == 2:
        data = data[None]
    k, my, mx = data.shape
    cam = gf.Camera.from_preset(camera) if isinstance(camera, str) else camera
    cam = cam.with_config(resolution=(my, mx))
    cfg = cam.config
    flux = np.broadcast_to(np.asarray(photons, dtype=np.float64), (k,))
    adu = np.empty_like(data)
    truth = np.empty_like(data)
    for c in range(k):
        rate = flux[c] * data[c] / exposure
        frame = cam.expose(
            rate,
            exposure,
            background=background / exposure,
            seed=None if seed is None else seed + c,
        )
        adu[c] = np.asarray(to_numpy(frame.data), dtype=np.float64)
        truth[c] = np.asarray(to_numpy(frame.truth.mean_photoelectrons), dtype=np.float64)
    em_gain = float(getattr(cfg, "em_gain", 1.0) or 1.0)
    electrons = (adu - cfg.bias_offset_adu) * cfg.gain_e_per_adu / em_gain
    ceiling = 2 ** int(cfg.bit_depth) - 1
    full_well = float(cfg.output_full_well_e or cfg.full_well_e)
    saturated = (adu >= ceiling) | (electrons * em_gain >= 0.98 * full_well)
    return DetectorImages(
        adu=adu,
        electrons=electrons,
        read_noise=float(cfg.read_noise_e) / em_gain,
        truth=truth,
        weights=np.where(saturated, 0.0, 1.0),
    )
