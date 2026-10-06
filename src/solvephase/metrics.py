"""Wavefront error metrics that respect phase-retrieval ambiguities.

Phase retrieval cannot see some components of the wavefront: piston always,
absolute tip/tilt when the image position is free, and for a single in-focus
image of a centro-symmetric pupil the *twin* ``phi(x) -> -phi(-x)``. These
helpers compare an estimate with a reference after removing exactly those
components, weighted by pupil intensity.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from .backend import to_numpy
from .pupil import Pupil

__all__ = ["remove_modes", "rms", "strehl_from_rms", "wavefront_error"]

RemoveSpec = str | Sequence[str] | None


def _removal_maps(pupil: Pupil, remove: RemoveSpec) -> list[np.ndarray]:
    if remove is None:
        return []
    names = [remove] if isinstance(remove, str) else list(remove)
    y, x = pupil.coordinates()
    maps: list[np.ndarray] = []
    for name in names:
        if name == "piston":
            maps.append(np.ones(pupil.shape))
        elif name in ("tip", "tiptilt"):
            if name == "tiptilt":
                maps += [np.ones(pupil.shape), np.asarray(x), np.asarray(y)]
            else:
                maps.append(np.asarray(x))
        elif name == "tilt":
            maps.append(np.asarray(y))
        elif name == "segment_piston":
            if pupil.segments is None:
                raise ValueError("segment_piston needs a segmented pupil")
            maps += [(pupil.segments == s).astype(float) for s in range(1, pupil.n_segments + 1)]
        else:
            raise ValueError(f"unknown mode to remove {name!r}")
    return maps


def remove_modes(opd: Any, pupil: Pupil, remove: RemoveSpec = "piston") -> np.ndarray:
    """Subtract the intensity-weighted least-squares fit of the named modes.

    ``remove`` is ``"piston"``, ``"tip"``, ``"tilt"``, ``"tiptilt"``
    (piston + tip + tilt), ``"segment_piston"``, a list of these, or None.
    """
    data = np.asarray(to_numpy(opd), dtype=np.float64)
    if data.shape != pupil.shape:
        raise ValueError(f"OPD shape {data.shape} does not match the pupil {pupil.shape}")
    mask = pupil.mask
    out = np.where(mask, data, 0.0)
    maps = _removal_maps(pupil, remove)
    if not maps:
        return out
    w = np.sqrt((pupil.amplitude**2)[mask])
    basis = np.stack([m[mask] for m in maps], axis=1)
    coef, *_ = np.linalg.lstsq(basis * w[:, None], data[mask] * w, rcond=None)
    out[mask] = data[mask] - basis @ coef
    return out


def rms(opd: Any, pupil: Pupil, remove: RemoveSpec = "piston") -> float:
    """Intensity-weighted RMS of an OPD (or phase) map over the pupil after :func:`remove_modes`."""
    resid = remove_modes(opd, pupil, remove)
    w = pupil.amplitude**2
    return float(math.sqrt(np.sum(w * resid**2) / np.sum(w)))


def strehl_from_rms(rms_opd: float, wavelength: float) -> float:
    """Marechal approximation ``exp(-(2 pi rms / lambda)^2)``."""
    return float(math.exp(-((2.0 * math.pi * rms_opd / wavelength) ** 2)))


def wavefront_error(
    estimate: Any,
    truth: Any,
    pupil: Pupil,
    *,
    remove: RemoveSpec = "piston",
    allow_twin: bool = False,
) -> float:
    """RMS difference between two OPD maps over the pupil, modulo ambiguities.

    Parameters
    ----------
    estimate, truth:
        OPD maps (same units; the result is in those units).
    remove:
        Modes removed from the difference before the RMS (see
        :func:`remove_modes`).
    allow_twin:
        Also compare against the twin ``-estimate(-x)`` and return the smaller
        error (for single-image retrieval without diversity). Only meaningful
        for pupils symmetric under 180-degree rotation.
    """
    est = np.asarray(to_numpy(estimate), dtype=np.float64)
    ref = np.asarray(to_numpy(truth), dtype=np.float64)
    err = rms(est - ref, pupil, remove)
    if allow_twin:
        twin = -est[::-1, ::-1]
        err = min(err, rms(twin - ref, pupil, remove))
    return err
