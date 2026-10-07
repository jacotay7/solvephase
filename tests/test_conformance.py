"""solvephase follows the stack conventions (aocore CONVENTIONS.md)."""

from __future__ import annotations

import numpy as np
from aocore import conformance as cf

import solvephase as sp

LAM = 1.6e-6
PUPIL = sp.Pupil.circular(32, 8.0, obscuration=0.14)


def _model(window: int, sampling: float = 4.0) -> sp.FocalPlaneModel:
    return sp.FocalPlaneModel(PUPIL, LAM, window, sampling=sampling)


def test_image_centring_and_tilt_direction() -> None:
    model = _model(48)

    def image(opd: np.ndarray) -> np.ndarray:
        return sp.to_numpy(model.images(opd))[0]

    cf.check_image_centring(image, pupil_shape=PUPIL.shape)
    cf.check_tilt_direction(
        image, pupil_shape=PUPIL.shape, pitch=PUPIL.pitch, pixel_scale=model.pixel_scale
    )


def test_normalized_images_sum_to_one() -> None:
    model = _model(128)
    cf.check_unit_flux(lambda opd: sp.to_numpy(model.images(opd))[0], pupil_shape=PUPIL.shape)


def test_rms_definition() -> None:
    cf.check_rms(lambda opd, amplitude: sp.rms(opd, amplitude))
