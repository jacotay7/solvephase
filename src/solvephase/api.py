"""One-call focal-plane phase retrieval: :func:`retrieve`."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from .algorithms.gerchberg_saxton import gerchberg_saxton
from .backend import BackendLike, to_numpy
from .basis import Basis
from .focal import FocalPlaneModel, zernike_diversity
from .losses import Loss
from .pupil import Pupil
from .result import Result
from .retrieval import FocalPlaneProblem, solve

__all__ = ["retrieve"]


def _as_pupil(pupil: Pupil | int, diameter: float | None) -> Pupil:
    if isinstance(pupil, Pupil):
        return pupil
    if isinstance(pupil, (int, np.integer)):
        return Pupil.circular(int(pupil), diameter or 1.0)
    raise TypeError("pupil must be a Pupil or an int (grid size of a circular pupil)")


def _as_diversity(pupil: Pupil, diversity: Any, n_images: int) -> np.ndarray | None:
    if diversity is None:
        if n_images > 1:
            raise ValueError(
                f"{n_images} images were given but no diversity: give one OPD map (or one "
                "defocus RMS in metres) per image"
            )
        return None
    if isinstance(diversity, dict):
        noll = int(diversity.get("noll", 4))
        return zernike_diversity(pupil, noll, diversity["amounts"])
    if np.ndim(diversity) == 1 and all(np.isscalar(v) for v in diversity):
        return zernike_diversity(pupil, 4, [float(v) for v in diversity])
    maps = np.asarray(diversity, dtype=np.float64)
    if maps.ndim == 2:
        maps = maps[None]
    return maps


def _as_basis(pupil: Pupil, basis: Any) -> Basis | None:
    if basis is None or (isinstance(basis, str) and basis.lower() == "zonal"):
        return None
    if isinstance(basis, Basis):
        return basis
    if isinstance(basis, (int, np.integer)):
        return Basis.zernike(pupil, int(basis))
    raise TypeError("basis must be a Basis, an int (number of Zernike modes) or 'zonal'")


def retrieve(
    images: Any,
    pupil: Pupil | int,
    wavelength: float | Sequence[float],
    *,
    sampling: float | None = None,
    pixel_scale: float | None = None,
    diversity: Any = None,
    basis: Basis | int | str | None = 36,
    zonal_refinement: bool = False,
    method: str = "auto",
    loss: str | Loss = "poisson",
    read_noise: float = 0.0,
    weights: Any = None,
    fit_background: bool = True,
    fit_tilt: bool = False,
    spectral_weights: Sequence[float] | None = None,
    oversample: int = 1,
    diameter: float | None = None,
    start: Any = None,
    max_iter: int | None = None,
    device: BackendLike = "cpu",
    precision: str | None = None,
) -> Result:
    """Retrieve a pupil wavefront from focal-plane images in one call.

    Parameters
    ----------
    images:
        ``(K, my, mx)`` images (or one ``(my, mx)`` image), ideally in
        photo-electrons, background included.
    pupil:
        A :class:`~solvephase.Pupil`, or an int ``n`` for an ``n x n``
        circular pupil of ``diameter`` (default 1 m).
    wavelength:
        Wavelength in metres, or an array sampling the band.
    sampling, pixel_scale:
        Detector sampling in pixels per lambda/D, or pixel scale in radians
        (give one).
    diversity:
        Known per-image aberrations: ``(K, ny, nx)`` OPD maps in metres; a
        list of ``K`` numbers, read as defocus (Noll 4) RMS in metres per
        image; or ``{"noll": j, "amounts": [...]}``. Required for more than one
        image.
    basis:
        Wavefront parameterization: number of Zernike modes (default 36,
        tip/tilt upward), a :class:`~solvephase.Basis`, or ``"zonal"``.
    zonal_refinement:
        After the modal solve, refine pixel by pixel (L-BFGS).
    method:
        ``"auto"`` (robust multi-start: Levenberg-Marquardt from a flat start
        and from Gerchberg-Saxton, best kept, then a maximum-likelihood
        polish), ``"lm"``, ``"lbfgs"``, ``"gs"`` (Gerchberg-Saxton/Misell only)
        or ``"gs+lm"``.
    loss:
        Final data term: ``"poisson"`` (default; images in photo-electrons),
        ``"gaussian"`` or ``"amplitude"``.
    read_noise:
        Read noise in electrons (Poisson loss).
    weights:
        Per-pixel weights; zero masks bad pixels.
    fit_background:
        Fit a constant background per image.
    fit_tilt:
        Fit a registration tip/tilt per image relative to the first.
    spectral_weights:
        Relative photon flux per wavelength for broadband data.
    oversample:
        Model pixels per detector pixel per axis (pixel integration).
    diameter:
        Diameter in metres when ``pupil`` is an int.
    start:
        Initial guess: OPD map, coefficients or a previous :class:`Result`.
    max_iter:
        Iteration limit of the final stage.
    device, precision:
        ``"cpu"``, ``"gpu"`` or ``"auto"``; ``"single"`` or ``"double"``.

    Returns
    -------
    Result
        ``result.opd`` (metres), ``result.coefficients``, model images,
        flux/background, convergence history. ``result.extra["stages"]``
        records each stage's loss and time.

    Examples
    --------
    >>> import solvephase as sp
    >>> pupil = sp.Pupil.circular(64, 8.0, obscuration=0.14)
    >>> result = sp.retrieve(images, pupil, 1.6e-6, sampling=2.0,
    ...                      diversity=[0.0, 0.4e-6])  # doctest: +SKIP
    >>> result.opd, result.coefficients  # doctest: +SKIP
    """
    t0 = time.perf_counter()
    pupil = _as_pupil(pupil, diameter)
    data = np.asarray(to_numpy(images), dtype=np.float64)
    if data.ndim == 2:
        data = data[None]
    if data.ndim != 3:
        raise ValueError(f"images must be (my, mx) or (K, my, mx), got shape {data.shape}")
    div = _as_diversity(pupil, diversity, data.shape[0])
    if div is not None and div.shape[0] != data.shape[0]:
        raise ValueError(f"{div.shape[0]} diversity maps for {data.shape[0]} images")
    model = FocalPlaneModel(
        pupil,
        wavelength,
        data.shape[-2:],
        sampling=sampling,
        pixel_scale=pixel_scale,
        weights=spectral_weights,
        diversity=div,
        oversample=oversample,
        device=device,
        precision=precision,
    )
    modal = _as_basis(pupil, basis)
    common: dict[str, Any] = {
        "weights": weights,
        "fit_background": fit_background,
        "fit_tilt": fit_tilt and data.shape[0] > 1,
        "read_noise": read_noise,
    }
    final = FocalPlaneProblem(model, data, basis=modal, loss=loss, **common)
    stages: list[dict[str, Any]] = []

    def record(name: str, result: Result) -> Result:
        stages.append(
            {
                "stage": name,
                "loss": float(final.value(final.initial(result))),
                "time": result.elapsed,
            }
        )
        return result

    method = method.lower()
    if method == "gs":
        result = gerchberg_saxton(model, data, basis=modal, weights=weights)
        best = record("gs", result)
    elif method in ("lm", "lbfgs"):
        if method == "lm" and modal is None:
            raise ValueError("method='lm' needs a modal basis; use 'lbfgs' for a zonal solve")
        best = record(method, solve(final, method=method, start=start, max_iter=max_iter))
    elif method in ("auto", "gs+lm"):
        explorer = modal if modal is not None else Basis.zernike(pupil, 36)
        capture = FocalPlaneProblem(model, data, basis=explorer, loss="amplitude", **common)
        candidates: list[Result] = []
        if method == "auto" or start is not None:
            candidates.append(record("lm-amplitude", solve(capture, method="lm", start=start)))
        gs = gerchberg_saxton(model, data, iterations=200, weights=weights, start=start)
        candidates.append(record("gs->lm-amplitude", solve(capture, method="lm", start=gs)))
        best = min(candidates, key=lambda r: final.value(final.initial(r)))
        polish_method = "lm" if modal is not None else "lbfgs"
        best = record("polish", solve(final, method=polish_method, start=best, max_iter=max_iter))
    else:
        raise ValueError(f"unknown method {method!r}; use 'auto', 'lm', 'lbfgs', 'gs' or 'gs+lm'")

    if zonal_refinement and modal is not None:
        zonal = FocalPlaneProblem(model, data, basis=None, loss=loss, **common)
        refined = solve(zonal, method="lbfgs", start=best, max_iter=max_iter or 300)
        stages.append({"stage": "zonal", "loss": refined.loss, "time": refined.elapsed})
        best = refined
    best.elapsed = time.perf_counter() - t0
    best.extra["stages"] = stages
    best.extra["model"] = model
    if not math.isfinite(best.loss):
        best.loss = float(final.value(final.initial(best)))
    return best
