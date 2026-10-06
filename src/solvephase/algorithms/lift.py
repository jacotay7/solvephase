"""LIFT: low-order wavefront sensing from one astigmatic focal-plane image.

LIFT (LInearized Focal-plane Technique; Meimon, Fusco & Mugnier, Opt. Lett.
35, 3036, 2010; Plantet et al., Opt. Express 21, 16337, 2013) estimates a
handful of low-order modes (tip, tilt, defocus, astigmatism, coma ...) from a
single, possibly very faint, focal-plane image. A known astigmatism is added
in the optical path. Without it a single in-focus image cannot tell the sign
of the *even* part of the wavefront (defocus, astigmatism, spherical...): the
twin ``phi(x) -> -phi(-x)`` gives the same image. The astigmatism moves that
twin far from the true solution (for the total phase ``phi + phi_d`` the twin
needs ``phi_d -> -phi_d``), so a local maximum-likelihood search started near
zero lands on the true wavefront.

The estimator is maximum likelihood under photon and read noise, computed by
iterated linearization of the PSF around the current estimate (Gauss-Newton,
with the noise variance ``model + read_noise^2`` as weights). In solvephase
this is exactly :func:`solvephase.solve` with ``method="lm"`` on a
:class:`~solvephase.FocalPlaneProblem` with a Poisson (shifted-Poisson) loss,
a modal basis and a fitted flux; this module only sets that problem up, keeps
the model for repeated estimates (:class:`LIFT`), and computes the
Cramer-Rao lower bound of the modal coefficients (:func:`lift_crlb`), which
LIFT approaches at high flux.

Units: coefficients are RMS OPD in metres of unit-RMS Zernike modes (Noll
order, starting at tip by default); the astigmatism is RMS OPD in metres;
images are in photo-electrons (read noise in electrons RMS).
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from ..backend import BackendLike
from ..basis import Basis
from ..focal import FocalPlaneModel, zernike_diversity
from ..losses import GaussianLoss, Loss, PoissonLoss
from ..pupil import Pupil
from ..result import Result
from ..retrieval import FocalPlaneProblem, noise_weights, solve

__all__ = ["LIFT", "lift", "lift_crlb", "lift_diversity"]


def lift_diversity(
    pupil: Pupil, wavelength: float, astigmatism: float | None = None, *, noll: int = 5
) -> np.ndarray:
    """The known LIFT astigmatism as an OPD map ``(ny, nx)`` in metres.

    Parameters
    ----------
    pupil:
        The pupil.
    wavelength:
        Wavelength in metres (sets the default amplitude).
    astigmatism:
        RMS OPD of the astigmatism in metres. ``None`` gives
        :func:`default_astigmatism` (``wavelength / 10`` RMS).
    noll:
        Noll index of the astigmatism mode, 5 (oblique) or 6 (vertical).
    """
    amount = default_astigmatism(wavelength) if astigmatism is None else float(astigmatism)
    if not math.isfinite(amount):
        raise ValueError("astigmatism must be finite (RMS OPD in metres)")
    return np.asarray(zernike_diversity(pupil, noll, [amount])[0])


def default_astigmatism(wavelength: float) -> float:
    """Default LIFT astigmatism: ``wavelength / 10`` RMS OPD (about half a wave peak to valley).

    A unit-RMS Zernike astigmatism spans ``2 sqrt 6`` peak to valley over the
    disk, so this is ``0.49`` waves PV. It is close to the 170 nm RMS at
    1.6 um of VLT/IRLOS (Kuznetsov et al. 2024). The Cramer-Rao bound is flat
    between about 0.05 and 0.15 waves RMS, while the capture range grows with
    the astigmatism (the twin solution sits at ``-a_astig - 2 d``).
    """
    return float(wavelength) / 10.0


def _image_stack(image: Any) -> Any:
    shape = tuple(getattr(image, "shape", np.shape(image)))
    if len(shape) == 3 and shape[0] == 1:
        return image
    if len(shape) != 2:
        raise ValueError(
            f"LIFT uses one focal-plane image (my, mx); got shape {shape}. "
            "For several diversity images use FocalPlaneProblem + solve (phase diversity)."
        )
    return image


def _norm(coefficients: np.ndarray, basis: Basis) -> float:
    """Pupil-weighted RMS of the wavefront ``B c``."""
    opd = basis.synthesize(coefficients)
    w = basis.pupil.amplitude**2
    return float(np.sqrt(np.sum(w * opd**2) / np.sum(w)))


class LIFT:
    """A reusable LIFT wavefront sensor: the model is built once, estimates are repeated.

    Parameters
    ----------
    pupil:
        The entrance pupil.
    wavelength:
        Wavelength in metres (or an array sampling a band, see
        :class:`~solvephase.FocalPlaneModel`).
    image_shape:
        Detector image shape ``(my, mx)``.
    sampling, pixel_scale:
        Detector pixels per ``lambda / D`` or pixel scale in radians (give
        one).
    n_modes:
        Number of Zernike modes estimated, Noll order from tip (Noll 2). Ignored
        when ``basis`` is given.
    basis:
        Any modal :class:`~solvephase.Basis` instead of Zernikes.
    astigmatism:
        RMS OPD of the known astigmatism in metres (default
        ``wavelength / 10``, see :func:`default_astigmatism`).
    astigmatism_mode:
        Noll index of the diversity astigmatism (5 or 6).
    read_noise:
        Read-noise standard deviation in electrons.
    loss:
        ``"poisson"`` (maximum likelihood under photon + read noise; the
        default) or ``"gaussian"`` (weighted least squares with variances from
        the image, :func:`~solvephase.noise_weights`, as in the original
        LIFT); or a :class:`~solvephase.Loss`.
    fit_flux, fit_background, background:
        Nuisance parameters, see :class:`~solvephase.FocalPlaneProblem`.
    max_iter:
        Gauss-Newton (Levenberg-Marquardt) iteration limit per estimate.
    resolve_twin:
        After convergence, compare the solution with its exact twin (same
        image, see :meth:`twin`) and report the one with the smaller
        wavefront RMS. Only active when the twin is exact: centro-symmetric
        pupil, centred image window and a basis closed under the twin
        (Zernikes are).
    oversample, offset:
        Detector pixel integration and optical-axis offset, see
        :class:`~solvephase.FocalPlaneModel`.
    device, precision:
        Backend selection.

    Examples
    --------
    >>> sensor = LIFT(pupil, 1.6e-6, 32, sampling=2, n_modes=10)  # doctest: +SKIP
    >>> for frame in frames:  # doctest: +SKIP
    ...     result = sensor.estimate(frame)
    ...     dm_command -= gain * result.coefficients
    """

    def __init__(
        self,
        pupil: Pupil,
        wavelength: Any,
        image_shape: Any,
        *,
        sampling: float | None = None,
        pixel_scale: float | None = None,
        n_modes: int = 10,
        basis: Basis | None = None,
        astigmatism: float | None = None,
        astigmatism_mode: int = 5,
        read_noise: float = 0.0,
        loss: str | Loss = "poisson",
        fit_flux: bool = True,
        fit_background: bool = False,
        background: float = 0.0,
        max_iter: int = 20,
        resolve_twin: bool = True,
        oversample: int = 1,
        offset: Any = 0.0,
        device: BackendLike = "cpu",
        precision: str | None = None,
    ) -> None:
        if astigmatism_mode not in (5, 6):
            raise ValueError("astigmatism_mode must be 5 (oblique) or 6 (vertical astigmatism)")
        if read_noise < 0:
            raise ValueError("read_noise must be >= 0 (electrons RMS)")
        wl = np.atleast_1d(np.asarray(wavelength, dtype=np.float64))
        ref = float(wl.mean())
        self.astigmatism = default_astigmatism(ref) if astigmatism is None else float(astigmatism)
        self.diversity = lift_diversity(pupil, ref, self.astigmatism, noll=astigmatism_mode)
        self.model = FocalPlaneModel(
            pupil,
            wavelength,
            image_shape,
            sampling=sampling,
            pixel_scale=pixel_scale,
            diversity=[self.diversity],
            oversample=oversample,
            offset=offset,
            device=device,
            precision=precision,
        )
        if basis is None:
            if n_modes < 1:
                raise ValueError("n_modes must be >= 1")
            basis = Basis.zernike(pupil, int(n_modes))
        if basis.is_zonal:
            raise ValueError("LIFT estimates a few modes; give a modal basis, not the zonal one")
        self.basis = basis
        self.basis.on(self.model.backend)  # cache the mode matrix on the device once
        self.read_noise = float(read_noise)
        if isinstance(loss, str) and loss.lower() not in ("poisson", "gaussian"):
            raise ValueError(f"unknown LIFT loss {loss!r}; use 'poisson' or 'gaussian'")
        self.loss = loss
        self.fit_flux = bool(fit_flux)
        self.fit_background = bool(fit_background)
        self.background = float(background)
        self.max_iter = int(max_iter)
        self._twin_map = self._twin_operator() if resolve_twin else None
        self.resolve_twin = self._twin_map is not None

    def _twin_operator(self) -> tuple[np.ndarray, np.ndarray] | None:
        """``(T, t)`` with twin coefficients ``T c + t``, or None if the twin is not exact.

        For a centro-symmetric pupil and a centred detector window the pupil
        phases ``phi + phi_d`` and ``-(phi + phi_d)(-x)`` give identical
        images, so ``phi' = -(phi + phi_d)(-x) - phi_d`` fits the data as
        well as ``phi``.
        """
        pupil = self.model.pupil
        amp = pupil.amplitude
        centred = self.model.offset == (0.0, 0.0)
        if not centred or not np.allclose(amp, amp[::-1, ::-1], atol=1e-12):
            return None
        maps = self.basis.mode_maps()
        flipped = -maps[:, ::-1, ::-1]
        div = self.diversity
        shift = -div[::-1, ::-1] - div
        targets = np.concatenate([flipped, shift[None]])
        coeffs = self.basis.fit(targets)  # (n + 1, n)
        resid = self.basis.synthesize(coeffs) - targets
        mask = pupil.mask
        scale = max(float(np.sqrt(np.mean(targets[:, mask] ** 2))), 1e-300)
        if float(np.sqrt(np.mean(resid[:, mask] ** 2))) > 1e-6 * scale:
            return None
        return np.asarray(coeffs[:-1].T), np.asarray(coeffs[-1])

    def twin(self, coefficients: Any) -> np.ndarray:
        """The twin coefficients (metres) that produce exactly the same image.

        Raises ``ValueError`` when the twin is not exact for this sensor (an
        asymmetric pupil, an off-centre window, or a basis not closed under
        ``phi(x) -> -phi(-x)``).
        """
        if self._twin_map is None:
            self._twin_map = self._twin_operator()
            if self._twin_map is None:
                raise ValueError(
                    "the twin is not exact here: it needs a centro-symmetric pupil, a centred "
                    "image window and a basis closed under phi(x) -> -phi(-x) (e.g. Zernikes)"
                )
        mat, shift = self._twin_map
        return np.asarray(mat @ np.asarray(coefficients, dtype=np.float64) + shift)

    # ------------------------------------------------------------- metadata
    @property
    def backend(self) -> Any:
        """The model's :class:`~solvephase.Backend`."""
        return self.model.backend

    @property
    def n_modes(self) -> int:
        """Number of estimated modal coefficients."""
        return self.basis.n_modes

    def __repr__(self) -> str:
        return (
            f"LIFT(n_modes={self.n_modes}, astigmatism={self.astigmatism * 1e9:.1f} nm RMS, "
            f"image_shape={self.model.image_shape}, sampling={self.model.sampling:.3g}, "
            f"read_noise={self.read_noise:g}, backend={self.backend})"
        )

    # ------------------------------------------------------------- problems
    def problem(self, image: Any) -> FocalPlaneProblem:
        """The :class:`~solvephase.FocalPlaneProblem` LIFT solves for ``image``."""
        image = _image_stack(image)
        loss: str | Loss
        weights = None
        if isinstance(self.loss, Loss):
            loss = self.loss
        elif self.loss.lower() == "poisson":
            loss = PoissonLoss(read_noise=self.read_noise)
        else:
            loss = GaussianLoss()
            host = np.asarray(self.backend.to_numpy(image), dtype=np.float64)
            weights = noise_weights(host - self.background, self.read_noise)
        return FocalPlaneProblem(
            self.model,
            image,
            basis=self.basis,
            loss=loss,
            weights=weights,
            fit_flux=self.fit_flux,
            fit_background=self.fit_background,
            background=self.background,
        )

    def estimate(
        self, image: Any, start: Any = None, *, max_iter: int | None = None, **options: Any
    ) -> Result:
        """Estimate the modal coefficients from one image.

        Parameters
        ----------
        image:
            ``(my, mx)`` image in photo-electrons (NumPy or backend array).
        start:
            Starting point: ``None`` (flat), coefficients in metres, an OPD map
            or a previous :class:`~solvephase.Result` (warm start in a loop).
        max_iter:
            Override the iteration limit.
        **options:
            Passed to the Levenberg-Marquardt solver (``ftol``, ``damping``).

        Returns
        -------
        Result
            ``method="lift"``; ``coefficients`` in metres RMS per mode, the
            known astigmatism is *not* included in ``opd``.
        """
        t0 = time.perf_counter()
        problem = self.problem(image)
        options.setdefault("ftol", 1e-9)
        n_iter = max_iter or self.max_iter
        result = solve(problem, method="lm", start=start, max_iter=n_iter, **options)
        swapped = False
        if self.resolve_twin and result.coefficients is not None:
            twin = self.twin(result.coefficients)
            if _norm(twin, self.basis) < _norm(result.coefficients, self.basis):
                # The twin fits the data equally well; prefer the smaller wavefront.
                result = solve(problem, method="lm", start=twin, max_iter=n_iter, **options)
                swapped = True
        result.method = "lift"
        result.extra["twin_swapped"] = swapped
        result.elapsed = time.perf_counter() - t0
        result.extra["astigmatism"] = self.astigmatism
        result.extra["diversity_opd"] = self.diversity
        return result

    # ----------------------------------------------------------- statistics
    def expected_image(self, coefficients: Any = None, *, photons: float = 1.0) -> Any:
        """Noise-free image ``photons * PSF + background`` (backend array).

        ``coefficients`` are in metres (default: zero wavefront).
        """
        if coefficients is None:
            coefficients = np.zeros(self.n_modes)
        opd = self.basis.synthesize(np.asarray(coefficients, dtype=np.float64))
        return photons * self.model.images(opd)[0] + self.background

    def fisher(self, coefficients: Any = None, *, photons: float) -> np.ndarray:
        """Fisher information matrix at ``coefficients`` (metres) for ``photons`` detected.

        Covers the modal coefficients and, when fitted, the nuisance
        parameters (log-flux, background), in that order, in SI units for the
        coefficients (m^-2). Pixel noise is shifted Poisson: variance
        ``model + read_noise^2``.
        """
        if photons <= 0:
            raise ValueError("photons must be positive")
        if coefficients is None:
            coefficients = np.zeros(self.n_modes)
        coeffs = np.asarray(coefficients, dtype=np.float64)
        if coeffs.shape != (self.n_modes,):
            raise ValueError(f"expected {self.n_modes} coefficients, got shape {coeffs.shape}")
        image = self.expected_image(coeffs, photons=photons)
        problem = FocalPlaneProblem(
            self.model,
            image,
            basis=self.basis,
            loss=PoissonLoss(read_noise=self.read_noise),
            fit_flux=self.fit_flux,
            fit_background=self.fit_background,
            background=self.background,
        )
        x = problem.initial(coeffs)
        _, _, hess = problem.gauss_newton(x)
        fisher = np.asarray(self.backend.to_numpy(hess), dtype=np.float64)
        k_wave = 2.0 * math.pi / self.model.wavelength
        scale = np.ones(fisher.shape[0])
        scale[problem.layout.coeffs] = k_wave  # d/d(metres) = k d/d(radians)
        return np.asarray(fisher * np.outer(scale, scale))

    def crlb(self, coefficients: Any = None, *, photons: float) -> np.ndarray:
        """Cramer-Rao lower bound: covariance ``(n_modes, n_modes)`` of the coefficients in m^2.

        Nuisance parameters (flux, background) are marginalized: the bound is
        the coefficient block of the inverse Fisher matrix.
        """
        fisher = self.fisher(coefficients, photons=photons)
        cov = np.linalg.inv(fisher)
        n = self.n_modes
        return np.asarray(0.5 * (cov[:n, :n] + cov[:n, :n].T))


def lift(
    image: Any,
    pupil: Pupil,
    wavelength: Any,
    *,
    sampling: float | None = None,
    pixel_scale: float | None = None,
    n_modes: int = 10,
    basis: Basis | None = None,
    astigmatism: float | None = None,
    astigmatism_mode: int = 5,
    read_noise: float = 0.0,
    loss: str | Loss = "poisson",
    fit_background: bool = False,
    background: float = 0.0,
    start: Any = None,
    max_iter: int = 20,
    device: BackendLike = "cpu",
    precision: str | None = None,
    **options: Any,
) -> Result:
    """Estimate low-order modes from one astigmatic focal-plane image (LIFT).

    Parameters
    ----------
    image:
        ``(my, mx)`` image in photo-electrons, taken with the known
        astigmatism in the beam, background subtracted (or set ``background``
        / ``fit_background``).
    pupil, wavelength, sampling, pixel_scale:
        The optics, see :class:`~solvephase.FocalPlaneModel`.
    n_modes, basis:
        Zernike modes from tip (Noll 2), or any modal basis.
    astigmatism:
        Known astigmatism, RMS OPD in metres (default ``wavelength / 10``).
    astigmatism_mode:
        5 (oblique) or 6 (vertical astigmatism).
    read_noise:
        Read noise in electrons RMS.
    loss:
        ``"poisson"`` (default, maximum likelihood) or ``"gaussian"``.
    start:
        Optional starting point (coefficients, OPD or Result).
    max_iter:
        Gauss-Newton iteration limit.
    device, precision:
        Backend selection.
    **options:
        Passed to the Levenberg-Marquardt solver.

    Returns
    -------
    Result
        With ``coefficients`` (metres RMS per mode) and ``opd`` (without the
        known astigmatism).

    For repeated estimates with the same optics use :class:`LIFT`, which
    builds the model once.
    """
    host_shape = tuple(getattr(image, "shape", np.shape(image)))[-2:]
    sensor = LIFT(
        pupil,
        wavelength,
        host_shape,
        sampling=sampling,
        pixel_scale=pixel_scale,
        n_modes=n_modes,
        basis=basis,
        astigmatism=astigmatism,
        astigmatism_mode=astigmatism_mode,
        read_noise=read_noise,
        loss=loss,
        fit_background=fit_background,
        background=background,
        max_iter=max_iter,
        device=device,
        precision=precision,
    )
    return sensor.estimate(image, start, **options)


def lift_crlb(
    pupil: Pupil,
    wavelength: Any,
    image_shape: Any,
    coefficients: Any = None,
    *,
    photons: float,
    read_noise: float = 0.0,
    sampling: float | None = None,
    pixel_scale: float | None = None,
    n_modes: int = 10,
    basis: Basis | None = None,
    astigmatism: float | None = None,
    astigmatism_mode: int = 5,
    background: float = 0.0,
    fit_flux: bool = True,
    device: BackendLike = "cpu",
    precision: str | None = None,
) -> np.ndarray:
    """Cramer-Rao lower bound on the LIFT modal coefficients.

    The Fisher matrix ``J^T diag(1 / (m + read_noise^2)) J`` is built from the
    model Jacobian ``J`` (forward-mode derivatives of the expected image ``m``
    with respect to the coefficients and the flux), at the true coefficients.
    Its inverse bounds the covariance of any unbiased estimator.

    Parameters
    ----------
    pupil, wavelength, image_shape, sampling, pixel_scale:
        The optics.
    coefficients:
        True coefficients in metres (default: zero wavefront).
    photons:
        Total detected photo-electrons in the PSF.
    read_noise:
        Read noise in electrons RMS.
    n_modes, basis, astigmatism, astigmatism_mode:
        As for :func:`lift`.
    background:
        Known background in electrons per pixel (adds photon noise).
    fit_flux:
        Whether the flux is an unknown (marginalized) parameter.
    device, precision:
        Backend selection (double precision is advisable for the inverse).

    Returns
    -------
    numpy.ndarray
        ``(n_modes, n_modes)`` covariance bound in m^2; the square root of its
        diagonal is the smallest achievable RMS error per mode.
    """
    sensor = LIFT(
        pupil,
        wavelength,
        image_shape,
        sampling=sampling,
        pixel_scale=pixel_scale,
        n_modes=n_modes,
        basis=basis,
        astigmatism=astigmatism,
        astigmatism_mode=astigmatism_mode,
        read_noise=read_noise,
        fit_flux=fit_flux,
        background=background,
        device=device,
        precision=precision,
    )
    return sensor.crlb(coefficients, photons=photons)
