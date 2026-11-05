"""Focal-plane focus sensing with a focus mechanism.

A focus mechanism (a mirror or detector on a piston stage, a refocusing lens)
adds a known pupil OPD per unit of travel, its *response* ``r(x, y)``.
Images of a point source recorded at mechanism positions ``p_k`` therefore
carry the known diversity ``r * (p_k - p_ref)``: this is phase diversity in
which the mechanism itself makes the diversity (Gonsalves 1982; Paxman,
Schulz & Fienup 1992), so every move that corrects focus is also a
diversity measurement for the next estimate. :class:`FocusSensor` retrieves
the wavefront from any number of such images by maximum likelihood (the
:class:`~solvephase.retrieval.FocalPlaneProblem` machinery with a Poisson +
read-noise loss, registration and, by default, a fitted detector/jitter
blur) and converts it into the mechanism position that minimizes the RMS
wavefront error.

**One image.** A single in-focus image cannot tell the sign of the even part
of the wavefront for a centro-symmetric pupil: the twin
``phi(x) -> -phi(-x)`` gives the same image (the reason LIFT adds a known
astigmatism; Meimon, Fusco & Mugnier 2010). A pupil that is *not*
centro-symmetric (an odd number of spider vanes, off-axis vignetting) breaks
the symmetry, as does any known aberration, so one image can fix the sign
(Hubble's focus monitoring uses single-image phase retrieval in the same way;
Krist & Burrows 1995; Lallo et al. 2006). :meth:`FocusSensor.estimate` always
also solves from the twin of its best solution and reports how much less
likely any distinct alternative is (:attr:`FocusEstimate.probability`), so a
caller knows when one image was enough.

**Best position.** The position minimizing the intensity-weighted RMS of
``phi + r * (p - p_ref)`` after removing piston, tip and tilt (or the modes
named by ``exclude``) is linear in the wavefront,
``p* = p_ref - <P r, P phi> / <P r, P r>``, so its variance follows from the
Fisher matrix of the fit (the Gauss-Newton matrix of the negative
log-likelihood at the solution). :func:`combine_estimates` gives the position
minimizing the weighted sum of RMS errors over several stars (field points)
seen at once.

**Calibration.** :meth:`FocusSensor.calibrate` fits one scale factor on the
response from a through-focus series (``fit_diversity_scale``), which is how
the stroke-to-OPD sensitivity of a mechanism is measured in place.

Units: positions are in the mechanism's own unit (metres, micrometres,
steps); the response is OPD in metres per unit of position; wavefronts are
OPD in metres; images are in photo-electrons.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..backend import BackendLike, to_numpy
from ..basis import Basis
from ..focal import FocalPlaneModel
from ..metrics import RemoveSpec, remove_modes
from ..pupil import Pupil
from ..result import Result
from ..retrieval import FocalPlaneProblem, solve

__all__ = [
    "FocusEstimate",
    "FocusSensor",
    "ResponseCalibration",
    "best_focus",
    "combine_estimates",
    "outlier_mask",
    "twin",
]


def twin(opd: Any) -> np.ndarray:
    """The twin wavefront ``-opd(-x, -y)``: same image as ``opd`` for a centro-symmetric pupil.

    Valid on solvephase's pixel-centred pupil grids, where ``-x`` is the
    mirrored index.
    """
    arr = np.asarray(to_numpy(opd), dtype=np.float64)
    return np.asarray(-arr[::-1, ::-1])


def _weighted_dot(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * a * b))


def best_focus(opd: Any, response: Any, pupil: Pupil, *, exclude: RemoveSpec = "tiptilt") -> float:
    """Mechanism offset minimizing the RMS of ``opd + response * offset``.

    Parameters
    ----------
    opd:
        ``(ny, nx)`` wavefront at the current position, metres.
    response:
        ``(ny, nx)`` OPD per unit of mechanism travel, metres per unit.
    pupil:
        The pupil (its intensity weights the RMS).
    exclude:
        Modes ignored by the RMS (they do not degrade the image or are
        corrected elsewhere): default piston, tip and tilt.

    Returns
    -------
    float
        The offset to add to the current position, in the mechanism's unit.
    """
    w = pupil.amplitude**2
    resp = np.asarray(to_numpy(response), dtype=np.float64)
    pr = remove_modes(resp, pupil, exclude)
    den = _weighted_dot(pr, pr, w)
    if not den > 1e-12 * _weighted_dot(resp, resp, w):
        raise ValueError("the response is entirely in the excluded modes; it cannot focus")
    return -_weighted_dot(pr, remove_modes(opd, pupil, exclude), w) / den


@dataclass
class FocusEstimate:
    """Wavefront and best mechanism position from :meth:`FocusSensor.estimate`.

    Attributes
    ----------
    result:
        The retrieval at :attr:`reference` (OPD, coefficients, blur, fit
        diagnostics).
    reference:
        Mechanism position the retrieved wavefront refers to.
    current:
        Mechanism position of the last image (where the mechanism is now).
    position:
        Best position (minimum RMS wavefront error over the pupil).
    sigma:
        One-sigma uncertainty of :attr:`position` from the Fisher matrix
        (statistical only; a wrong model adds bias).
    probability:
        Posterior probability of this solution against the distinct
        alternatives found (the twin start; equal priors). Near 1 when the
        data decide the sign of the even modes.
    alternatives:
        ``(position, loss)`` of each distinct alternative local solution.
    n_images:
        Images used.
    chi2:
        Reduced fit statistic ``2 * loss / n_pixels``: about 1 when the model
        explains the data to the noise; much larger means a wrong local
        solution or a model error (pupil, sampling, response), and the
        position should not be trusted.
    """

    result: Result
    reference: float
    current: float
    position: float
    sigma: float
    probability: float = 1.0
    alternatives: list[tuple[float, float]] = field(default_factory=list)
    n_images: int = 1
    chi2: float = math.nan
    _num: float = 0.0  # <P r, P phi> at the reference, for combine_estimates
    _den: float = 1.0  # <P r, P r>

    @property
    def move(self) -> float:
        """Offset from the current position to :attr:`position`."""
        return self.position - self.current

    def should_move(self, deadband: float, *, k_sigma: float = 2.0) -> bool:
        """Whether a move is worth one actuation of the mechanism.

        Move when the required offset exceeds both ``deadband`` (the focus
        error you are willing to leave, in position units) and ``k_sigma``
        times the estimate's own uncertainty (do not chase noise).
        """
        return abs(self.move) > max(deadband, k_sigma * self.sigma)

    def summary(self) -> str:
        lines = [
            f"best position {self.position:+.4g} (+/- {self.sigma:.2g}) from "
            f"{self.n_images} image(s); move {self.move:+.4g} from {self.current:+.4g}",
            f"  probability {self.probability:.4f}, chi2 {self.chi2:.3g}"
            + (
                f"; alternatives at {', '.join(f'{p:+.4g}' for p, _ in self.alternatives)}"
                if self.alternatives
                else ""
            ),
        ]
        if self.result.blur is not None:
            lines.append(f"  blur sigma {np.asarray(self.result.blur).round(3)} px")
        return "\n".join(lines)


@dataclass
class ResponseCalibration:
    """Scale of a mechanism's response fitted from a through-focus series.

    ``scale`` multiplies the response the sensor was built with;
    ``sensor.with_response_scale(scale)`` applies it.
    """

    scale: float
    sigma: float
    result: Result
    positions: np.ndarray


class FocusSensor:
    """Focus sensing from point-source images at known mechanism positions.

    Parameters
    ----------
    pupil:
        Entrance pupil (amplitude, including spiders and vignetting).
    wavelength, image_shape, pixel_scale, sampling, weights, oversample:
        As for :class:`~solvephase.FocalPlaneModel`.
    response:
        ``(ny, nx)`` pupil OPD per unit of mechanism travel (metres per
        unit). For a pure focus mechanism this is a defocus map times its
        sensitivity; a ray-traced response also carries the mechanism's
        field-dependent side effects.
    basis:
        Wavefront basis or number of Zernike modes from tip (default 21,
        i.e. Noll 2-22 through the secondary spherical aberration).
    blur:
        Starting or fixed Gaussian blur (pixels) of the detector and jitter.
    fit_blur:
        Fit the blur width (default True; see
        :class:`~solvephase.FocalPlaneProblem`).
    read_noise:
        Read noise in electrons RMS (Poisson + read-noise likelihood).
    fit_background:
        Fit a constant background per image (default True), or ``"plane"``
        to also fit a linear ramp (stray-light gradients, nearby sources).
    exclude:
        Modes ignored when choosing the best position (default piston, tip,
        tilt).
    capture:
        Half-range of a scan of pure-focus starting points, as the RMS
        focus OPD in metres (default: half the reference wavelength). The
        objective is evaluated on a grid of mechanism offsets and the
        solver starts from the best ones, which extends the capture range
        from about 0.2 to several waves of defocus. ``0`` disables it.
    capture_step:
        Spacing of that scan as RMS focus OPD in metres (default
        ``wavelength / 20``).
    device, precision:
        Backend selection.
    """

    def __init__(
        self,
        pupil: Pupil,
        wavelength: float | Sequence[float],
        image_shape: Any,
        response: Any,
        *,
        pixel_scale: float | None = None,
        sampling: float | None = None,
        weights: Sequence[float] | None = None,
        oversample: int = 1,
        basis: Basis | int = 21,
        blur: Any = 0.0,
        fit_blur: bool | str = True,
        read_noise: float = 0.0,
        fit_background: bool | str = True,
        exclude: RemoveSpec = "tiptilt",
        capture: float | None = None,
        capture_step: float | None = None,
        device: BackendLike = "cpu",
        precision: str | None = None,
    ) -> None:
        self.pupil = pupil
        self.wavelength = wavelength
        self.image_shape = image_shape
        resp = np.asarray(to_numpy(response), dtype=np.float64)
        if resp.shape != pupil.shape:
            raise ValueError(f"response {resp.shape} does not match the pupil {pupil.shape}")
        if not np.all(np.isfinite(resp)):
            raise ValueError("response contains non-finite values")
        self.response = np.where(pupil.mask, resp, 0.0)
        self.pixel_scale = pixel_scale
        self.sampling = sampling
        self.weights = weights
        self.oversample = oversample
        self.basis = Basis.zernike(pupil, int(basis)) if isinstance(basis, int) else basis
        self.blur = blur
        self.fit_blur = fit_blur
        self.read_noise = float(read_noise)
        self.fit_background = fit_background
        self.exclude = exclude
        self.device = device
        self.precision = precision
        w = pupil.amplitude**2
        self._w = w
        self._pr = remove_modes(self.response, pupil, exclude)
        self._den = _weighted_dot(self._pr, self._pr, w)
        if not self._den > 1e-12 * _weighted_dot(self.response, self.response, w):
            raise ValueError("the response is entirely in the excluded modes; it cannot focus")
        # d position / d coefficient (metres of RMS OPD) for the uncertainty.
        modes = self.basis.mode_maps()
        self._grad = np.array(
            [
                -_weighted_dot(self._pr, remove_modes(m, pupil, exclude), w) / self._den
                for m in modes
            ]
        )
        lam = float(np.average(np.atleast_1d(wavelength), weights=weights))
        self.capture = 0.5 * lam if capture is None else float(capture)
        self.capture_step = lam / 20.0 if capture_step is None else float(capture_step)
        if self.capture < 0 or self.capture_step <= 0:
            raise ValueError("capture must be >= 0 and capture_step > 0 (RMS OPD in metres)")

    # ----------------------------------------------------------- building
    def with_response_scale(self, scale: float) -> FocusSensor:
        """A copy whose response is multiplied by ``scale`` (e.g. a calibration)."""
        new = object.__new__(FocusSensor)
        new.__dict__.update(self.__dict__)
        new.response = self.response * float(scale)
        new._pr = self._pr * float(scale)
        new._den = self._den * float(scale) ** 2
        new._grad = self._grad / float(scale)
        return new

    def model(self, positions: Sequence[float], reference: float) -> FocalPlaneModel:
        """Forward model of images taken at ``positions``, wavefront at ``reference``."""
        offsets = np.asarray(positions, dtype=np.float64) - float(reference)
        diversity = [self.response * d if d else None for d in offsets]
        return FocalPlaneModel(
            self.pupil,
            self.wavelength,
            self.image_shape,
            pixel_scale=self.pixel_scale,
            sampling=self.sampling,
            weights=self.weights,
            diversity=diversity,
            oversample=self.oversample,
            blur=self.blur,
            device=self.device,
            precision=self.precision,
        )

    def problem(
        self,
        images: Any,
        positions: Sequence[float] | np.ndarray,
        reference: float,
        *,
        pixel_weights: Any = None,
        fit_diversity_scale: bool = False,
    ) -> FocalPlaneProblem:
        """The :class:`~solvephase.FocalPlaneProblem` behind :meth:`estimate`."""
        data = np.asarray(to_numpy(images), dtype=np.float64)
        if data.ndim == 2:
            data = data[None]
        positions = list(np.atleast_1d(np.asarray(positions, dtype=np.float64)))
        if len(positions) != data.shape[0]:
            raise ValueError(f"{data.shape[0]} image(s) but {len(positions)} position(s)")
        model = self.model(positions, reference)
        return FocalPlaneProblem(
            model,
            data,
            basis=self.basis,
            loss="poisson",
            read_noise=self.read_noise,
            weights=pixel_weights,
            fit_background=self.fit_background,
            fit_tilt=data.shape[0] > 1,
            fit_blur=self.fit_blur,
            blur=None if not self.fit_blur else (self.blur if np.any(self.blur) else None),
            fit_diversity_scale=fit_diversity_scale,
        )

    # ---------------------------------------------------------- estimating
    def best_position(self, opd: Any, reference: float) -> float:
        """Position minimizing the RMS of the wavefront ``opd`` measured at ``reference``."""
        phi = remove_modes(np.asarray(to_numpy(opd), dtype=np.float64), self.pupil, self.exclude)
        return float(reference) - _weighted_dot(self._pr, phi, self._w) / self._den

    @property
    def focus_rms_per_unit(self) -> float:
        """RMS focus OPD (metres, excluded modes removed) per unit of mechanism travel."""
        return math.sqrt(self._den / float(np.sum(self._w)))

    def _scan_starts(self, problem: FocalPlaneProblem, keep: int = 2) -> list[np.ndarray]:
        """The ``keep`` best pure-focus starting wavefronts on the capture grid."""
        if self.capture <= 0:
            return []
        unit = self.focus_rms_per_unit
        n = math.floor(self.capture / self.capture_step)
        offsets = np.arange(-n, n + 1) * self.capture_step / unit
        values = []
        for d in offsets:
            x = problem.initial(self.response * d)
            values.append(problem.value(x))
        order = np.argsort(values)
        picked: list[float] = []
        for i in order:  # best local minima of the scan, not neighbours of one another
            if all(abs(offsets[i] - q) > 1.5 * self.capture_step / unit for q in picked):
                picked.append(float(offsets[i]))
            if len(picked) == keep:
                break
        return [self.response * d for d in picked if d != 0.0]

    def _sigma(self, problem: FocalPlaneProblem, result: Result) -> float:
        """Fisher (Cramer-Rao) standard deviation of the best position at ``result``."""
        x = problem.initial(result)
        _, _, hess = problem.gauss_newton(x)
        h = np.asarray(to_numpy(hess), dtype=np.float64)
        cov = np.linalg.pinv(h, rcond=1e-12, hermitian=True)
        sl = problem.layout.coeffs
        k = 2.0 * math.pi / problem.model.wavelength
        cov_c = cov[sl, sl] / k**2  # m^2
        return float(math.sqrt(max(self._grad @ cov_c @ self._grad, 0.0)))

    def estimate(
        self,
        images: Any,
        positions: float | Sequence[float],
        *,
        reference: float | None = None,
        start: Any = None,
        pixel_weights: Any = None,
        twin_check: bool = True,
        max_iter: int | None = None,
    ) -> FocusEstimate:
        """Wavefront and best position from images at known mechanism positions.

        Parameters
        ----------
        images:
            ``(K, my, mx)`` (or ``(my, mx)``) images in photo-electrons,
            registered to within a few pixels (cut around the star).
        positions:
            Mechanism position of each image (the last is the current one).
        reference:
            Position the retrieved wavefront refers to (default: the current
            one).
        start:
            Optional starting wavefront: an OPD map at ``reference`` or a
            previous :class:`FocusEstimate` (re-referenced through the
            response). A flat start is always tried as well.
        pixel_weights:
            Per-pixel weights, zero to mask (saturation, cosmic rays).
        twin_check:
            Also solve from the twin of the best solution and report the
            alternative (always useful for one image; cheap insurance for
            more).
        max_iter:
            Iteration limit per start.
        """
        t0 = time.perf_counter()
        pos = np.atleast_1d(np.asarray(positions, dtype=np.float64))
        current = float(pos[-1])
        ref = current if reference is None else float(reference)
        problem = self.problem(images, pos, ref, pixel_weights=pixel_weights)
        starts: list[Any] = [None, *self._scan_starts(problem)]
        if isinstance(start, FocusEstimate):
            starts.insert(0, self._rereference(start, ref))
        elif start is not None:
            starts.insert(0, start)
        results = [solve(problem, method="lm", start=s, max_iter=max_iter) for s in starts]
        best = min(results, key=lambda r: r.loss)
        if twin_check:
            results.append(solve(problem, method="lm", start=twin(best.opd), max_iter=max_iter))
            best = min(results, key=lambda r: r.loss)
        position = self.best_position(best.opd, ref)
        # Distinct solutions differ in focus by more than 3 nm RMS.
        tol = 3e-9 / self.focus_rms_per_unit
        alternatives: list[tuple[float, float]] = []
        for r in results:
            p = self.best_position(r.opd, ref)
            distinct = all(abs(p - q) > tol for q, _ in alternatives)
            if r is not best and abs(p - position) > tol and distinct:
                alternatives.append((p, float(r.loss)))
        gaps = np.array([loss - best.loss for _, loss in alternatives])
        probability = float(1.0 / (1.0 + np.sum(np.exp(-np.clip(gaps, 0.0, 700.0)))))
        sigma = self._sigma(problem, best)
        best.elapsed = time.perf_counter() - t0
        phi = remove_modes(np.asarray(to_numpy(best.opd)), self.pupil, self.exclude)
        return FocusEstimate(
            result=best,
            reference=ref,
            current=current,
            position=position,
            sigma=sigma,
            probability=probability,
            alternatives=alternatives,
            n_images=int(pos.size),
            chi2=float(2.0 * best.loss / float(np.count_nonzero(to_numpy(problem.weights)))),
            _num=_weighted_dot(self._pr, phi, self._w),
            _den=self._den,
        )

    def _rereference(self, est: FocusEstimate, reference: float) -> Result:
        """``est``'s wavefront moved to ``reference`` through the response."""
        r = est.result
        opd = np.asarray(to_numpy(r.opd)) + self.response * (reference - est.reference)
        return Result(
            method=r.method,
            opd=opd,
            phase=None,
            amplitude=r.amplitude,
            pupil=r.pupil,
            wavelength=r.wavelength,
            blur=r.blur,
        )

    # --------------------------------------------------------- calibration
    def calibrate(
        self,
        images: Any,
        positions: Sequence[float],
        *,
        reference: float | None = None,
        start: Any = None,
        max_iter: int | None = None,
    ) -> ResponseCalibration:
        """Fit the scale of the response from a through-focus series of one star.

        Uses all images jointly (one common wavefront, one scale on every
        image's diversity, per-image registration and a common blur). The
        sign and shape of the response stay as given; spread the positions
        on both sides of focus over about +/- one wave of defocus for a
        well-conditioned fit.
        """
        pos = np.asarray(positions, dtype=np.float64)
        if pos.size < 2 or np.ptp(pos) == 0:
            raise ValueError("calibration needs images at two or more distinct positions")
        ref = float(np.mean(pos)) if reference is None else float(reference)
        problem = self.problem(images, pos, ref, fit_diversity_scale=True)
        starts = [None] if start is None else [start, None]
        results = [solve(problem, method="lm", start=s, max_iter=max_iter) for s in starts]
        best = min(results, key=lambda r: r.loss)
        x = problem.initial(best)
        _, _, hess = problem.gauss_newton(x)
        cov = np.linalg.pinv(np.asarray(to_numpy(hess), dtype=np.float64), hermitian=True)
        sl = problem.layout.diversity_scale
        sigma = float(math.sqrt(max(cov[sl, sl][0, 0], 0.0)))
        assert best.diversity_scale is not None
        return ResponseCalibration(
            scale=float(best.diversity_scale), sigma=sigma, result=best, positions=pos
        )


def outlier_mask(
    estimates: Sequence[FocusEstimate], reject: float = 5.0, floor: float = 0.0
) -> np.ndarray:
    """Stars whose own best position disagrees with the others.

    A star is an outlier when its best position is further from the median of
    all stars than ``max(floor, reject * 1.4826 * MAD)`` (mechanism units).
    A star that converged to a wrong solution (a local minimum far from focus,
    a corrupted stamp) otherwise drags a field-weighted combination by a large
    fraction of its error. Needs at least three stars; never rejects half or
    more of them.

    Returns
    -------
    Boolean array, True for the stars to drop.
    """
    pos = np.array([e.position for e in estimates], dtype=np.float64)
    if len(pos) < 3:
        return np.zeros(len(pos), dtype=bool)
    med = float(np.median(pos))
    dev = np.abs(pos - med)
    limit = max(float(floor), reject * 1.4826 * float(np.median(dev)))
    out = dev > limit
    if 2 * int(np.sum(out)) >= len(pos):
        return np.zeros(len(pos), dtype=bool)
    return out


def combine_estimates(
    estimates: Sequence[FocusEstimate],
    weights: Sequence[float] | None = None,
    *,
    reject: float | None = None,
    floor: float = 0.0,
) -> tuple[float, float]:
    """Best common position for several stars (field points) and its uncertainty.

    Minimizes ``sum_f w_f * RMS_f^2`` (each star's RMS after the move), the
    field-weighted mean-square wavefront error, which is the right target
    when one mechanism focuses the whole field. Equal weights by default.
    With ``reject``, stars flagged by :func:`outlier_mask` (``reject``,
    ``floor``) get zero weight.

    Returns
    -------
    position, sigma
    """
    if not estimates:
        raise ValueError("no estimates to combine")
    w = np.ones(len(estimates)) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (len(estimates),) or np.any(w < 0) or not np.any(w > 0):
        raise ValueError("weights must be non-negative, one per estimate")
    if reject is not None:
        w = np.where(outlier_mask(estimates, reject, floor), 0.0, w)
    den = np.array([e._den for e in estimates])
    num = np.array([e._num for e in estimates])
    ref = np.array([e.reference for e in estimates])
    sig = np.array([e.sigma for e in estimates])
    total = float(np.sum(w * den))
    position = float(np.sum(w * (den * ref - num)) / total)
    sigma = float(math.sqrt(np.sum((w * den * sig) ** 2)) / total)
    return position, sigma
