"""Wavefront parameterizations: modal and zonal bases over a pupil.

A basis maps a coefficient vector ``c`` to an OPD map ``B c`` over the
pupil's illuminated pixels. Modal bases (Zernike, Karhunen-Loeve, Fourier,
segment piston/tip/tilt, deformable-mirror influence functions) are dense
``(n_modes, n_valid)`` matrices; the zonal basis is the identity on the valid
pixels and never forms a matrix.

Units: modes are dimensionless shapes normalized to unit RMS over the pupil
(unless built from influence functions or arrays you supply), so a
coefficient is the RMS OPD of that mode in metres.

Zernike, KL and Fourier modes come from :mod:`aobasis`, evaluated at the
pupil's pixel centres, so a phase retrieval result maps directly onto the
basis used by the rest of the AO toolchain.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np

from .backend import Backend
from .pupil import Pupil

__all__ = ["Basis"]


def _pixel_positions(pupil: Pupil) -> np.ndarray:
    """``(n_valid, 2)`` ``(x, y)`` coordinates of the illuminated pixels (aobasis order)."""
    y, x = pupil.coordinates()
    mask = pupil.mask
    return np.column_stack((x[mask], y[mask]))


def _weighted_orthonormalize(modes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Gram-Schmidt (via QR) with inner product ``sum w a b / sum w``, order kept."""
    w = weights / weights.sum()
    sw = np.sqrt(w)
    q, r = np.linalg.qr((modes * sw).T)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    q = q * signs
    return np.asarray((q / sw[:, None]).T)


class Basis:
    """A linear wavefront parameterization over a :class:`~solvephase.pupil.Pupil`.

    Build one with a constructor (:meth:`zernike`, :meth:`kl`, :meth:`zonal`,
    ...), then use :meth:`synthesize` and :meth:`fit`. Solvers move the basis
    to their device with :meth:`on`.

    Attributes
    ----------
    pupil:
        The pupil whose valid pixels the modes are defined on.
    modes:
        ``(n_modes, n_valid)`` host matrix, or ``None`` for the zonal basis.
    labels:
        One short name per mode.
    variances:
        Optional prior variance of each coefficient in m^2 (KL eigenvalues),
        used for maximum a posteriori regularization.
    kind:
        ``"zernike"``, ``"kl"``, ``"zonal"``, ``"segments"``, ``"dm"``, ...
    """

    def __init__(
        self,
        pupil: Pupil,
        modes: np.ndarray | None,
        *,
        labels: Sequence[str] | None = None,
        variances: np.ndarray | None = None,
        kind: str = "custom",
    ) -> None:
        self.pupil = pupil
        self.kind = kind
        if modes is not None:
            modes = np.ascontiguousarray(modes, dtype=np.float64)
            if modes.ndim != 2 or modes.shape[1] != pupil.n_valid:
                raise ValueError(
                    f"modes must be (n_modes, n_valid={pupil.n_valid}), got {modes.shape}"
                )
            if not np.all(np.isfinite(modes)):
                raise ValueError("modes contain non-finite values")
        self.modes = modes
        n = self.n_modes
        self.labels = list(labels) if labels is not None else [f"{kind}{i}" for i in range(n)]
        if len(self.labels) != n:
            raise ValueError(f"got {len(self.labels)} labels for {n} modes")
        if variances is not None:
            variances = np.asarray(variances, dtype=np.float64)
            if variances.shape != (n,) or np.any(variances <= 0):
                raise ValueError("variances must be positive, one per mode")
        self.variances = variances
        self._device_cache: dict[Backend, Any] = {}

    # ----------------------------------------------------------- properties
    @property
    def is_zonal(self) -> bool:
        """Whether this is the identity (pixel) basis."""
        return self.modes is None

    @property
    def n_modes(self) -> int:
        """Number of coefficients."""
        return self.pupil.n_valid if self.modes is None else int(self.modes.shape[0])

    def __len__(self) -> int:
        return self.n_modes

    def __repr__(self) -> str:
        return f"Basis(kind={self.kind!r}, n_modes={self.n_modes}, pupil={self.pupil.name!r})"

    def mode_maps(self, indices: Any = None) -> np.ndarray:
        """Return modes as ``(k, ny, nx)`` maps (zero outside the pupil)."""
        if self.modes is None:
            raise ValueError("the zonal basis has no mode maps; use synthesize()")
        sel = self.modes if indices is None else self.modes[indices]
        sel = np.atleast_2d(sel)
        out = np.zeros((sel.shape[0], *self.pupil.shape))
        out[:, self.pupil.mask] = sel
        return out

    # --------------------------------------------------------- host algebra
    def synthesize(self, coefficients: Any) -> np.ndarray:
        """OPD map(s) ``B c`` for coefficient vector(s) ``(..., n_modes)`` (host)."""
        c = np.asarray(coefficients, dtype=np.float64)
        if c.shape[-1] != self.n_modes:
            raise ValueError(f"expected {self.n_modes} coefficients, got {c.shape[-1]}")
        values = c if self.modes is None else c @ self.modes
        out = np.zeros((*c.shape[:-1], *self.pupil.shape))
        out[..., self.pupil.mask] = values
        return out

    def fit(self, opd: Any, *, weights: Any = None, rcond: float = 1e-10) -> np.ndarray:
        """Least-squares coefficients of an OPD map over the pupil (host).

        ``weights`` defaults to the pupil intensity ``amplitude**2``.
        """
        values = np.asarray(opd, dtype=np.float64)[..., self.pupil.mask]
        if self.modes is None:
            return values
        w = (
            (self.pupil.amplitude**2)[self.pupil.mask]
            if weights is None
            else (np.asarray(weights, dtype=np.float64)[self.pupil.mask])
        )
        sw = np.sqrt(w)
        sol, *_ = np.linalg.lstsq((self.modes * sw).T, (values * sw).T, rcond=rcond)
        return np.asarray(sol.T)

    def project(self, opd: Any) -> np.ndarray:
        """Best approximation of an OPD map within the span of the basis."""
        return self.synthesize(self.fit(opd))

    # --------------------------------------------------------------- device
    def on(self, backend: Backend) -> Any:
        """The ``(n_modes, n_valid)`` mode matrix on ``backend`` (cached)."""
        if self.modes is None:
            raise ValueError("the zonal basis has no mode matrix")
        if backend not in self._device_cache:
            self._device_cache[backend] = backend.asarray(self.modes, dtype="real")
        return self._device_cache[backend]

    # ---------------------------------------------------------- combinators
    def __add__(self, other: Basis) -> Basis:
        return self.concatenate(other)

    def concatenate(self, *others: Basis) -> Basis:
        """Stack the modes of several bases defined on the same pupil."""
        parts = [self, *others]
        if any(b.is_zonal for b in parts):
            raise ValueError("the zonal basis cannot be concatenated with other bases")
        if any(
            b.pupil.shape != self.pupil.shape or b.pupil.n_valid != self.pupil.n_valid
            for b in parts
        ):
            raise ValueError("bases must share a pupil")
        modes = np.concatenate([b.modes for b in parts if b.modes is not None])
        labels = [lab for b in parts for lab in b.labels]
        variances = None
        if all(b.variances is not None for b in parts):
            variances = np.concatenate([b.variances for b in parts if b.variances is not None])
        return Basis(
            self.pupil,
            modes,
            labels=labels,
            variances=variances,
            kind="+".join(b.kind for b in parts),
        )

    def subset(self, indices: Any) -> Basis:
        """A basis made of some of these modes."""
        if self.modes is None:
            raise ValueError("cannot take a subset of the zonal basis")
        idx = np.arange(self.n_modes)[indices]
        return Basis(
            self.pupil,
            self.modes[idx],
            labels=[self.labels[i] for i in np.atleast_1d(idx)],
            variances=None if self.variances is None else self.variances[idx],
            kind=self.kind,
        )

    def orthonormalized(self) -> Basis:
        """Orthonormal over the pupil (intensity-weighted, unit RMS), same span and order."""
        if self.modes is None:
            return self
        w = (self.pupil.amplitude**2)[self.pupil.mask]
        modes = _weighted_orthonormalize(self.modes, w)
        return Basis(self.pupil, modes, labels=self.labels, kind=self.kind)

    # -------------------------------------------------------- constructors
    @classmethod
    def zonal(cls, pupil: Pupil) -> Basis:
        """One coefficient per illuminated pixel (OPD in metres)."""
        return cls(pupil, None, kind="zonal")

    @classmethod
    def from_maps(
        cls,
        pupil: Pupil,
        maps: Any,
        *,
        labels: Sequence[str] | None = None,
        variances: Any = None,
        kind: str = "custom",
    ) -> Basis:
        """Wrap ``(n_modes, ny, nx)`` mode maps (values outside the pupil are ignored)."""
        maps = np.asarray(maps, dtype=np.float64)
        if maps.ndim == 2:
            maps = maps[None]
        if maps.shape[1:] != pupil.shape:
            raise ValueError(f"mode maps {maps.shape[1:]} do not match the pupil {pupil.shape}")
        return cls(pupil, maps[:, pupil.mask], labels=labels, variances=variances, kind=kind)

    @classmethod
    def zernike(
        cls,
        pupil: Pupil,
        n_modes: int,
        *,
        start: int = 2,
        ordering: str = "noll",
        annular: bool | None = None,
        orthonormalize: bool = False,
    ) -> Basis:
        """Zernike polynomials (via :class:`aobasis.ZernikeBasisGenerator`).

        Parameters
        ----------
        n_modes:
            Number of modes returned.
        start:
            First index in ``ordering`` (Noll: 1 = piston, 2 = tip, 4 = defocus).
            The default 2 skips piston, which phase retrieval cannot see.
        ordering:
            ``"noll"``, ``"ansi"`` or ``"fringe"``.
        annular:
            Use annular Zernikes orthonormal over the obscured pupil. Defaults
            to True when the pupil has an :attr:`~Pupil.obscuration`.
        orthonormalize:
            Re-orthonormalize over the sampled pupil (spiders, gaps, edge
            pixels), which improves solver conditioning. Coefficients then
            belong to the orthonormalized modes, not the textbook polynomials.

        Mode labels are ``Z<j>`` in the chosen ordering, and each mode has unit
        RMS over the continuous (annular) disk, so a coefficient is RMS OPD in
        metres.
        """
        import aobasis

        if n_modes < 1:
            raise ValueError("n_modes must be >= 1")
        if start < 1:
            raise ValueError("start must be >= 1")
        obscuration = pupil.obscuration if (annular is None or annular) else 0.0
        positions = _pixel_positions(pupil)
        gen = aobasis.ZernikeBasisGenerator(
            positions, pupil_radius=pupil.diameter / 2.0, obscuration=obscuration
        )
        total = n_modes + start - 1
        with warnings.catch_warnings():
            # Anti-aliased rim pixels sit slightly outside the nominal radius; the
            # polynomial is evaluated at their true radius, which is what we want.
            warnings.simplefilter("ignore")
            m2c = gen.generate(total, ordering=ordering)
        modes = m2c[:, start - 1 :].T
        # aobasis counts every ordering from 1; ANSI indices conventionally start at 0.
        first = start - 1 if ordering == "ansi" else start
        labels = [f"Z{j}" for j in range(first, first + n_modes)]
        basis = cls(pupil, modes, labels=labels, kind="zernike")
        if orthonormalize:
            basis = basis.orthonormalized()
        return basis

    @classmethod
    def from_aobasis(
        cls,
        pupil: Pupil,
        generator: Any,
        n_modes: int,
        *,
        kind: str | None = None,
        **generate_kwargs: Any,
    ) -> Basis:
        """Evaluate any :mod:`aobasis` generator class at the pupil's pixels.

        ``generator`` is a generator class (e.g. ``aobasis.FourierBasisGenerator``)
        or a factory ``f(positions) -> generator``; it is constructed with the
        pupil's pixel positions in metres. Keywords go to ``generate``.
        """
        positions = _pixel_positions(pupil)
        gen = generator(positions)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m2c = gen.generate(n_modes, **generate_kwargs)
        name = kind or type(gen).__name__.replace("BasisGenerator", "").lower()
        return cls(pupil, np.asarray(m2c).T, kind=name)

    @classmethod
    def fourier(cls, pupil: Pupil, n_modes: int, *, orthonormalize: bool = True) -> Basis:
        """Sine/cosine modes (via :class:`aobasis.FourierBasisGenerator`), piston-free."""
        import aobasis

        basis = cls.from_aobasis(
            pupil,
            lambda pos: aobasis.FourierBasisGenerator(pos, pupil.diameter),
            n_modes,
            kind="fourier",
            ignore_piston=True,
            normalize="rms",
        )
        return basis.orthonormalized() if orthonormalize else basis

    @classmethod
    def kl(
        cls,
        pupil: Pupil,
        n_modes: int,
        *,
        r0: float = 0.15,
        outer_scale: float = 30.0,
        r0_wavelength: float = 500e-9,
        max_points: int = 2500,
    ) -> Basis:
        """Karhunen-Loeve modes of von Karman turbulence (via :mod:`aobasis`).

        The covariance is diagonalized on a coarse grid of at most
        ``max_points`` pupil samples and the modes are interpolated (cubic) to
        the full pupil, then re-orthonormalized. :attr:`variances` holds each
        coefficient's turbulence variance in m^2 (OPD), so the basis can act as
        a Bayesian prior.

        Parameters
        ----------
        r0:
            Fried parameter in metres at ``r0_wavelength``.
        outer_scale:
            Outer scale in metres (``numpy.inf`` for Kolmogorov).
        """
        import aobasis
        from scipy.interpolate import RegularGridInterpolator

        ny, nx = pupil.shape
        factor = 1
        while pupil.n_valid / factor**2 > max_points:
            factor += 1
        coarse_n = math.ceil(max(ny, nx) / factor)
        coarse_pitch = pupil.pitch * max(ny, nx) / coarse_n
        from .propagation import centered_coordinates

        cy = centered_coordinates(coarse_n, coarse_pitch)
        cx = centered_coordinates(coarse_n, coarse_pitch)
        yy, xx = np.meshgrid(cy, cx, indexing="ij")
        r_out = pupil.diameter / 2.0 + coarse_pitch
        r_in = max(0.0, pupil.obscuration * pupil.diameter / 2.0 - coarse_pitch)
        rr = np.hypot(xx, yy)
        coarse_mask = (rr <= r_out) & (rr >= r_in)
        positions = np.column_stack((xx[coarse_mask], yy[coarse_mask]))
        gen = aobasis.KLBasisGenerator(
            positions,
            fried_parameter=r0,
            outer_scale=outer_scale,
            r0_wavelength=r0_wavelength,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m2c = gen.generate(n_modes, ignore_piston=True, normalize="rms")
        # With unit-RMS modes, aobasis eigenvalues are the coefficient variances in
        # rad^2 of phase at r0_wavelength.
        eig_rad2 = np.asarray(gen.eigenvalues, dtype=np.float64)[:n_modes]
        if factor == 1 and coarse_mask.sum() == pupil.n_valid:
            fine = m2c.T
        else:
            y, x = pupil.coordinates()
            pts = np.column_stack((y[pupil.mask], x[pupil.mask]))
            fine = np.empty((n_modes, pupil.n_valid))
            grid = np.zeros((coarse_n, coarse_n))
            for k in range(n_modes):
                grid[:] = 0.0
                grid[coarse_mask] = m2c[:, k]
                interp = RegularGridInterpolator(
                    (cy, cx), grid, method="cubic", bounds_error=False, fill_value=None
                )
                fine[k] = interp(pts)
        # Re-orthonormalize on the full pupil (unit RMS, so coefficients are RMS
        # OPD in metres) and convert the variances from rad^2 to m^2 of OPD.
        w = (pupil.amplitude**2)[pupil.mask]
        fine = _weighted_orthonormalize(fine, w)
        opd_var = eig_rad2 * (r0_wavelength / (2 * np.pi)) ** 2
        labels = [f"KL{i + 1}" for i in range(n_modes)]
        return cls(pupil, fine, labels=labels, variances=opd_var, kind="kl")

    @classmethod
    def segments(cls, pupil: Pupil, terms: Sequence[str] = ("piston", "tip", "tilt")) -> Basis:
        """Per-segment piston, tip and tilt for a segmented pupil.

        Each mode is non-zero on one segment and unit RMS over it (piston = 1,
        tip/tilt are ramps in x/y about the segment centroid). Requires
        :attr:`Pupil.segments`.
        """
        if pupil.segments is None:
            raise ValueError("this pupil has no segment labels")
        allowed = {"piston", "tip", "tilt"}
        if not set(terms) <= allowed:
            raise ValueError(f"terms must be drawn from {sorted(allowed)}")
        seg = pupil.segments[pupil.mask]
        y, x = pupil.coordinates()
        y, x = y[pupil.mask], x[pupil.mask]
        w = (pupil.amplitude**2)[pupil.mask]
        modes, labels = [], []
        for s in range(1, pupil.n_segments + 1):
            on = seg == s
            if not np.any(on):
                continue
            ws = w * on
            cx = np.sum(ws * x) / ws.sum()
            cy = np.sum(ws * y) / ws.sum()
            for term in terms:
                if term == "piston":
                    m = on.astype(np.float64)
                elif term == "tip":
                    m = np.where(on, x - cx, 0.0)
                else:
                    m = np.where(on, y - cy, 0.0)
                rms = math.sqrt(np.sum(ws * m**2) / ws.sum())
                modes.append(m / rms)
                labels.append(f"S{s}-{term}")
        return cls(pupil, np.array(modes), labels=labels, kind="segments")

    @classmethod
    def influence_functions(
        cls,
        pupil: Pupil,
        influence: Any,
        *,
        m2c: Any = None,
        labels: Sequence[str] | None = None,
    ) -> Basis:
        """Deformable-mirror basis: coefficients are DM commands.

        Parameters
        ----------
        influence:
            ``(n_actuators, ny, nx)`` influence-function OPD maps in metres per
            unit command (zero outside the pupil is not required).
        m2c:
            Optional ``(n_actuators, n_modes)`` modal-to-command matrix (e.g.
            from :mod:`aobasis`); coefficients are then modal commands.
        """
        ifs = np.asarray(influence, dtype=np.float64)
        if ifs.ndim != 3 or ifs.shape[1:] != pupil.shape:
            raise ValueError(f"influence must be (n_actuators, *{pupil.shape}), got {ifs.shape}")
        modes = ifs[:, pupil.mask]
        if m2c is not None:
            modes = np.asarray(m2c, dtype=np.float64).T @ modes
        return cls(pupil, modes, labels=labels, kind="dm")

    @classmethod
    def gaussian_dm(
        cls,
        pupil: Pupil,
        actuators_across: int,
        *,
        coupling: float = 0.15,
        stroke: float = 1e-6,
    ) -> Basis:
        """Square-grid DM with Gaussian influence functions (via :mod:`aobasis`).

        Actuators are on a ``actuators_across``-wide square grid spanning the
        pupil diameter (rim actuators on the edge), keeping those within one
        pitch of the pupil. A unit command gives ``stroke`` metres of OPD at
        the actuator.
        """
        import aobasis

        grid = aobasis.make_circular_actuator_grid(pupil.diameter, actuators_across)
        pitch = pupil.diameter / (actuators_across - 1)
        r = np.hypot(grid[:, 0], grid[:, 1])
        grid = grid[r <= pupil.diameter / 2.0 + pitch]
        points = _pixel_positions(pupil)
        ifs = aobasis.gaussian_influence_functions(grid, points, coupling=coupling, pitch=pitch)
        labels = [f"act{i}" for i in range(grid.shape[0])]
        basis = cls(pupil, (ifs * stroke).T, labels=labels, kind="dm")
        basis.actuator_positions = grid  # type: ignore[attr-defined]
        return basis
