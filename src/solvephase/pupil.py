"""Entrance-pupil amplitude maps.

A :class:`Pupil` is a sampled, real, non-negative amplitude transmission on a
square-pixel grid, with the physical metadata every solver needs: pixel pitch,
reference diameter (which defines lambda/D and the Zernike radius), central
obscuration ratio, and optionally a segment label image for segmented
telescopes.

Analytic constructors anti-alias edges by supersampling each pixel, so the
amplitude at the rim is the fraction of the pixel that transmits. Coordinates
follow :func:`solvephase.propagation.centered_coordinates`: arrays are
``(y, x)`` and the optical axis sits midway across the grid.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .propagation import centered_coordinates

__all__ = ["Pupil"]

MaskFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _supersampled(n: int, pitch: float, inside: MaskFunction, supersample: int) -> np.ndarray:
    """Average a boolean ``inside(y, x)`` test over ``supersample^2`` points per pixel."""
    if supersample < 1:
        raise ValueError("supersample must be >= 1")
    coords = centered_coordinates(n, pitch)
    offsets = ((np.arange(supersample) + 0.5) / supersample - 0.5) * pitch
    out = np.zeros((n, n), dtype=np.float64)
    for oy in offsets:
        yy = (coords + oy)[:, None]
        for ox in offsets:
            xx = (coords + ox)[None, :]
            out += inside(np.broadcast_to(yy, (n, n)), np.broadcast_to(xx, (n, n)))
    return out / supersample**2


def _rotate(y: np.ndarray, x: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    if angle_deg == 0.0:
        return y, x
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return -x * s + y * c, x * c + y * s


def _spider_test(
    n_spiders: int, width: float, angle_deg: float, offset: float = 0.0
) -> MaskFunction:
    """Return a test that is True on (thin, radial, centred) spider vanes."""

    def blocked(y: np.ndarray, x: np.ndarray) -> np.ndarray:
        out = np.zeros(np.broadcast(y, x).shape, dtype=bool)
        for k in range(n_spiders):
            a = math.radians(angle_deg + 360.0 * k / n_spiders)
            ca, sa = math.cos(a), math.sin(a)
            along = x * ca + y * sa
            across = -x * sa + y * ca
            out |= (np.abs(across - offset) <= width / 2.0) & (along >= 0.0)
        return out

    return blocked


def _hexagon_test(width: float, angle_deg: float) -> MaskFunction:
    """Regular hexagon of flat-to-flat ``width``; edge normals at ``angle + k 60 deg``."""
    normals = [
        (math.cos(math.radians(angle_deg + 60.0 * k)), math.sin(math.radians(angle_deg + 60.0 * k)))
        for k in range(3)
    ]

    def inside(y: np.ndarray, x: np.ndarray) -> np.ndarray:
        ok = np.ones(np.broadcast(y, x).shape, dtype=bool)
        for cx, cy in normals:
            ok &= np.abs(x * cx + y * cy) <= width / 2.0
        return ok

    return inside


def _hex_lattice(rings: int, spacing: float, angle_deg: float) -> list[tuple[int, float, float]]:
    """Hexagonal lattice sites within ``rings`` of the centre: (ring, x, y)."""
    a1 = math.radians(angle_deg)
    a2 = math.radians(angle_deg + 60.0)
    e1 = (spacing * math.cos(a1), spacing * math.sin(a1))
    e2 = (spacing * math.cos(a2), spacing * math.sin(a2))
    sites = []
    for i in range(-rings, rings + 1):
        for j in range(-rings, rings + 1):
            ring = max(abs(i), abs(j), abs(i + j))
            if ring <= rings:
                sites.append((ring, i * e1[0] + j * e2[0], i * e1[1] + j * e2[1]))
    sites.sort(key=lambda s: (s[0], math.atan2(s[2], s[1]) % (2 * math.pi)))
    return sites


@dataclass(frozen=True, eq=False)
class Pupil:
    """A sampled entrance pupil.

    Attributes
    ----------
    amplitude:
        ``(ny, nx)`` real amplitude transmission, ``>= 0`` (host NumPy).
    pitch:
        Pixel pitch in metres.
    diameter:
        Reference diameter in metres. It sets ``lambda / D`` and the unit
        radius of Zernike polynomials.
    obscuration:
        Central obscuration as a fraction of ``diameter`` (used by annular
        Zernikes; ``0`` for an unobscured pupil).
    segments:
        Optional ``(ny, nx)`` integer label image: ``0`` outside, ``1..S`` for
        each segment of a segmented aperture.
    name:
        A short descriptive label.
    """

    amplitude: np.ndarray
    pitch: float
    diameter: float
    obscuration: float = 0.0
    segments: np.ndarray | None = field(default=None)
    name: str = "custom"

    def __post_init__(self) -> None:
        amp = np.asarray(self.amplitude, dtype=np.float64)
        if amp.ndim != 2:
            raise ValueError(f"pupil amplitude must be 2-D, got shape {amp.shape}")
        if not np.all(np.isfinite(amp)) or np.any(amp < 0):
            raise ValueError("pupil amplitude must be finite and non-negative")
        if not np.any(amp > 0):
            raise ValueError("pupil amplitude is zero everywhere")
        if not (self.pitch > 0 and math.isfinite(self.pitch)):
            raise ValueError(f"pitch must be positive and finite, got {self.pitch}")
        if not (self.diameter > 0 and math.isfinite(self.diameter)):
            raise ValueError(f"diameter must be positive and finite, got {self.diameter}")
        if not 0.0 <= self.obscuration < 1.0:
            raise ValueError(f"obscuration must be in [0, 1), got {self.obscuration}")
        amp.setflags(write=False)
        object.__setattr__(self, "amplitude", amp)
        if self.segments is not None:
            seg = np.asarray(self.segments)
            if seg.shape != amp.shape or seg.dtype.kind not in "iu":
                raise ValueError("segments must be an integer label image matching amplitude")
            seg = seg.astype(np.int64)
            seg.setflags(write=False)
            object.__setattr__(self, "segments", seg)

    # ------------------------------------------------------------ properties
    @property
    def shape(self) -> tuple[int, int]:
        """Grid shape ``(ny, nx)``."""
        return (int(self.amplitude.shape[0]), int(self.amplitude.shape[1]))

    @property
    def mask(self) -> np.ndarray:
        """Boolean support: pixels with non-zero amplitude."""
        return np.asarray(self.amplitude > 0)

    @property
    def n_valid(self) -> int:
        """Number of pixels in :attr:`mask`."""
        return int(np.count_nonzero(self.mask))

    @property
    def area(self) -> float:
        """Transmitted area ``sum(amplitude^2) * pitch^2`` in square metres."""
        return float(np.sum(self.amplitude**2) * self.pitch**2)

    @property
    def n_segments(self) -> int:
        """Number of labelled segments (0 for a monolithic pupil)."""
        return 0 if self.segments is None else int(self.segments.max())

    def coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(y, x)`` coordinate grids in metres."""
        ny, nx = self.shape
        y = centered_coordinates(ny, self.pitch)[:, None]
        x = centered_coordinates(nx, self.pitch)[None, :]
        return np.broadcast_to(y, self.shape), np.broadcast_to(x, self.shape)

    def lambda_over_d(self, wavelength: float) -> float:
        """``wavelength / diameter`` in radians."""
        return float(wavelength) / self.diameter

    def __repr__(self) -> str:
        ny, nx = self.shape
        return (
            f"Pupil(name={self.name!r}, shape=({ny}, {nx}), diameter={self.diameter:g} m, "
            f"pitch={self.pitch:.4g} m, obscuration={self.obscuration:g}, "
            f"segments={self.n_segments})"
        )

    # ---------------------------------------------------------- manipulation
    def with_amplitude(self, amplitude: Any) -> Pupil:
        """Copy with a different amplitude map on the same grid."""
        return replace(self, amplitude=np.asarray(amplitude, dtype=np.float64))

    def padded(self, n: int) -> Pupil:
        """Embed the pupil centred in a larger ``n x n`` grid (same pitch)."""
        ny, nx = self.shape
        if n < max(ny, nx):
            raise ValueError(f"cannot pad a {self.shape} pupil to {n}")
        if (n - ny) % 2 or (n - nx) % 2:
            raise ValueError("padding must add the same number of pixels on each side")
        py, px = (n - ny) // 2, (n - nx) // 2
        amp = np.zeros((n, n))
        amp[py : py + ny, px : px + nx] = self.amplitude
        seg = None
        if self.segments is not None:
            seg = np.zeros((n, n), dtype=np.int64)
            seg[py : py + ny, px : px + nx] = self.segments
        return replace(self, amplitude=amp, segments=seg)

    def downsampled(self, factor: int) -> Pupil:
        """Average ``factor x factor`` pixel blocks (pitch grows by ``factor``)."""
        ny, nx = self.shape
        if factor < 1 or ny % factor or nx % factor:
            raise ValueError(f"factor {factor} must divide the grid shape {self.shape}")
        amp = self.amplitude.reshape(ny // factor, factor, nx // factor, factor).mean(axis=(1, 3))
        seg = None
        if self.segments is not None:
            blocks = self.segments.reshape(ny // factor, factor, nx // factor, factor)
            seg = blocks.max(axis=(1, 3))
        return replace(self, amplitude=amp, pitch=self.pitch * factor, segments=seg)

    # ---------------------------------------------------------- constructors
    @classmethod
    def from_array(
        cls,
        amplitude: Any,
        *,
        diameter: float | None = None,
        pitch: float | None = None,
        obscuration: float = 0.0,
        segments: Any = None,
        name: str = "custom",
    ) -> Pupil:
        """Wrap an existing amplitude map.

        Give ``diameter`` (the grid spans it across the largest illuminated
        extent if ``pitch`` is omitted), ``pitch``, or both. With neither, the
        grid width is taken as 1 m.
        """
        amp = np.asarray(amplitude, dtype=np.float64)
        if amp.ndim != 2:
            raise ValueError(f"pupil amplitude must be 2-D, got shape {amp.shape}")
        if pitch is None and diameter is None:
            pitch = 1.0 / max(amp.shape)
        if pitch is None:
            assert diameter is not None
            pitch = diameter / _illuminated_extent(amp)
        if diameter is None:
            diameter = _illuminated_extent(amp) * pitch
        return cls(
            amp,
            pitch=float(pitch),
            diameter=float(diameter),
            obscuration=obscuration,
            segments=segments,
            name=name,
        )

    @classmethod
    def circular(
        cls,
        n: int,
        diameter: float = 1.0,
        *,
        obscuration: float = 0.0,
        spiders: int = 0,
        spider_width: float = 0.0,
        spider_angle: float = 0.0,
        fill: float = 1.0,
        supersample: int = 4,
        name: str | None = None,
    ) -> Pupil:
        """Circular, optionally obscured pupil with radial spider vanes.

        Parameters
        ----------
        n:
            Grid size in pixels (``n x n``).
        diameter:
            Outer diameter in metres.
        obscuration:
            Central obscuration diameter as a fraction of ``diameter``.
        spiders:
            Number of radial vanes, evenly spaced from ``spider_angle``.
        spider_width:
            Vane width in metres.
        spider_angle:
            Angle of the first vane in degrees, counter-clockwise from +x.
        fill:
            Fraction of the grid width the diameter spans (``<= 1``).
        supersample:
            Sub-samples per pixel axis for anti-aliased edges (1 = binary).
        """
        if not 0 < fill <= 1:
            raise ValueError("fill must be in (0, 1]")
        pitch = diameter / (n * fill)
        r_out, r_in = diameter / 2.0, obscuration * diameter / 2.0
        vanes = _spider_test(spiders, spider_width, spider_angle) if spiders else None

        def inside(y: np.ndarray, x: np.ndarray) -> np.ndarray:
            r = np.hypot(x, y)
            ok = (r <= r_out) & (r >= r_in)
            if vanes is not None:
                ok &= ~vanes(y, x)
            return ok

        amp = _supersampled(n, pitch, inside, supersample)
        label = name or ("annular" if obscuration else "circular")
        return cls(amp, pitch=pitch, diameter=diameter, obscuration=obscuration, name=label)

    @classmethod
    def segmented_hexagonal(
        cls,
        n: int,
        *,
        rings: int,
        segment_size: float,
        gap: float = 0.0,
        missing_center: bool = True,
        obscuration: float = 0.0,
        spiders: int = 0,
        spider_width: float = 0.0,
        spider_angle: float = 90.0,
        orientation: float = 0.0,
        fill: float = 1.0,
        supersample: int = 4,
        name: str = "segmented",
    ) -> Pupil:
        """Hexagonally segmented aperture (Keck-, JWST-, ELT-like).

        Parameters
        ----------
        rings:
            Number of segment rings around the centre.
        segment_size:
            Segment flat-to-flat width in metres.
        gap:
            Gap between neighbouring segments in metres.
        missing_center:
            Leave out the central segment.
        obscuration:
            Circular central obscuration as a fraction of the circumscribed
            diameter.
        orientation:
            Rotation of the segment lattice in degrees.

        Notes
        -----
        The returned pupil's :attr:`~Pupil.segments` labels the segments
        ``1..S``, ordered by ring and then angle; :attr:`~Pupil.diameter` is the
        circumscribed diameter.
        """
        spacing = segment_size + gap
        sites = [
            s for s in _hex_lattice(rings, spacing, orientation) if s[0] > 0 or not missing_center
        ]
        hexagon = _hexagon_test(segment_size, orientation)
        circ = segment_size / math.sqrt(3.0)  # centre-to-corner distance
        diameter = 2.0 * max(math.hypot(x, y) + circ for _, x, y in sites)
        pitch = diameter / (n * fill)
        r_in = obscuration * diameter / 2.0
        vanes = _spider_test(spiders, spider_width, spider_angle) if spiders else None

        def blocked(y: np.ndarray, x: np.ndarray) -> np.ndarray:
            out = np.zeros(np.broadcast(y, x).shape, dtype=bool)
            if r_in > 0:
                out |= np.hypot(x, y) < r_in
            if vanes is not None:
                out |= vanes(y, x)
            return out

        amp: np.ndarray = np.zeros((n, n))
        labels = np.zeros((n, n), dtype=np.int64)
        coords = centered_coordinates(n, pitch)
        for index, (_, sx, sy) in enumerate(sites, start=1):

            def inside(y: np.ndarray, x: np.ndarray, sx: float = sx, sy: float = sy) -> np.ndarray:
                return hexagon(y - sy, x - sx) & ~blocked(y, x)

            part = _supersampled(n, pitch, inside, supersample)
            amp += part
            centre = hexagon(coords[:, None] - sy, coords[None, :] - sx)
            labels[(part > 0) & ((labels == 0) | centre)] = index
        amp = np.clip(amp, 0.0, 1.0)
        labels[amp <= 0] = 0
        return cls(
            amp,
            pitch=pitch,
            diameter=diameter,
            obscuration=obscuration,
            segments=labels,
            name=name,
        )

    # ----------------------------------------------------- telescope presets
    @classmethod
    def keck(cls, n: int = 256, *, supersample: int = 4) -> Pupil:
        """Keck-like primary: 36 pointy-top hexagonal segments, 10.95 m across corners.

        The segment pitch is 10.95 / 7 m (1.8 m corner to corner) with 3 mm
        gaps; six 26 mm support arms at 30 + 60 k degrees and a 2.57 m circular
        central obscuration. :attr:`diameter` is the conventional 10.95 m (the
        circumscribed circle is 11.19 m). This follows the geometry of makewfs' Keck HAKA
        example, without its fitted hexagonal secondary shadow and offset.
        Good for algorithm development, not instrument modelling.
        """
        gap, size = 0.003, 10.95 / 7.0
        circ = size / math.sqrt(3.0)
        circumscribed = 2.0 * max(math.hypot(x, y) + circ for _, x, y in _hex_lattice(3, size, 0.0))
        pupil = cls.segmented_hexagonal(
            n,
            rings=3,
            segment_size=size - gap,
            gap=gap,
            obscuration=2.5746 / circumscribed,
            spiders=6,
            spider_width=0.026,
            spider_angle=30.0,
            orientation=0.0,
            supersample=supersample,
            name="keck",
        )
        # Report the conventional 10.95 m diameter (it sets lambda/D).
        return replace(pupil, diameter=10.95, obscuration=2.5746 / 10.95)

    @classmethod
    def jwst(cls, n: int = 256, *, supersample: int = 4) -> Pupil:
        """JWST-like primary: 18 hexagonal segments of 1.32 m with three struts.

        Approximate geometry (7 mm gaps, 0.74 m central obscuration,
        three 0.1 m struts) for algorithm development.
        """
        return cls.segmented_hexagonal(
            n,
            rings=2,
            segment_size=1.32,
            gap=0.007,
            obscuration=0.74 / 6.6,
            spiders=3,
            spider_width=0.1,
            spider_angle=90.0,
            orientation=90.0,
            supersample=supersample,
            name="jwst",
        )

    @classmethod
    def vlt(cls, n: int = 256, *, supersample: int = 4) -> Pupil:
        """VLT Unit Telescope: 8.0 m with a 1.12 m obscuration and four vanes."""
        return cls.circular(
            n,
            8.0,
            obscuration=1.12 / 8.0,
            spiders=4,
            spider_width=0.04,
            spider_angle=45.0,
            supersample=supersample,
            name="vlt",
        )

    @classmethod
    def hst(cls, n: int = 256, *, supersample: int = 4) -> Pupil:
        """Hubble-like: 2.4 m, 0.33 obscuration and four 2.6 cm vanes."""
        return cls.circular(
            n,
            2.4,
            obscuration=0.33,
            spiders=4,
            spider_width=0.026,
            spider_angle=45.0,
            supersample=supersample,
            name="hst",
        )


def _illuminated_extent(amp: np.ndarray) -> float:
    """Width in pixels of the illuminated region along its larger axis."""
    rows = np.flatnonzero(np.any(amp > 0, axis=1))
    cols = np.flatnonzero(np.any(amp > 0, axis=0))
    if rows.size == 0:
        raise ValueError("pupil amplitude is zero everywhere")
    return float(max(rows[-1] - rows[0] + 1, cols[-1] - cols[0] + 1))
