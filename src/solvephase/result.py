"""The :class:`Result` returned by every solver."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import numpy as np

from .backend import to_numpy
from .basis import Basis
from .metrics import RemoveSpec, rms
from .pupil import Pupil

__all__ = ["Result"]


@dataclass
class Result:
    """Outcome of a phase retrieval.

    Array fields are backend arrays (CuPy for GPU solves) until you call
    :meth:`to_numpy`. Unknown or not-applicable fields are ``None``.

    Attributes
    ----------
    method:
        Algorithm name.
    opd:
        ``(ny, nx)`` retrieved OPD in metres, zero outside the pupil.
    phase:
        The same wavefront as phase in radians at :attr:`wavelength`.
    amplitude:
        ``(ny, nx)`` pupil amplitude (the known pupil unless it was fitted).
    pupil:
        The pupil the wavefront is defined on.
    wavelength:
        Reference wavelength of :attr:`phase`, in metres.
    coefficients:
        Modal coefficients (RMS OPD in metres per unit-RMS mode) when a modal
        basis was used.
    basis:
        The basis :attr:`coefficients` refer to.
    model_images:
        ``(K, my, mx)`` model of the data at the solution, in data units.
    flux, background:
        Per-channel fitted flux (data units, full PSF) and background (data
        units per pixel).
    tilts:
        ``(K, 2)`` per-channel (tip, tilt) OPD RMS in metres added to the
        common wavefront (registration), when fitted.
    loss:
        Final objective value.
    history:
        Objective value per iteration.
    times:
        Seconds since the start per :attr:`history` entry.
    n_iter:
        Iterations performed.
    converged:
        Whether a convergence criterion stopped the solver.
    message:
        Why it stopped.
    elapsed:
        Total wall-clock seconds, including setup.
    device:
        ``"cpu"`` or ``"gpu"``.
    extra:
        Algorithm-specific outputs (object estimate, support, error metrics ...).
    """

    method: str
    opd: Any
    phase: Any
    amplitude: Any
    pupil: Pupil | None
    wavelength: float
    coefficients: np.ndarray | None = None
    basis: Basis | None = None
    model_images: Any = None
    flux: np.ndarray | None = None
    background: np.ndarray | None = None
    tilts: np.ndarray | None = None
    loss: float = math.nan
    history: list[float] = field(default_factory=list)
    times: list[float] = field(default_factory=list)
    n_iter: int = 0
    converged: bool = False
    message: str = ""
    elapsed: float = 0.0
    device: str = "cpu"
    extra: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- views
    def to_numpy(self) -> Result:
        """Copy with every array field on the host."""
        changes: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in ("pupil", "basis"):
                continue
            if hasattr(value, "shape") and not isinstance(value, np.ndarray):
                changes[f.name] = to_numpy(value)
        extra = {k: (to_numpy(v) if hasattr(v, "shape") else v) for k, v in self.extra.items()}
        changes["extra"] = extra
        return replace(self, **changes)

    def rms(self, remove: RemoveSpec = "piston") -> float:
        """RMS of the retrieved OPD in metres over the pupil (see :func:`~solvephase.rms`)."""
        if self.pupil is None:
            raise ValueError("this result has no pupil")
        return rms(self.opd, self.pupil, remove)

    def residuals(self, images: Any) -> np.ndarray:
        """``images - model_images`` on the host."""
        if self.model_images is None:
            raise ValueError("this result has no model images")
        return np.asarray(to_numpy(images)) - np.asarray(to_numpy(self.model_images))

    def summary(self) -> str:
        """A short human-readable report."""
        lines = [f"solvephase {self.method} on {self.device}: {self.message or 'done'}"]
        lines.append(
            f"  iterations {self.n_iter}, converged {self.converged}, "
            f"{self.elapsed * 1e3:.1f} ms, final loss {self.loss:.6g}"
        )
        if self.pupil is not None and self.opd is not None:
            lines.append(f"  wavefront RMS (piston removed) {self.rms() * 1e9:.2f} nm")
        if self.coefficients is not None and self.basis is not None:
            order = np.argsort(-np.abs(self.coefficients))[:5]
            top = ", ".join(
                f"{self.basis.labels[i]}={self.coefficients[i] * 1e9:+.1f} nm" for i in order
            )
            lines.append(f"  largest modes: {top}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"Result(method={self.method!r}, device={self.device!r}, n_iter={self.n_iter}, "
            f"converged={self.converged}, loss={self.loss:.6g})"
        )

    # ------------------------------------------------------------------- I/O
    def save(self, path: str | Path) -> Path:
        """Save arrays and metadata to a compressed ``.npz`` file."""
        host = self.to_numpy()
        out = Path(path)
        if out.suffix != ".npz":
            out = out.with_suffix(".npz")
        payload: dict[str, Any] = {
            "method": host.method,
            "wavelength": host.wavelength,
            "loss": host.loss,
            "n_iter": host.n_iter,
            "converged": host.converged,
            "message": host.message,
            "elapsed": host.elapsed,
            "device": host.device,
            "history": np.asarray(host.history),
        }
        for name in (
            "opd",
            "phase",
            "amplitude",
            "coefficients",
            "model_images",
            "flux",
            "background",
            "tilts",
        ):
            value = getattr(host, name)
            if value is not None:
                payload[name] = np.asarray(value)
        if host.basis is not None:
            payload["basis_labels"] = np.asarray(host.basis.labels)
        if host.pupil is not None:
            payload["pupil_amplitude"] = host.pupil.amplitude
            payload["pupil_pitch"] = host.pupil.pitch
            payload["pupil_diameter"] = host.pupil.diameter
        np.savez_compressed(out, **payload)
        return out

    # ------------------------------------------------------------- plotting
    def plot(self, images: Any = None, *, unit: str = "nm", path: str | Path | None = None) -> Any:
        """Plot the OPD, and data/model/residual images if ``images`` is given.

        Needs matplotlib (``pip install 'solvephase[plot]'``). Returns the figure.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Result.plot needs matplotlib: pip install 'solvephase[plot]'"
            ) from exc
        scale = {"nm": 1e9, "um": 1e6, "m": 1.0}[unit]
        host = self.to_numpy()
        opd = np.asarray(host.opd) * scale
        if host.pupil is not None:
            opd = np.where(host.pupil.mask, opd, np.nan)
        ncols = 1 if images is None or host.model_images is None else 4
        fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 3.8), squeeze=False)
        ax = axes[0, 0]
        im = ax.imshow(opd, origin="lower", cmap="RdBu_r")
        ax.set_title(f"{host.method}: OPD [{unit}]")
        fig.colorbar(im, ax=ax, fraction=0.046)
        if ncols == 4:
            data = np.asarray(to_numpy(images))
            data = data[0] if data.ndim == 3 else data
            model = np.asarray(host.model_images)[0]
            for a, img, title in zip(
                axes[0, 1:],
                (data, model, data - model),
                ("data (ch 0)", "model (ch 0)", "residual"),
            ):
                im = a.imshow(
                    img, origin="lower", cmap="magma" if title != "residual" else "RdBu_r"
                )
                a.set_title(title)
                fig.colorbar(im, ax=a, fraction=0.046)
        for a in axes.ravel():
            a.set_xticks([])
            a.set_yticks([])
        fig.tight_layout()
        if path is not None:
            fig.savefig(path, dpi=120)
        return fig
