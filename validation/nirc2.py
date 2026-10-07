"""Validation of solvephase on real Keck/NIRC2 focus-diversity data.

The data are daytime calibrations of the Keck adaptive-optics bench, taken
with NIRC2's narrow camera through the full circular bench pupil (internal
source, no telescope). Each run is a stack of nine images at focus-stage
offsets from -7 to +2.5, taken while the 349-actuator Xinetics deformable
mirror held a known command. In six runs a known pattern (coma, or coma plus
trefoil at three strengths) was added to that command. The data belong to the
observatory and are not distributed with solvephase; this script needs a copy.

The checks use only quantities that need no external calibration:

1. ``fit``: one wavefront per run (Zernike modes 2-37) reproduces all nine
   images with a common defocus scale and sampling, calibrated once on the
   reference run.
2. ``injection``: subtracting the reference run's wavefront from each
   injection run's wavefront recovers the commanded DM pattern. One DM
   orientation (fitted once over all runs) and one gain in nm per volt
   explain every run. Paired runs that differ only in injection strength
   recover the commanded ratio, which needs no gain at all.
3. ``prediction``: with gain and orientation fixed, the DM command difference
   between the unsharpened and sharpened runs (the IDL image-sharpening
   correction) predicts their measured wavefront difference, with no free
   parameter.
4. ``basis``: the recovered DM gain does not depend on the number of modes
   fitted.
5. ``keck``: the fitted DM gain and pupil size agree with the Keck AO
   software's Xinetics parameters (600 nm of wavefront per volt; a 5.52 in
   control aperture over a 7 mm actuator pitch), which the fit never sees.

Tip, tilt and focus are excluded from run-to-run comparisons: each frame is
cropped around its own centroid, and the sharpening loop moves the focus stage
between runs. Fits with more than ~36 modes are not used: on these data the
likelihood has flat directions there (solutions ~300 nm RMS apart differ by a
few per cent in chi-squared), consistent with unmodelled pupil illumination
and detector structure.

Usage::

    python validation/nirc2.py --data /path/to/image_sharpening_datasets --output DIR
    SOLVEPHASE_NIRC2_DATA=/path/to/... python validation/nirc2.py --quick

The data directory holds one sub-directory per run, named
``[idl_image_sharpening_]<label>_<YYYYMMDD-HHMMSS>``, each with
``raw_images.npy`` (9, 512, 512), ``distances.npy`` (9,),
``initial_dm_command.npy`` (21, 21), ``script_params.json`` and, for
injection runs, ``injected.npy`` (21, 21).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates

import solvephase as sp

#: Reference run (IDL sharpening applied, nothing injected).
REFERENCE = "no_injection"
#: Run without the IDL sharpening correction.
UNSHARPENED = "no_image_sharpening"
#: Runs that differ only in the strength of the injected pattern.
PAIRS = (
    ("coma_injection", "coma_injection_half_strength"),
    ("injected_combined_b", "injected_combined_b_stronger"),
)
CROP = 128
PUPIL_PIXELS = 64
N_MODES = 36  # Noll 2..37
COMPARED = slice(3, N_MODES)  # Noll 5..37
DETECTOR_GAIN = 4.0  # e-/DN, for photon-noise weights
ACTUATORS = 21
#: Keck AO software: Xinetics DM conversion, 600 nm per volt.
KECK_NM_PER_VOLT = 600.0
#: Xinetics control aperture (5.52 in) over the 7 mm actuator pitch, as a radius.
KECK_PUPIL_RADIUS_PITCHES = 5.52 * 25.4 / 7.0 / 2.0


def check(
    name: str, value: float, threshold: float, *, below: bool = True, **extra: Any
) -> dict[str, Any]:
    passed = bool(value <= threshold) if below else bool(value >= threshold)
    rel = "<=" if below else ">="
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {value:.4g} ({rel} {threshold:g})")
    return {
        "name": name,
        "value": float(value),
        "threshold": threshold,
        "below": below,
        "passed": passed,
        **extra,
    }


# ------------------------------------------------------------------- data
def load_runs(root: Path) -> dict[str, Path]:
    runs = {}
    for path in sorted(root.iterdir()):
        if not (path / "raw_images.npy").is_file():
            continue
        label = re.sub(r"_\d{8}-\d{6}$", "", path.name).removeprefix("idl_image_sharpening_")
        runs[label] = path
    missing = {REFERENCE, UNSHARPENED, *sum(PAIRS, ())} - set(runs)
    if missing:
        raise SystemExit(f"data directory {root} lacks runs: {sorted(missing)}")
    return runs


def crop_stack(run: Path) -> tuple[np.ndarray, np.ndarray]:
    """Frames cropped around their own centroids, and the focus-stage offsets."""
    raw = np.load(run / "raw_images.npy", allow_pickle=False).astype(np.float64)
    distances = np.load(run / "distances.npy", allow_pickle=False).astype(np.float64)
    # Coarse centre: the brightest smoothed pixel of the frame nearest focus.
    near = gaussian_filter(np.clip(raw[np.argmin(np.abs(distances))], 0, None), 4.0)
    cy, cx = np.unravel_index(np.argmax(near), near.shape)
    half, search = CROP // 2, 96
    frames = []
    for frame in raw:
        y0, x0 = max(cy - search, 0), max(cx - search, 0)
        win = gaussian_filter(np.clip(frame[y0 : cy + search, x0 : cx + search], 0, None), 3.0)
        win = np.where(win > 0.2 * win.max(), win, 0.0)
        yy, xx = np.indices(win.shape)
        y = int(np.rint((win * yy).sum() / win.sum())) + y0
        x = int(np.rint((win * xx).sum() / win.sum())) + x0
        frames.append(frame[y - half : y + half, x - half : x + half])
    return np.asarray(frames), distances


def border_sigma(frames: np.ndarray, width: int = 12) -> np.ndarray:
    """Robust per-frame noise from the crop border (median absolute deviation)."""
    out = []
    for f in frames:
        b = np.concatenate(
            [f[:width].ravel(), f[-width:].ravel(), f[:, :width].ravel(), f[:, -width:].ravel()]
        )
        out.append(1.4826 * np.median(np.abs(b - np.median(b))))
    return np.asarray(out)


# ------------------------------------------------------------- retrieval
def make_problem(
    frames: np.ndarray,
    distances: np.ndarray,
    wavelength: float,
    *,
    scale: float,
    sampling: float,
    n_modes: int,
) -> sp.FocalPlaneProblem:
    pupil = sp.Pupil.circular(PUPIL_PIXELS)
    diversity = sp.zernike_diversity(pupil, 4, list(scale * distances))
    model = sp.FocalPlaneModel(
        pupil, wavelength, frames.shape[1:], sampling=sampling, diversity=diversity
    )
    sigma = border_sigma(frames)
    weights = 1.0 / (sigma[:, None, None] ** 2 + np.maximum(frames, 0.0) / DETECTOR_GAIN)
    return sp.FocalPlaneProblem(
        model,
        frames,
        basis=sp.Basis.zernike(pupil, n_modes),
        loss="gaussian",
        weights=weights,
        fit_flux=True,
        fit_background=True,
        fit_tilt=True,
    )


def retrieve(
    frames: np.ndarray,
    distances: np.ndarray,
    wavelength: float,
    *,
    scale: float,
    sampling: float,
    n_modes: int = N_MODES,
) -> dict[str, Any]:
    problem = make_problem(
        frames, distances, wavelength, scale=scale, sampling=sampling, n_modes=n_modes
    )
    result = sp.solve(problem, method="lm", max_iter=100)
    model = np.asarray(sp.to_numpy(result.model_images))
    sigma = border_sigma(frames)
    w = 1.0 / (sigma[:, None, None] ** 2 + np.maximum(frames, 0.0) / DETECTOR_GAIN)
    chi2 = float(((frames - model) ** 2 * w).mean())
    return {
        "coefficients": np.asarray(sp.to_numpy(result.coefficients)),
        "model": model,
        "chi2": chi2,
        "basis": problem.basis,
    }


def calibrate(
    frames: np.ndarray, distances: np.ndarray, wavelength: float, quick: bool
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Defocus per stage unit (RMS metres) and sampling (pixels per lambda/D) by profile likelihood."""
    samplings = np.array([2.95, 3.05, 3.15, 3.25])
    scales = np.array([0.18, 0.19, 0.20, 0.21]) * 1e-6
    if quick:
        samplings, scales = samplings[1:3], scales[1:3]
    grid = np.array(
        [
            [
                retrieve(frames, distances, wavelength, scale=sc, sampling=s, n_modes=21)["chi2"]
                for sc in scales
            ]
            for s in samplings
        ]
    )
    i, j = np.unravel_index(np.argmin(grid), grid.shape)
    return float(scales[j]), float(samplings[i]), grid, samplings, scales


# ------------------------------------------------------------------- DM
def fill_inactive(command: np.ndarray) -> np.ndarray:
    """Extend the active actuators' values outward (nearest) so interpolation has no rim."""
    active = command != 0
    idx = distance_transform_edt(~active, return_distances=False, return_indices=True)
    return command[tuple(idx)]


def dihedral(a: np.ndarray, k: int) -> np.ndarray:
    r = np.rot90(a, k % 4)
    return r.T if k >= 4 else r


def command_map(command: np.ndarray, *, k: int, radius: float, angle: float) -> np.ndarray:
    """Interpolate a 21x21 DM command onto the pupil grid.

    ``k`` picks one of the eight flips/rotations of the actuator grid, ``radius``
    is the pupil radius in actuator pitches and ``angle`` a residual rotation in
    degrees.
    """
    c = dihedral(fill_inactive(np.asarray(command, dtype=np.float64)), k)
    n = PUPIL_PIXELS
    x = (np.arange(n) - (n - 1) / 2) / (n / 2)
    yy, xx = np.meshgrid(x, x, indexing="ij")
    th = np.deg2rad(angle)
    xr, yr = np.cos(th) * xx - np.sin(th) * yy, np.sin(th) * xx + np.cos(th) * yy
    mid = (ACTUATORS - 1) / 2
    return map_coordinates(c, [mid + yr * radius, mid + xr * radius], order=3, mode="nearest")


def command_modes(basis: sp.Basis, command: np.ndarray, geometry: dict[str, float]) -> np.ndarray:
    return basis.fit(command_map(command, **geometry))[COMPARED]  # type: ignore[arg-type]


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def fit_geometry(
    basis: sp.Basis, deltas: dict[str, np.ndarray], commands: dict[str, np.ndarray]
) -> dict[str, float]:
    """DM orientation: the flip, radius and rotation that best correlate commands with deltas over all runs."""

    def score(k: int, radius: float, angle: float) -> float:
        g = {"k": k, "radius": radius, "angle": angle}
        return float(
            np.mean([correlation(deltas[n], command_modes(basis, commands[n], g)) for n in deltas])
        )

    k = max(range(8), key=lambda k: score(k, 10.0, 0.0))
    angles = np.arange(-10.0, 10.01, 1.0)
    angle = float(angles[np.argmax([score(k, 10.0, a) for a in angles])])
    radii = np.arange(9.0, 11.01, 0.25)
    radius = float(radii[np.argmax([score(k, r, angle) for r in radii])])
    return {"k": k, "radius": radius, "angle": angle}


# ------------------------------------------------------------- figures
def save_figures(
    out: Path,
    ref: dict[str, Any],
    frames: np.ndarray,
    distances: np.ndarray,
    maps: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(distances)
    fig, ax = plt.subplots(3, len(order), figsize=(2.0 * len(order), 6.4))
    for j, k in enumerate(order):
        vmax = np.percentile(frames[k], 99.9)
        stretch = np.arcsinh(vmax / 30)
        for i, img in enumerate([frames[k], ref["model"][k], frames[k] - ref["model"][k]]):
            a = ax[i, j]
            if i < 2:
                a.imshow(
                    np.arcsinh(np.clip(img, 0, None) / 30),
                    origin="lower",
                    cmap="magma",
                    vmin=0,
                    vmax=stretch,
                )
            else:
                a.imshow(img, origin="lower", cmap="RdBu_r", vmin=-0.15 * vmax, vmax=0.15 * vmax)
            a.set_xticks([])
            a.set_yticks([])
        ax[0, j].set_title(f"stage {distances[k]:+.1f}", fontsize=9)
    for i, label in enumerate(["NIRC2", "solvephase", "residual"]):
        ax[i, 0].set_ylabel(label)
    fig.suptitle(
        f"Reference run: one wavefront (Noll 2-37) fits nine defocused images (reduced chi2 {ref['chi2']:.1f})"
    )
    plt.tight_layout()
    plt.savefig(out / "nirc2_fit.png", dpi=70)
    plt.close(fig)

    runs = list(maps)
    fig = plt.figure(figsize=(2.4 * len(runs), 9.0))
    grid = fig.add_gridspec(3, len(runs), height_ratios=[1, 1, 1.5])
    mask = sp.Pupil.circular(PUPIL_PIXELS).mask
    for j, n in enumerate(runs):
        got, want, corr = maps[n]
        v = np.abs(want[mask]).max()
        for i, m in enumerate([got, want]):
            a = fig.add_subplot(grid[i, j])
            im = a.imshow(np.where(mask, m, np.nan), origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
            a.set_xticks([])
            a.set_yticks([])
            if j == 0:
                a.set_ylabel(["retrieved change", "command x gain"][i])
            if i == 0:
                a.set_title(f"{n}\ncorr {corr:.3f}", fontsize=8)
            else:
                fig.colorbar(im, ax=a, fraction=0.046, pad=0.02, label="nm")
    half = len(runs) // 2
    a1 = fig.add_subplot(grid[2, :half])
    a2 = fig.add_subplot(grid[2, half:])
    names = list(summary["gains"])
    a1.barh(names, [summary["gains"][n] * 1e9 for n in names])
    a1.axvline(summary["gain"] * 1e9, color="k", ls="--")
    a1.set_xlabel("DM gain (nm OPD per volt)")
    a1.tick_params(labelsize=8)
    for n, (cmd, got) in summary["scatter"].items():
        a2.plot(cmd * 1e9, got * 1e9, ".", alpha=0.7, label=n)
    lim = max(np.abs(a2.get_xlim()).max(), np.abs(a2.get_ylim()).max())
    a2.plot([-lim, lim], [-lim, lim], "k--", lw=1)
    a2.set_xlabel("command x gain, per Zernike mode (nm)")
    a2.set_ylabel("retrieved change, per mode (nm)")
    a2.legend(fontsize=7)
    fig.suptitle(
        "Injected DM patterns recovered from NIRC2 images (Noll 5-37, one gain, one DM orientation)"
    )
    fig.tight_layout()
    plt.savefig(out / "nirc2_injections.png", dpi=70)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default=os.environ.get("SOLVEPHASE_NIRC2_DATA"))
    parser.add_argument("--output", default="validation-artifacts")
    parser.add_argument(
        "--quick", action="store_true", help="coarser calibration, no basis-size check"
    )
    args = parser.parse_args(argv)
    if not args.data:
        parser.error("give --data or set SOLVEPHASE_NIRC2_DATA")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    runs = load_runs(Path(args.data))
    params = json.loads((runs[REFERENCE] / "script_params.json").read_text(encoding="utf-8"))
    wavelength = float(params["wavelength"]) * 1e-6

    stacks = {n: crop_stack(p) for n, p in runs.items()}
    frames, distances = stacks[REFERENCE]
    scale, sampling, grid, samplings, scales = calibrate(frames, distances, wavelength, args.quick)
    print(
        f"calibration: {scale * 1e9:.0f} nm RMS defocus per stage unit, {sampling:.2f} px per lambda/D"
    )

    fits = {n: retrieve(*stacks[n], wavelength, scale=scale, sampling=sampling) for n in runs}
    basis = fits[REFERENCE]["basis"]
    checks = []
    for n, f in fits.items():
        checks.append(check(f"fit reduced chi2, {n}", f["chi2"], 10.0))

    injected = {
        n: np.load(p / "injected.npy", allow_pickle=False)
        for n, p in runs.items()
        if (p / "injected.npy").is_file()
    }
    base = fits[REFERENCE]["coefficients"][COMPARED]
    deltas = {n: fits[n]["coefficients"][COMPARED] - base for n in injected}
    geometry = fit_geometry(basis, deltas, injected)
    print(
        f"DM orientation: flip/rotation {geometry['k']}, pupil radius {geometry['radius']} pitches, rotation {geometry['angle']} deg"
    )

    modes = {n: command_modes(basis, c, geometry) for n, c in injected.items()}
    gains = {n: float(deltas[n] @ modes[n] / (modes[n] @ modes[n])) for n in injected}
    gain = float(np.mean(list(gains.values())))
    for n in injected:
        checks.append(
            check(
                f"injection recovery correlation, {n}",
                correlation(deltas[n], modes[n]),
                0.90,
                below=False,
            )
        )
    scatter = float(np.std(list(gains.values())) / gain)
    checks.append(
        check("DM gain scatter between runs (relative)", scatter, 0.10, gain_nm_per_volt=gain * 1e9)
    )
    checks.append(
        check(
            f"DM gain vs Keck software {KECK_NM_PER_VOLT:.0f} nm/V (relative)",
            abs(gain * 1e9 / KECK_NM_PER_VOLT - 1),
            0.15,
            gain_nm_per_volt=gain * 1e9,
        )
    )
    checks.append(
        check(
            f"pupil radius vs Xinetics control aperture {KECK_PUPIL_RADIUS_PITCHES:.2f} pitches (relative)",
            abs(geometry["radius"] / KECK_PUPIL_RADIUS_PITCHES - 1),
            0.05,
            radius_pitches=geometry["radius"],
        )
    )
    for weak, strong in PAIRS:
        a, b = (
            (weak, strong)
            if np.linalg.norm(modes[weak]) < np.linalg.norm(modes[strong])
            else (strong, weak)
        )
        want = np.linalg.norm(modes[b]) / np.linalg.norm(modes[a])
        got = np.linalg.norm(deltas[b]) / np.linalg.norm(deltas[a])
        checks.append(
            check(
                f"strength ratio {b} / {a} (relative error vs commanded {want:.3f})",
                abs(got / want - 1),
                0.10,
                measured=got,
                commanded=want,
            )
        )

    command = np.load(runs[UNSHARPENED] / "initial_dm_command.npy", allow_pickle=False).astype(
        np.float64
    )
    command -= np.load(runs[REFERENCE] / "initial_dm_command.npy", allow_pickle=False)
    predicted = gain * command_modes(basis, command, geometry)
    measured = fits[UNSHARPENED]["coefficients"][COMPARED] - base
    checks.append(
        check(
            "sharpening correction predicted with no free parameter: correlation",
            correlation(measured, predicted),
            0.85,
            below=False,
        )
    )
    amp = np.linalg.norm(measured) / np.linalg.norm(predicted)
    checks.append(
        check(
            "sharpening correction amplitude, measured / predicted - 1",
            abs(amp - 1),
            0.20,
            ratio=amp,
        )
    )

    basis_gains = {N_MODES: gain}
    if not args.quick:
        for n_modes in (21, 28):
            sel = slice(3, n_modes)
            small = {
                n: retrieve(
                    *stacks[n], wavelength, scale=scale, sampling=sampling, n_modes=n_modes
                )["coefficients"][sel]
                for n in [REFERENCE, *injected]
            }
            b = sp.Basis.zernike(sp.Pupil.circular(PUPIL_PIXELS), n_modes)
            g = []
            for n in injected:
                z = b.fit(command_map(injected[n], **geometry))[sel]  # type: ignore[arg-type]
                d = small[n] - small[REFERENCE]
                g.append(d @ z / (z @ z))
            basis_gains[n_modes] = float(np.mean(g))
        spread = max(basis_gains.values()) / min(basis_gains.values()) - 1
        checks.append(check("DM gain change across 21/28/36 fitted modes (relative)", spread, 0.05))

    maps = {}
    for n in injected:
        full_d, full_m = np.zeros(N_MODES), np.zeros(N_MODES)
        full_d[COMPARED], full_m[COMPARED] = deltas[n], gain * modes[n]
        maps[n] = (
            basis.synthesize(full_d) * 1e9,
            basis.synthesize(full_m) * 1e9,
            correlation(deltas[n], modes[n]),
        )
    summary = {
        "gains": gains,
        "gain": gain,
        "scatter": {n: (gain * modes[n], deltas[n]) for n in injected},
    }
    save_figures(out, fits[REFERENCE], frames, distances, maps, summary)

    report = {
        "solvephase": sp.__version__,
        "quick": args.quick,
        "seconds": time.perf_counter() - t0,
        "calibration": {
            "defocus_rms_m_per_stage_unit": scale,
            "sampling_pixels_per_lambda_over_d": sampling,
            "chi2_grid": grid.tolist(),
            "grid_samplings": samplings.tolist(),
            "grid_scales_m": scales.tolist(),
        },
        "dm": {
            "gain_nm_opd_per_volt": gain * 1e9,
            "per_run": {n: g * 1e9 for n, g in gains.items()},
            "by_n_modes": {str(k): v * 1e9 for k, v in basis_gains.items()},
            "geometry": geometry,
        },
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
    }
    (out / "nirc2.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    n_fail = sum(not c["passed"] for c in checks)
    print(
        f"{len(checks) - n_fail}/{len(checks)} checks passed in {report['seconds']:.0f} s -> {out}"
    )
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
