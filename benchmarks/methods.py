"""Standardized comparison of every solvephase method.

Runs each algorithm on a reference problem and measures what a user needs to
choose between them:

* **data** - what has to be measured (number of images, diversity, a DM ...);
* **accuracy** - final wavefront (or object/phase) error at a standard SNR;
* **speed** - warm time to solution on CPU and GPU;
* **capture range** - success rate versus aberration size (focal-plane
  methods), success meaning a final error below 20 % of the aberration.

Focal-plane methods all see the *same* wavefront: a VLT-like pupil, H band,
0.1 waves RMS over Zernike modes 4-33, 10^6 photons per image, 2 e- read
noise. Each method gets the data it needs (two defocus-diverse images, one
astigmatic image, an extended scene, or a closed loop with a DM).

Every run is recorded twice: once with a callback that snapshots the solver
state (for the showcase animation), and once un-instrumented for the timing;
the recorded timestamps are rescaled to the un-instrumented duration.

Usage::

    python benchmarks/methods.py --output benchmarks/artifacts/methods.json
    python benchmarks/methods.py --quick            # smaller capture sweep
    python benchmarks/methods.py --markdown benchmarks/artifacts/methods.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import solvephase as sp
from solvephase.algorithms.phase_diversity import PhaseDiversityProblem
from solvephase.algorithms.wirtinger import relative_error, wirtinger
from solvephase.operators import CodedDiffractionOperator

LAM = 1.6e-6
N = 64
SAMPLING = 2.0
PHOTONS = 1e6
READ_NOISE = 2.0
BACKGROUND = 5.0
DEFOCUS = 0.3 * LAM  # RMS defocus diversity of the second image


@dataclass
class Trace:
    """One method's run on its reference problem."""

    key: str
    label: str
    family: str
    data: str
    requires: str
    unknowns: str
    times: list[float] = field(default_factory=list)  # seconds, rescaled
    errors: list[float] = field(default_factory=list)  # relative error per snapshot
    frames: list[np.ndarray] = field(default_factory=list)  # display maps
    seconds: float = math.nan  # warm, un-instrumented, CPU
    seconds_gpu: float = math.nan
    error: float = math.nan  # final relative error
    error_abs: float = math.nan  # final error in natural units (nm or relative)
    error_unit: str = "nm"
    iterations: int = 0
    capture: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            k: getattr(self, k)
            for k in (
                "key",
                "label",
                "family",
                "data",
                "requires",
                "unknowns",
                "seconds",
                "seconds_gpu",
                "error",
                "error_abs",
                "error_unit",
                "iterations",
                "capture",
            )
        }
        out["trace"] = {"times": self.times, "errors": self.errors}
        out["extra"] = {k: v for k, v in self.extra.items() if isinstance(v, (int, float, str))}
        return out


# ------------------------------------------------------------------ problems
class FocalSetup:
    """The shared focal-plane reference problem."""

    def __init__(self, rms_waves: float = 0.1, seed: int = 1, n_modes: int = 30, start: int = 4):
        self.pupil = sp.Pupil.vlt(N)
        self.truth = sp.random_aberration(
            self.pupil, rms_waves * LAM, n_modes=n_modes, start=start, power=1.0, seed=seed
        )
        self.seed = seed
        self.scale = sp.rms(self.truth, self.pupil, "tiptilt")
        self.diversity = sp.zernike_diversity(self.pupil, 4, [0.0, DEFOCUS])

    def model(self, device: str = "cpu", diversity: Any = "default") -> sp.FocalPlaneModel:
        div = self.diversity if isinstance(diversity, str) else diversity
        return sp.FocalPlaneModel(
            self.pupil, LAM, N, sampling=SAMPLING, diversity=div, device=device
        )

    def images(self) -> np.ndarray:
        return sp.simulate_images(
            self.model(),
            self.truth,
            photons=PHOTONS,
            background=BACKGROUND,
            read_noise=READ_NOISE,
            seed=self.seed + 100,
        )

    def error(self, opd: Any) -> float:
        return (
            sp.wavefront_error(sp.to_numpy(opd), self.truth, self.pupil, remove="tiptilt")
            / self.scale
        )


def _timed(fn: Callable[[], Any], device: str) -> tuple[Any, float]:
    be = sp.get_backend(device)
    be.synchronize()
    t0 = time.perf_counter()
    out = fn()
    be.synchronize()
    return out, time.perf_counter() - t0


def _state_opd(problem: Any, state: Any, device: str) -> np.ndarray:
    """OPD map of a recorded solver state (an internal parameter vector)."""
    x = sp.get_backend(device).asarray(state, dtype=state.dtype)
    if isinstance(problem, sp.FocalPlaneProblem):
        return np.asarray(sp.to_numpy(problem.result(x, method="snapshot").opd))
    return np.asarray(sp.to_numpy(problem.result(x).opd))


def _rescale(times: list[float], instrumented: float, plain: float) -> list[float]:
    factor = plain / instrumented if instrumented > 0 else 1.0
    return [t * factor for t in times]


# ---------------------------------------------------------- focal-plane runs
def run_misell(s: FocalSetup, device: str, record: bool) -> tuple[Any, list[tuple[float, Any]]]:
    model = s.model(device)
    images = s.images()
    snaps: list[tuple[float, Any]] = []
    t0 = time.perf_counter()

    def cb(it: int, theta: Any, err: float) -> None:
        snaps.append((time.perf_counter() - t0, sp.to_numpy(theta).copy()))

    # GS output is fitted to a modal basis, as in practice (e.g. JWST's hybrid
    # diversity algorithm): zonal GS also fits photon noise into weakly
    # constrained pupil frequencies.
    res = sp.gerchberg_saxton(
        model,
        images,
        iterations=300,
        background=BACKGROUND,
        check_every=5,
        basis=_misell_basis(s),
        callback=cb if record else None,
    )
    return res, snaps


def _misell_basis(s: FocalSetup) -> sp.Basis:
    return sp.Basis.zernike(s.pupil, 36)


def _focal_problem(s: FocalSetup, device: str, basis: Any) -> sp.FocalPlaneProblem:
    return sp.FocalPlaneProblem(
        s.model(device),
        s.images(),
        basis=basis,
        loss="poisson",
        read_noise=READ_NOISE,
        fit_background=True,
    )


def run_lm(s: FocalSetup, device: str, record: bool) -> tuple[Any, list[tuple[float, Any]]]:
    prob = _focal_problem(s, device, sp.Basis.zernike(s.pupil, 36))
    snaps: list[tuple[float, Any]] = [(0.0, prob.initial())]
    t0 = time.perf_counter()

    def cb(it: int, x: Any, f: float) -> None:
        snaps.append((time.perf_counter() - t0, x.copy()))

    res = sp.solve(prob, method="lm", callback=cb if record else None)
    return (res, prob), snaps


def run_lbfgs(s: FocalSetup, device: str, record: bool) -> tuple[Any, list[tuple[float, Any]]]:
    prob = _focal_problem(s, device, None)
    snaps: list[tuple[float, Any]] = [(0.0, prob.initial())]
    t0 = time.perf_counter()

    def cb(it: int, x: Any, f: float) -> None:
        if it % 3 == 0:
            snaps.append((time.perf_counter() - t0, x.copy()))

    res = sp.solve(prob, method="lbfgs", max_iter=300, callback=cb if record else None)
    return (res, prob), snaps


def run_retrieve(s: FocalSetup, device: str, record: bool) -> tuple[Any, list[tuple[float, Any]]]:
    res = sp.retrieve(
        s.images(),
        s.pupil,
        LAM,
        sampling=SAMPLING,
        diversity=[0.0, DEFOCUS],
        read_noise=READ_NOISE,
        device=device,
    )
    return res, []


def _extended_scene(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:N, :N] - (N - 1) / 2
    scene = np.full((N, N), 1e-3)
    for _ in range(10):
        cy, cx = rng.uniform(-N / 5, N / 5, 2)
        width = rng.uniform(1.0, 3.0)
        scene += rng.uniform(0.3, 1.0) * np.exp(-0.5 * ((yy - cy) ** 2 + (xx - cx) ** 2) / width**2)
    return scene


def _pd_data(s: FocalSetup) -> tuple[np.ndarray, np.ndarray]:
    div = sp.zernike_diversity(s.pupil, 4, [0.0, LAM / (2 * math.sqrt(3))])
    psf = sp.to_numpy(s.model("cpu", div).images(s.truth))
    scene = _extended_scene(s.seed)
    kernel = np.fft.rfft2(np.fft.ifftshift(psf, axes=(-2, -1)))
    clean = np.fft.irfft2(np.fft.rfft2(scene) * kernel, s=(N, N))
    rng = np.random.default_rng(s.seed + 7)
    images = rng.poisson(np.maximum(clean, 0) * 1000 / clean[0].mean()).astype(float)
    return images, div


def run_pd(s: FocalSetup, device: str, record: bool) -> tuple[Any, list[tuple[float, Any]]]:
    images, div = _pd_data(s)
    model = s.model(device, div)
    basis = sp.Basis.zernike(s.pupil, 33, start=4)
    prob = PhaseDiversityProblem(model, images, basis=basis)
    snaps: list[tuple[float, Any]] = [(0.0, prob.initial(None))]
    t0 = time.perf_counter()

    def cb(it: int, x: Any, f: float) -> None:
        snaps.append((time.perf_counter() - t0, x.copy()))

    res = sp.phase_diversity(model, images, basis=basis, callback=cb if record else None)
    return (res, prob), snaps


def run_lift(s: FocalSetup, device: str, record: bool) -> tuple[Any, list[tuple[float, Any]]]:
    sensor = sp.LIFT(
        s.pupil, LAM, N, sampling=SAMPLING, n_modes=10, read_noise=READ_NOISE, device=device
    )
    image = sp.simulate_images(
        sensor.model, s.truth, photons=PHOTONS, read_noise=READ_NOISE, seed=s.seed + 200
    )[0]
    prob = sensor.problem(image)
    snaps: list[tuple[float, Any]] = [(0.0, prob.initial())]
    t0 = time.perf_counter()

    def cb(it: int, x: Any, f: float) -> None:
        snaps.append((time.perf_counter() - t0, x.copy()))

    res = sensor.estimate(image, **({"callback": cb} if record else {}))
    return (res, prob), snaps


def run_ff(
    s: FocalSetup, device: str, record: bool, steps: int = 30
) -> tuple[Any, list[tuple[float, Any]]]:
    """Fast & Furious closed loop; only the sensor's own processing is timed."""
    ff = sp.FastAndFurious(s.pupil, LAM, N, sampling=SAMPLING, device=device)
    be = ff.backend
    model = sp.FocalPlaneModel(s.pupil, LAM, N, sampling=SAMPLING, device=device)
    rng = np.random.default_rng(s.seed + 300)
    truth = be.asarray(s.truth, dtype="real")
    mask = be.asarray(s.pupil.mask, dtype="real")
    dm = be.zeros(s.pupil.shape)
    change = None
    snaps: list[tuple[float, Any]] = [(0.0, sp.to_numpy(truth).copy())]
    clock = 0.0
    for _ in range(steps):
        expected = sp.to_numpy(model.images((truth + dm) * mask))[0] * PHOTONS
        image = rng.poisson(np.maximum(expected, 0)) + rng.normal(0, READ_NOISE, expected.shape)
        image = be.asarray(image, dtype="real")
        be.synchronize()
        t = time.perf_counter()
        estimate = ff.step(image, change)
        be.synchronize()
        clock += time.perf_counter() - t
        new_dm = dm - 0.5 * estimate * mask
        change = new_dm - dm
        dm = new_dm
        if record:
            snaps.append((clock, sp.to_numpy(truth + dm).copy()))
    return {"residual": sp.to_numpy(truth + dm), "seconds": clock, "steps": steps}, snaps


FOCAL: list[dict[str, Any]] = [
    {
        "key": "misell",
        "label": "Gerchberg-Saxton / Misell",
        "run": run_misell,
        "data": "2+ images with known diversity",
        "requires": "known pupil",
        "unknowns": "zonal phase, unwrapped and fitted to a basis",
    },
    {
        "key": "lm",
        "label": "Nonlinear ML, modal (LM)",
        "run": run_lm,
        "data": "1+ images; 2+ with diversity to fix the sign",
        "requires": "known pupil",
        "unknowns": "modal coefficients + flux, background, registration",
    },
    {
        "key": "lbfgs",
        "label": "Nonlinear ML, zonal (L-BFGS)",
        "run": run_lbfgs,
        "data": "2+ images with known diversity",
        "requires": "known pupil",
        "unknowns": "phase per pixel (+ amplitude)",
    },
    {
        "key": "retrieve",
        "label": "retrieve() (robust default)",
        "run": run_retrieve,
        "data": "2+ images with known diversity",
        "requires": "known pupil",
        "unknowns": "modal (+ optional zonal refinement)",
    },
    {
        "key": "pd",
        "label": "Phase diversity, extended object",
        "run": run_pd,
        "data": "2+ images of the same scene with known diversity",
        "requires": "known pupil; compact scene",
        "unknowns": "modal coefficients + the object",
    },
    {
        "key": "lift",
        "label": "LIFT",
        "run": run_lift,
        "data": "1 image with a known astigmatism bias",
        "requires": "known pupil",
        "unknowns": "~10 low-order modes",
    },
    {
        "key": "ff",
        "label": "Fast & Furious",
        "run": run_ff,
        "data": "1 image per step, in closed loop",
        "requires": "a DM; small residuals; symmetric pupil",
        "unknowns": "zonal phase (sequential)",
    },
]


def _focal_trace(spec: dict[str, Any], s: FocalSetup, device: str, record: bool) -> Trace:
    trace = Trace(
        spec["key"], spec["label"], "focal-plane", spec["data"], spec["requires"], spec["unknowns"]
    )
    run = spec["run"]
    run(s, device, False)  # warm-up (FFT plans, kernels)
    (out_rec, snaps), t_rec = _timed(lambda: run(s, device, record), device)
    (out, _), t_plain = _timed(lambda: run(s, device, False), device)
    key = spec["key"]
    if key == "ff":
        # The sensor's wavefront estimate is minus the DM shape: truth - residual.
        t_plain, t_rec = out["seconds"], out_rec["seconds"]
        final = s.truth - out["residual"]
        trace.iterations = out["steps"]
    elif isinstance(out, tuple):
        res, _prob = out
        final = res.opd
        trace.iterations = res.n_iter
    else:
        final = out.opd
        trace.iterations = out.n_iter
    trace.seconds = t_plain
    trace.error = s.error(final)
    trace.error_abs = trace.error * s.scale * 1e9
    if key == "lift":
        # LIFT measures only its low-order modes; also report the error on those.
        basis = out[0].basis
        target = basis.project(s.truth)
        in_model = sp.wavefront_error(sp.to_numpy(final), target, s.pupil, remove="tiptilt")
        trace.extra["in_model_error_nm"] = in_model * 1e9
        trace.extra["n_modes"] = basis.n_modes
    if record:
        times, frames = [], []
        for t, state in snaps:
            if key == "misell":
                phase = sp.unwrap_phase(state, s.pupil.mask, weights=s.pupil.amplitude)
                opd = _misell_basis(s).project(sp.to_numpy(phase) * LAM / (2 * math.pi))
            elif key == "ff":
                opd = s.truth - state
            else:
                opd = _state_opd(out_rec[1], state, device)
            times.append(t)
            frames.append(np.where(s.pupil.mask, opd, np.nan))
        trace.times = _rescale(times, t_rec, t_plain)
        trace.frames = frames
        trace.errors = [s.error(np.nan_to_num(f)) for f in frames]
        if key == "retrieve" or not trace.frames:
            trace.times, trace.frames = (
                [t_plain],
                [np.where(s.pupil.mask, sp.to_numpy(final), np.nan)],
            )
            trace.errors = [trace.error]
    return trace


# ------------------------------------------------------- other families
def run_cdi_case(device: str) -> tuple[Trace, dict[str, Any]]:
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:48, :48] - 23.5
    outline = np.hypot(yy, xx) < 18 * (1 + 0.25 * np.cos(3 * np.arctan2(yy, xx)))
    obj = np.where(outline, 0.3 + gaussian_filter(rng.uniform(size=(48, 48)), 1.5), 0.0)
    data = sp.simulate_cdi(obj, oversampling=2, photons=1e9, seed=1)
    support = sp.autocorrelation_support(data.intensity)
    mags = np.sqrt(sp.to_numpy(data.intensity))

    def run() -> Any:
        return sp.cdi(
            mags,
            support,
            schedule="hio:600,er:100",
            starts=8,
            constraint="positive",
            shrinkwrap={"every": 20, "sigma": 3.0, "sigma_min": 1.0, "threshold": 0.1},
            seed=2,
            device=device,
        )

    run()
    res, t = _timed(run, device)
    aligned, err = sp.align_object(res.object, data.object)
    trace = Trace(
        "cdi",
        "CDI: HIO→ER + shrinkwrap, 8 starts",
        "imaging",
        "1 oversampled diffraction pattern",
        "isolated object (oversampling ≥ 2)",
        "complex object",
        seconds=t,
        error=float(err),
        error_abs=float(err),
        error_unit="relative",
        iterations=res.n_iter,
    )
    return trace, {
        "intensity": sp.to_numpy(data.intensity),
        "object": np.abs(sp.to_numpy(aligned)),
        "truth": np.abs(sp.to_numpy(data.object)),
    }


def run_generic_case(device: str) -> tuple[Trace, dict[str, Any]]:
    n = 64
    be = sp.get_backend(device)
    op = CodedDiffractionOperator.random((n, n), 6, seed=1, backend=be)
    yy, xx = np.mgrid[:n, :n] - n / 2
    truth = np.exp(-(xx**2 + yy**2) / (2 * (n / 5) ** 2)) * np.exp(
        1j * 2 * np.pi * (xx + 0.5 * yy) / n
    )
    truth = truth * (1.0 + 0.5 * np.cos(2 * np.pi * yy / 16)) + 0.05
    y = np.abs(sp.to_numpy(op.forward(be.asarray(truth, dtype="complex")))) ** 2

    def run() -> Any:
        return wirtinger(op, y, method="lbfgs", tol=1e-8, seed=3)

    run()
    res, t = _timed(run, device)
    err = float(relative_error(res.x, truth))
    x = sp.to_numpy(res.x)
    x = x * np.exp(-1j * np.angle(np.vdot(x, truth)))
    trace = Trace(
        "generic",
        "Generic: L-BFGS (coded diffraction, 6 masks)",
        "generic",
        "6 coded diffraction patterns",
        "known random masks (m/n ≳ 4)",
        "complex vector",
        seconds=t,
        error=err,
        error_abs=err,
        error_unit="relative",
        iterations=res.n_iter,
    )
    return trace, {"estimate": np.abs(x), "phase": np.angle(x), "truth": np.abs(truth)}


def run_tie_case(device: str) -> tuple[Trace, dict[str, Any]]:
    n, pitch, wl = 256, 1e-6, 0.55e-6
    yy, xx = (np.mgrid[:n, :n] - n / 2) * pitch
    phase = sum(
        a * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * s**2))
        for a, x0, y0, s in [
            (2.0, 20e-6, 10e-6, 12e-6),
            (-1.5, -25e-6, -15e-6, 9e-6),
            (1.0, 5e-6, -30e-6, 15e-6),
        ]
    )
    amplitude = 1.0 - 0.3 * np.exp(-((xx + 10e-6) ** 2 + (yy - 20e-6) ** 2) / (2 * (20e-6) ** 2))
    z = [-2e-6, 0.0, 2e-6]
    stack = sp.simulate_defocus_stack(amplitude * np.exp(1j * phase), z, pitch, wl)

    def run() -> Any:
        return sp.tie(stack, z, pitch=pitch, wavelength=wl, method="dct", device=device)

    run()
    res, t = _timed(run, device)
    rec = sp.to_numpy(res.phase)
    err = float(np.std((rec - rec.mean()) - (phase - phase.mean())) / np.std(phase))
    trace = Trace(
        "tie",
        "TIE (non-uniform intensity, DCT)",
        "near-field",
        "3 intensities at ±dz",
        "weak defocus; sample inside the field",
        "phase map",
        seconds=t,
        error=err,
        error_abs=err,
        error_unit="relative",
        iterations=1,
    )
    return trace, {"intensity": sp.to_numpy(stack)[2], "phase": rec, "truth": phase}


OTHERS = [run_cdi_case, run_generic_case, run_tie_case]


# ------------------------------------------------------------ capture sweep
def _final_error(spec: dict[str, Any], s: FocalSetup) -> float:
    """One untimed run; relative wavefront error of the final estimate."""
    out, _ = spec["run"](s, "cpu", False)
    if spec["key"] == "ff":
        return s.error(s.truth - out["residual"])
    res = out[0] if isinstance(out, tuple) else out
    return s.error(res.opd)


def capture_sweep(levels: list[float], trials: int) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for spec in FOCAL:
        rates = {}
        for level in levels:
            ok = 0
            for t in range(trials):
                if spec["key"] == "lift":
                    s = FocalSetup(level, seed=500 + t, n_modes=9, start=2)
                else:
                    s = FocalSetup(level, seed=500 + t)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        ok += _final_error(spec, s) < 0.2
                except Exception as exc:  # a diverging method counts as a failure
                    print(f"  {spec['key']} {level}: {type(exc).__name__}", file=sys.stderr)
            rates[f"{level:g}"] = ok / trials
            print(f"capture {spec['key']:8s} {level:.2f} waves: {ok}/{trials}", file=sys.stderr)
        out[spec["key"]] = rates
    return out


def run_all(
    *, quick: bool, record: bool = True, gpu: bool = True
) -> tuple[list[Trace], dict[str, Any]]:
    warnings.filterwarnings("ignore")
    setup = FocalSetup()
    traces: list[Trace] = []
    for spec in FOCAL:
        trace = _focal_trace(spec, setup, "cpu", record)
        if gpu and sp.gpu_available():
            gpu_trace = _focal_trace(spec, setup, "gpu", record=False)
            trace.seconds_gpu = gpu_trace.seconds
        print(
            f"{trace.key:8s} error {trace.error:.3%} ({trace.error_abs:.2f} nm) "
            f"cpu {trace.seconds:.3f}s gpu {trace.seconds_gpu:.3f}s",
            file=sys.stderr,
        )
        traces.append(trace)
    gallery: dict[str, Any] = {
        "truth": np.where(setup.pupil.mask, setup.truth, np.nan),
        "images": setup.images(),
        "pd_images": _pd_data(setup)[0],
    }
    for fn in OTHERS:
        trace, panels = fn("cpu")
        if gpu and sp.gpu_available():
            trace.seconds_gpu = fn("gpu")[0].seconds
        gallery[trace.key] = panels
        print(
            f"{trace.key:8s} error {trace.error:.2e} cpu {trace.seconds:.3f}s gpu {trace.seconds_gpu:.3f}s",
            file=sys.stderr,
        )
        traces.append(trace)
    return traces, gallery


# ---------------------------------------------------------------- markdown
def _fmt_time(t: float) -> str:
    if not math.isfinite(t):
        return "-"
    return f"{t * 1e3:.0f} ms" if t < 1 else f"{t:.2f} s"


def markdown(artifact: dict[str, Any]) -> str:
    rows = artifact["methods"]
    focal = [r for r in rows if r["family"] == "focal-plane"]
    other = [r for r in rows if r["family"] != "focal-plane"]
    levels = artifact.get("capture_levels", [])
    lines = [
        "| Method | Data needed | Extra requirements | Recovers | CPU | GPU | Error at 0.1 λ RMS |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in focal:
        lines.append(
            f"| {r['label']} | {r['data']} | {r['requires']} | {r['unknowns']} | "
            f"{_fmt_time(r['seconds'])} | {_fmt_time(r['seconds_gpu'])} | "
            f"{r['error_abs']:.2f} nm ({r['error']:.1%})"
            + (
                f"; {r['extra']['in_model_error_nm']:.2f} nm on its {r['extra']['n_modes']} modes"
                if "in_model_error_nm" in r.get("extra", {})
                else ""
            )
            + " |"
        )
    out = ["### Focal-plane methods", "", "\n".join(lines), ""]
    if levels:
        head = "| Method | " + " | ".join(f"{lv:g} λ" for lv in levels) + " |"
        sep = "|---|" + "---|" * len(levels)
        body = []
        for r in focal:
            cap = r.get("capture", {})
            cells = [f"{100 * cap.get(f'{lv:g}', math.nan):.0f} %" if cap else "-" for lv in levels]
            body.append(f"| {r['label']} | " + " | ".join(cells) + " |")
        out += [
            "### Capture range",
            "",
            "Success rate (final error < 20 % of the aberration) versus aberration RMS:",
            "",
            "\n".join([head, sep, *body]),
            "",
        ]
    lines = [
        "| Method | Data needed | Requirements | Recovers | CPU | GPU | Error |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in other:
        lines.append(
            f"| {r['label']} | {r['data']} | {r['requires']} | {r['unknowns']} | "
            f"{_fmt_time(r['seconds'])} | {_fmt_time(r['seconds_gpu'])} | {r['error']:.1e} (relative) |"
        )
    out += ["### Other measurement models", "", "\n".join(lines), ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", default=None, help="write the JSON artifact here")
    parser.add_argument("--markdown", default=None, help="print tables from an existing artifact")
    parser.add_argument("--no-capture", action="store_true")
    args = parser.parse_args(argv)
    if args.markdown:
        with open(args.markdown, encoding="utf-8") as handle:
            print(markdown(json.load(handle)))
        return 0
    traces, _ = run_all(quick=args.quick, record=False)
    levels = [0.05, 0.1, 0.2, 0.3, 0.5] if args.quick else [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
    if not args.no_capture:
        capture = capture_sweep(levels, 2 if args.quick else 6)
        for t in traces:
            t.capture = capture.get(t.key, {})
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from run import environment

    artifact = {
        "environment": environment(),
        "capture_levels": [] if args.no_capture else levels,
        "methods": [t.summary() for t in traces],
    }
    print(markdown(artifact))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(artifact, handle, indent=2, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
