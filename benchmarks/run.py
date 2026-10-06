"""solvephase benchmark suite.

Measures warm, steady-state speed of every algorithm family on CPU and GPU,
separately from setup cost, and writes a JSON artifact with the hardware and
dependency context. Each case reports the median of several repeats.

Usage::

    python benchmarks/run.py                       # all cases, every available device
    python benchmarks/run.py --quick               # small sizes (CI smoke run)
    python benchmarks/run.py --device gpu --cases focal_lm cdi
    python benchmarks/run.py --output results.json

Render a Markdown table from an artifact with ``benchmarks/render_table.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import numpy as np

import solvephase as sp

CASES: dict[str, Callable[[Any, bool], list[dict[str, Any]]]] = {}


def case(
    name: str,
) -> Callable[[Callable[..., list[dict[str, Any]]]], Callable[..., list[dict[str, Any]]]]:
    def register(fn: Callable[..., list[dict[str, Any]]]) -> Callable[..., list[dict[str, Any]]]:
        CASES[name] = fn
        return fn

    return register


def timeit(fn: Callable[[], Any], backend: Any, *, repeats: int = 5, warmup: int = 1) -> float:
    """Median wall time of ``fn`` in seconds, synchronizing the device."""
    for _ in range(warmup):
        fn()
    backend.synchronize()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        backend.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


LAM = 1.0e-6


def _focal_setup(n: int, backend: Any, *, channels: int = 2, wavelengths: int = 1):
    pupil = sp.Pupil.circular(n, 1.0, obscuration=0.15)
    lams = np.linspace(0.9, 1.1, wavelengths) * LAM if wavelengths > 1 else LAM
    div = sp.zernike_diversity(pupil, 4, np.linspace(0.0, 0.3 * LAM, channels))
    model = sp.FocalPlaneModel(pupil, lams, n, sampling=2.0, diversity=div, device=backend)
    opd = sp.random_aberration(pupil, 0.08 * LAM, n_modes=30, start=4, seed=1)
    images = sp.simulate_images(model, opd, photons=1e7, seed=2)
    return pupil, model, opd, images


@case("focal_lm")
def bench_focal_lm(backend: Any, quick: bool) -> list[dict[str, Any]]:
    """Time-to-solution: 36-mode Levenberg-Marquardt, 2 diversity images, Poisson."""
    rows = []
    for n in [64, 128] if quick else [64, 128, 256]:
        pupil, model, opd, images = _focal_setup(n, backend)
        prob = sp.FocalPlaneProblem(
            model, images, basis=sp.Basis.zernike(pupil, 36), loss="poisson"
        )
        result: dict[str, Any] = {}

        def run() -> None:
            result["r"] = sp.solve(prob, method="lm")

        t = timeit(run, backend, repeats=3)
        r = result["r"]
        err = sp.wavefront_error(sp.to_numpy(r.opd), opd, pupil, remove="tiptilt")
        rows.append({"size": n, "seconds": t, "iterations": r.n_iter, "error_nm": err * 1e9})
    return rows


@case("focal_gradient")
def bench_focal_gradient(backend: Any, quick: bool) -> list[dict[str, Any]]:
    """One objective + analytic gradient evaluation (zonal), 2 images; and 5-wavelength broadband."""
    rows = []
    sizes = [128, 256] if quick else [128, 256, 512]
    for n in sizes:
        for nl in (1, 5):
            if nl == 5 and n > 256:
                continue
            _, model, _, images = _focal_setup(n, backend, wavelengths=nl)
            prob = sp.FocalPlaneProblem(model, images, basis=None, loss="poisson")
            x = prob.initial()
            t = timeit(lambda: prob.objective(x), backend, repeats=7)
            rows.append(
                {
                    "size": n,
                    "wavelengths": nl,
                    "engine": model.propagator.method,
                    "seconds": t,
                    "evals_per_s": 1 / t,
                }
            )
    return rows


@case("gerchberg_saxton")
def bench_gs(backend: Any, quick: bool) -> list[dict[str, Any]]:
    """Gerchberg-Saxton/Misell iterations per second, 3 diversity images."""
    rows = []
    for n in [128, 256] if quick else [128, 256, 512]:
        _, model, _, images = _focal_setup(n, backend, channels=3)
        iters = 20 if quick else 50

        def run() -> None:
            sp.gerchberg_saxton(
                model, images, iterations=iters, check_every=iters, tol=0.0, unwrap=False
            )

        t = timeit(run, backend, repeats=3)
        rows.append({"size": n, "iterations_per_s": iters / t})
    return rows


@case("retrieve_auto")
def bench_retrieve(backend: Any, quick: bool) -> list[dict[str, Any]]:
    """End-to-end :func:`solvephase.retrieve` (multi-start capture + ML polish)."""
    rows = []
    for n in [64] if quick else [64, 128]:
        pupil, model, opd, images = _focal_setup(n, backend)
        div = model.diversity_opd
        out: dict[str, Any] = {}

        def run() -> None:
            out["r"] = sp.retrieve(images, pupil, LAM, sampling=2.0, diversity=div, device=backend)

        t = timeit(run, backend, repeats=2)
        err = sp.wavefront_error(sp.to_numpy(out["r"].opd), opd, pupil, remove="tiptilt")
        rows.append({"size": n, "seconds": t, "error_nm": err * 1e9})
    return rows


@case("unwrap")
def bench_unwrap(backend: Any, quick: bool) -> list[dict[str, Any]]:
    """Weighted least-squares phase unwrapping on an obscured, spidered pupil."""
    rows = []
    for n in [128, 256] if quick else [128, 256, 512]:
        pupil = sp.Pupil.circular(n, 1.0, obscuration=0.3, spiders=4, spider_width=0.02)
        phase = sp.Basis.zernike(pupil, 10).synthesize(
            np.random.default_rng(0).standard_normal(10) * 8
        )
        wrapped = backend.asarray(sp.wrap(phase), dtype=np.float64)
        t = timeit(
            lambda: sp.unwrap_phase(wrapped, pupil.mask, weights=pupil.amplitude),
            backend,
            repeats=3,
        )
        rows.append({"size": n, "seconds": t})
    return rows


def _optional_cases() -> None:
    """Register benchmarks of optional algorithm modules when they are importable."""
    try:
        from bench_algorithms import register  # type: ignore[import-not-found]
    except ImportError:
        return
    register(case, timeit)


def environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "solvephase": sp.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        import scipy

        info["scipy"] = scipy.__version__
    except ImportError:  # pragma: no cover
        pass
    try:
        with open("/proc/cpuinfo") as handle:
            for line in handle:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    if sp.gpu_available():
        import cupy

        props = cupy.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        info["gpu"] = name.decode() if isinstance(name, bytes) else name
        info["cupy"] = cupy.__version__
        info["cuda_runtime"] = cupy.cuda.runtime.runtimeGetVersion()
    return info


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    _optional_cases()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--quick", action="store_true", help="small sizes for a smoke run")
    parser.add_argument("--device", choices=["cpu", "gpu", "all"], default="all")
    parser.add_argument("--precision", choices=["single", "double", "default"], default="default")
    parser.add_argument("--cases", nargs="*", default=None, help=f"subset of {sorted(CASES)}")
    parser.add_argument("--output", default=None, help="write the JSON artifact here")
    args = parser.parse_args(argv)

    devices = ["cpu", "gpu"] if args.device == "all" else [args.device]
    devices = [d for d in devices if d == "cpu" or sp.gpu_available()]
    names = args.cases or sorted(CASES)
    unknown = set(names) - set(CASES)
    if unknown:
        parser.error(f"unknown cases {sorted(unknown)}; available: {sorted(CASES)}")
    results: list[dict[str, Any]] = []
    for device in devices:
        precision = None if args.precision == "default" else args.precision
        backend = sp.get_backend(device, precision)
        for name in names:
            t0 = time.perf_counter()
            try:
                rows = CASES[name](backend, args.quick)
            except Exception as exc:  # report and keep going
                rows = [{"error": f"{type(exc).__name__}: {exc}"}]
            for row in rows:
                row.update({"case": name, "device": device, "precision": backend.precision})
                results.append(row)
                printable = {
                    k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                }
                print(json.dumps(printable))
            print(f"# {name} on {device}: {time.perf_counter() - t0:.1f} s", file=sys.stderr)
    artifact = {"environment": environment(), "quick": args.quick, "results": results}
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(artifact, handle, indent=2)
        print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
