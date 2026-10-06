"""Head-to-head comparison with the usual alternatives.

No maintained Python package ships a focal-plane phase retrieval solver, so the
baselines are what a user would otherwise write:

* **HCIPy + SciPy L-BFGS-B**: an HCIPy Fraunhofer forward model, a Poisson
  objective and ``scipy.optimize.minimize`` with finite-difference gradients;
* **HCIPy + SciPy least_squares**: the same model with a trust-region
  Gauss-Newton solver and a finite-difference Jacobian (the LIFT-style
  approach);
* **NumPy Misell (textbook)**: a plain ``numpy.fft`` multi-plane
  Gerchberg-Saxton loop with ``fftshift`` in every iteration;
* **NumPy HIO (textbook)** for coherent diffraction imaging.

Every method sees the same data; time includes the solve but not data
generation. Usage::

    python benchmarks/compare.py --output compare.json
    python benchmarks/compare.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from typing import Any

import numpy as np

import solvephase as sp

LAM = 1.0e-6


def focal_problem(n: int, n_modes: int):
    pupil = sp.Pupil.circular(n, 1.0, obscuration=0.15, supersample=1)
    basis = sp.Basis.zernike(pupil, n_modes)
    truth = sp.random_aberration(pupil, 0.08 * LAM, n_modes=n_modes, start=2, seed=1)
    div = sp.zernike_diversity(pupil, 4, [0.0, 0.25 * LAM])
    model = sp.FocalPlaneModel(pupil, LAM, n, sampling=2.0, diversity=div, offset=-0.5)
    images = sp.simulate_images(model, truth, photons=1e7, seed=2)
    return pupil, basis, truth, div, images


def run_solvephase(pupil, basis, truth, div, images, device: str) -> dict[str, Any]:
    n = images.shape[-1]
    model = sp.FocalPlaneModel(
        pupil, LAM, n, sampling=2.0, diversity=div, offset=-0.5, device=device
    )
    prob = sp.FocalPlaneProblem(model, images, basis=basis, loss="poisson")
    sp.solve(prob, method="lm", max_iter=2)  # warm-up (plans, kernels)
    model.backend.synchronize()
    t0 = time.perf_counter()
    res = sp.solve(prob, method="lm")
    model.backend.synchronize()
    dt = time.perf_counter() - t0
    err = sp.wavefront_error(sp.to_numpy(res.opd), truth, pupil, remove="tiptilt")
    return {"seconds": dt, "error_nm": err * 1e9, "iterations": res.n_iter}


def _hcipy_model(pupil, basis, div, n):
    import hcipy as hc

    grid = hc.make_pupil_grid(pupil.shape[0], pupil.diameter)
    focal = hc.make_focal_grid(q=2, num_airy=n / 4, spatial_resolution=LAM / pupil.diameter)
    prop = hc.FraunhoferPropagator(grid, focal)
    modes = basis.mode_maps()
    amp = pupil.amplitude.ravel()

    def images_for(coeffs: np.ndarray) -> np.ndarray:
        opd = np.tensordot(coeffs, modes, axes=1)
        out = []
        for d in div:
            wf = hc.Wavefront(
                hc.Field(amp * np.exp(2j * np.pi * (opd + d).ravel() / LAM), grid), LAM
            )
            out.append(np.asarray(prop(wf).power.shaped))
        return np.stack(out)

    return images_for


def run_hcipy_lbfgs(pupil, basis, truth, div, images) -> dict[str, Any]:
    from scipy.optimize import minimize

    n = images.shape[-1]
    images_for = _hcipy_model(pupil, basis, div, n)
    norm = images_for(np.zeros(basis.n_modes)).sum(axis=(1, 2))
    flux = images.sum(axis=(1, 2)) / norm
    d = images

    def objective(c_nm: np.ndarray) -> float:
        m = images_for(c_nm * 1e-9) * flux[:, None, None] + 1e-6
        return float(np.sum(m - d * np.log(m)))

    t0 = time.perf_counter()
    res = minimize(objective, np.zeros(basis.n_modes), method="L-BFGS-B", options={"maxiter": 300})
    dt = time.perf_counter() - t0
    err = sp.wavefront_error(basis.synthesize(res.x * 1e-9), truth, pupil, remove="tiptilt")
    return {
        "seconds": dt,
        "error_nm": err * 1e9,
        "iterations": int(res.nit),
        "evaluations": int(res.nfev),
    }


def run_hcipy_least_squares(pupil, basis, truth, div, images) -> dict[str, Any]:
    from scipy.optimize import least_squares

    n = images.shape[-1]
    images_for = _hcipy_model(pupil, basis, div, n)
    norm = images_for(np.zeros(basis.n_modes)).sum(axis=(1, 2))
    flux = images.sum(axis=(1, 2)) / norm
    sd = np.sqrt(np.maximum(images, 0))

    def residual(c_nm: np.ndarray) -> np.ndarray:
        m = images_for(c_nm * 1e-9) * flux[:, None, None]
        return (np.sqrt(np.maximum(m, 0)) - sd).ravel()

    t0 = time.perf_counter()
    res = least_squares(residual, np.zeros(basis.n_modes), method="trf", x_scale=10.0)
    dt = time.perf_counter() - t0
    err = sp.wavefront_error(basis.synthesize(res.x * 1e-9), truth, pupil, remove="tiptilt")
    return {"seconds": dt, "error_nm": err * 1e9, "evaluations": int(res.nfev)}


def run_numpy_misell(pupil, truth, div, images, iterations: int) -> dict[str, Any]:
    """A textbook multi-plane GS: pad, fftshift, FFT, replace modulus, inverse."""
    n = pupil.shape[0]
    big = 2 * n
    amp = pupil.amplitude
    target = np.sqrt(np.maximum(images, 0))
    phase = np.zeros((n, n))
    t0 = time.perf_counter()
    for _ in range(iterations):
        acc = np.zeros((n, n), complex)
        for k, d in enumerate(div):
            field = np.zeros((big, big), complex)
            field[n // 2 : n // 2 + n, n // 2 : n // 2 + n] = amp * np.exp(
                1j * (phase + 2 * np.pi * d / LAM)
            )
            focal = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
            crop = focal[n // 2 : n // 2 + n, n // 2 : n // 2 + n]
            scale = np.sqrt(np.sum(np.abs(crop) ** 2) / np.sum(target[k] ** 2))
            focal[n // 2 : n // 2 + n, n // 2 : n // 2 + n] = (
                scale * target[k] * np.exp(1j * np.angle(crop))
            )
            back = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(focal)))
            acc += back[n // 2 : n // 2 + n, n // 2 : n // 2 + n] * np.exp(-2j * np.pi * d / LAM)
        phase = np.angle(acc)
    dt = time.perf_counter() - t0
    return {"seconds": dt, "iterations_per_s": iterations / dt}


def run_solvephase_gs(pupil, div, images, iterations: int, device: str) -> dict[str, Any]:
    n = images.shape[-1]
    model = sp.FocalPlaneModel(
        pupil, LAM, n, sampling=2.0, diversity=div, offset=-0.5, device=device
    )
    sp.gerchberg_saxton(model, images, iterations=3, unwrap=False)
    model.backend.synchronize()
    t0 = time.perf_counter()
    sp.gerchberg_saxton(
        model, images, iterations=iterations, check_every=iterations, tol=0.0, unwrap=False
    )
    model.backend.synchronize()
    dt = time.perf_counter() - t0
    return {"seconds": dt, "iterations_per_s": iterations / dt}


def run_numpy_hio(
    mags: np.ndarray, support: np.ndarray, iterations: int, beta: float = 0.9
) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    x = rng.uniform(size=mags.shape) * support
    m = np.fft.ifftshift(mags)
    s = np.fft.ifftshift(support)
    x = np.fft.ifftshift(x)
    t0 = time.perf_counter()
    for _ in range(iterations):
        f = np.fft.fft2(x)
        y = np.fft.ifft2(m * np.exp(1j * np.angle(f)))
        ok = s & (y.real >= 0)
        x = np.where(ok, y, x - beta * y)
    dt = time.perf_counter() - t0
    return {"seconds": dt, "iterations_per_s": iterations / dt}


def run_solvephase_hio(
    mags, support, iterations: int, device: str, starts: int = 1
) -> dict[str, Any]:
    sp.cdi(mags, support, schedule="hio:5", device=device, starts=starts, constraint="positive")
    t0 = time.perf_counter()
    sp.cdi(
        mags,
        support,
        schedule=f"hio:{iterations}",
        device=device,
        starts=starts,
        check_every=iterations,
        tol=0.0,
        constraint="positive",
    )
    dt = time.perf_counter() - t0
    return {"seconds": dt, "iterations_per_s": starts * iterations / dt}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    devices = ["cpu"] + (["gpu"] if sp.gpu_available() else [])
    rows: list[dict[str, Any]] = []

    def add(problem: str, method: str, row: dict[str, Any]) -> None:
        row.update({"problem": problem, "method": method})
        rows.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))

    n, n_modes = (64, 20) if args.quick else (128, 36)
    pupil, basis, truth, div, images = focal_problem(n, n_modes)
    label = f"focal {n}x{n}, {n_modes} modes, 2 images"
    for device in devices:
        add(
            label,
            f"solvephase LM ({device})",
            run_solvephase(pupil, basis, truth, div, images, device),
        )
    try:
        import hcipy  # noqa: F401

        add(
            label,
            "HCIPy + SciPy least_squares",
            run_hcipy_least_squares(pupil, basis, truth, div, images),
        )
        if not args.quick:
            add(label, "HCIPy + SciPy L-BFGS-B", run_hcipy_lbfgs(pupil, basis, truth, div, images))
    except ImportError:
        print("hcipy not installed; skipping HCIPy baselines", file=sys.stderr)

    iters = 20 if args.quick else 100
    label = f"Misell/GS {n}x{n}, 2 images (iterations/s)"
    add(label, "NumPy textbook loop", run_numpy_misell(pupil, truth, div, images, iters))
    for device in devices:
        add(label, f"solvephase ({device})", run_solvephase_gs(pupil, div, images, iters, device))

    size = 128 if args.quick else 256
    obj = np.zeros((size, size))
    yy, xx = np.mgrid[:size, :size] - (size - 1) / 2
    inside = np.hypot(yy, xx) < size / 5
    obj[inside] = np.random.default_rng(1).uniform(0.5, 1.0, inside.sum())
    mags = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(obj))))
    support = np.hypot(yy, xx) < size / 4.5
    label = f"CDI HIO {size}x{size} (iterations/s)"
    add(label, "NumPy textbook loop", run_numpy_hio(mags, support, iters))
    for device in devices:
        add(label, f"solvephase ({device})", run_solvephase_hio(mags, support, iters, device))
        add(
            label,
            f"solvephase ({device}, 16 starts, per start)",
            run_solvephase_hio(mags, support, iters // 2, device, 16),
        )
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump({"environment": _env(), "rows": rows}, handle, indent=2)
    return 0


def _env() -> dict[str, Any]:
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from run import environment

    return environment()


if __name__ == "__main__":
    sys.exit(main())
