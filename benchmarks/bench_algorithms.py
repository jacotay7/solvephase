"""Benchmark cases for the algorithm families (registered by ``run.py``)."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

import solvephase as sp

LAM = 1.0e-6


def register(case: Callable[[str], Callable[..., Any]], timeit: Callable[..., float]) -> None:
    @case("cdi")
    def bench_cdi(backend: Any, quick: bool) -> list[dict[str, Any]]:
        """HIO/RAAR iterations per second, one start and 16 batched starts."""
        rows = []
        for size in [128, 256] if quick else [256, 512, 1024]:
            yy, xx = np.mgrid[:size, :size] - (size - 1) / 2
            obj = np.where(np.hypot(yy, xx) < size / 5, 1.0, 0.0)
            mags = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(obj))))
            support = np.hypot(yy, xx) < size / 4.5
            for algorithm in ("hio", "raar"):
                for starts in (1, 16):
                    if starts == 16 and size > 512:
                        continue
                    iters = 20 if quick else 100

                    def run(
                        starts: int = starts, algorithm: str = algorithm, iters: int = iters
                    ) -> None:
                        sp.cdi(
                            mags,
                            support,
                            schedule=f"{algorithm}:{iters}",
                            starts=starts,
                            device=backend,
                            check_every=iters,
                            tol=0.0,
                            constraint="positive",
                        )

                    t = timeit(run, backend, repeats=3)
                    rows.append(
                        {
                            "size": size,
                            "algorithm": algorithm,
                            "starts": starts,
                            "iterations_per_s": starts * iters / t,
                        }
                    )
        return rows

    @case("tie")
    def bench_tie(backend: Any, quick: bool) -> list[dict[str, Any]]:
        """Transport-of-intensity solve (3 planes, non-uniform intensity)."""
        rows = []
        for n in [256, 512] if quick else [512, 1024, 2048]:
            pitch = 1e-6
            yy, xx = (np.mgrid[:n, :n] - n / 2) * pitch
            phase = 2.0 * np.exp(-(xx**2 + yy**2) / (2 * (n * pitch / 8) ** 2))
            stack = sp.simulate_defocus_stack(
                np.exp(1j * phase), [-2e-6, 0.0, 2e-6], pitch, 0.55e-6
            )
            stack = backend.asarray(sp.to_numpy(stack), dtype="real")
            for method in ("fft", "dct"):
                t = timeit(
                    lambda method=method: sp.tie(
                        stack,
                        [-2e-6, 0.0, 2e-6],
                        pitch=pitch,
                        wavelength=0.55e-6,
                        method=method,
                        device=backend,
                    ),
                    backend,
                    repeats=3,
                )
                rows.append({"size": n, "method": method, "seconds": t})
        return rows

    @case("lift")
    def bench_lift(backend: Any, quick: bool) -> list[dict[str, Any]]:
        """LIFT estimate of 10 modes from one image (sensor constructed once)."""
        rows = []
        for n in [32, 64] if quick else [32, 64, 128]:
            pupil = sp.Pupil.circular(n, 8.0, obscuration=0.1)
            sensor = sp.LIFT(
                pupil, 1.6e-6, n, sampling=2.0, n_modes=10, read_noise=1.0, device=backend
            )
            truth = sp.random_aberration(pupil, 0.08 * 1.6e-6, n_modes=10, seed=4)
            image = sp.simulate_images(sensor.model, truth, photons=1e5, seed=5)[0]
            t = timeit(lambda: sensor.estimate(image), backend, repeats=5)
            rows.append({"size": n, "seconds": t, "estimates_per_s": 1 / t})
        return rows

    @case("fast_furious")
    def bench_ff(backend: Any, quick: bool) -> list[dict[str, Any]]:
        """Fast & Furious per-step latency."""
        rows = []
        for n in [64, 128] if quick else [64, 128, 256]:
            pupil = sp.Pupil.circular(n, 1.0)
            ff = sp.FastAndFurious(pupil, LAM, n, sampling=2.0, device=backend)
            image = sp.simulate_images(
                ff_model(pupil, n, backend),
                sp.random_aberration(pupil, 0.05 * LAM, seed=1),
                photons=1e6,
                seed=2,
            )[0]
            dm = np.zeros(pupil.shape)
            ff.step(image, dm)
            t = timeit(lambda: ff.step(image, dm), backend, repeats=20)
            rows.append({"size": n, "seconds": t, "steps_per_s": 1 / t})
        return rows

    @case("phase_diversity")
    def bench_pd(backend: Any, quick: bool) -> list[dict[str, Any]]:
        """Extended-object phase diversity: full solve, 20 modes, 2 images."""
        rows = []
        for n in [64] if quick else [64, 128]:
            pupil = sp.Pupil.circular(n, 1.0)
            div = sp.zernike_diversity(pupil, 4, [0.0, LAM / (2 * math.sqrt(3))])
            model = sp.FocalPlaneModel(pupil, LAM, n, sampling=2.0, diversity=div, device=backend)
            truth = sp.random_aberration(pupil, 0.07 * LAM, n_modes=20, start=4, seed=1)
            rng = np.random.default_rng(2)
            yy, xx = np.mgrid[:n, :n] - (n - 1) / 2
            scene = np.full((n, n), 1e-3)
            for _ in range(10):
                cy, cx = rng.uniform(-n / 5, n / 5, 2)
                scene += rng.uniform(0.3, 1) * np.exp(-0.5 * ((yy - cy) ** 2 + (xx - cx) ** 2) / 4)
            psf = sp.to_numpy(model.images(truth))
            kernel = np.fft.rfft2(np.fft.ifftshift(psf, axes=(-2, -1)))
            images = np.fft.irfft2(np.fft.rfft2(scene) * kernel, s=(n, n)) * 1000
            out: dict[str, Any] = {}

            def run() -> None:
                out["r"] = sp.phase_diversity(
                    model, images, basis=sp.Basis.zernike(pupil, 20, start=4)
                )

            t = timeit(run, backend, repeats=2)
            err = sp.wavefront_error(sp.to_numpy(out["r"].opd), truth, pupil, remove="tiptilt")
            rows.append({"size": n, "seconds": t, "error_nm": err * 1e9})
        return rows

    try:
        from solvephase.algorithms.wirtinger import wirtinger
        from solvephase.operators import CodedDiffractionOperator
    except ImportError:  # pragma: no cover
        return

    @case("wirtinger")
    def bench_wirtinger(backend: Any, quick: bool) -> list[dict[str, Any]]:
        """Coded-diffraction phase retrieval (6 masks): time to 1e-6 relative error."""
        rows = []
        for n in [64] if quick else [64, 128, 256]:
            op = CodedDiffractionOperator.random((n, n), 6, seed=1, backend=backend)
            rng = np.random.default_rng(3)
            x = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
            y = abs(sp.to_numpy(op.forward(backend.asarray(x, dtype="complex")))) ** 2
            for method in ("raf", "lbfgs"):
                out: dict[str, Any] = {}

                def run(method: str = method) -> None:
                    out["r"] = wirtinger(op, y, method=method, tol=1e-6)

                t = timeit(run, backend, repeats=2)
                rows.append(
                    {"size": n, "method": method, "seconds": t, "iterations": out["r"].n_iter}
                )
        return rows


def ff_model(pupil: Any, n: int, backend: Any) -> Any:
    return sp.FocalPlaneModel(pupil, LAM, n, sampling=2.0, device=backend)
