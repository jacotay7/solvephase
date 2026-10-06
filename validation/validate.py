"""Physics and statistics validation of solvephase.

Each check compares solvephase with theory or an independent code and records
a metric with its acceptance threshold. The script writes ``validation.json``
(metrics, thresholds, pass/fail) and ``validation.png`` (figures) and exits
non-zero if any check fails.

Checks
------
1. ``hcipy``: focal-plane images agree with HCIPy's Fraunhofer propagator
   (independent implementation) for FFT and MFT sampling and broadband light.
2. ``airy``: the peak of an unaberrated circular pupil's image matches the
   analytic value ``A theta^2 / lambda^2`` (normalization), and its first dark
   ring falls at 1.22 lambda/D.
3. ``cramer_rao``: the Poisson maximum-likelihood estimator of modal
   coefficients is efficient - its Monte-Carlo variance matches the
   Cramer-Rao lower bound from the Fisher information.
4. ``capture``: :func:`solvephase.retrieve` recovers random aberrations from
   two defocus-diversity images; success rate versus aberration size.
5. ``precision``: single and double precision reach the same solution.
6. ``extensions``: algorithm families (CDI, TIE, generic, phase diversity,
   LIFT, Fast & Furious) reach their documented accuracy on reference
   problems, when their modules are present.

Usage::

    python validation/validate.py --output validation-artifacts
    python validation/validate.py --quick --output validation-artifacts   # CI
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

import solvephase as sp

LAM = 1.0e-6


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


# --------------------------------------------------------------------------- 1
def validate_hcipy(quick: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import hcipy as hc
    except ImportError:
        print("[SKIP] hcipy not installed")
        return [], {}
    warnings.filterwarnings("ignore", module="hcipy")
    n, d = 64, 1.0
    pupil = sp.Pupil.circular(n, d, obscuration=0.2, spiders=3, spider_width=0.03, supersample=1)
    opd = sp.random_aberration(pupil, 0.08 * LAM, n_modes=30, seed=1)
    grid = hc.make_pupil_grid(n, d)
    field = hc.Field(pupil.amplitude.ravel(), grid)
    out, figures = [], {}
    for q, num_airy, lams in [
        (2, 16, [LAM]),
        (3, 10, [LAM]),
        (1.37, 12, [LAM]),
        (2, 12, [0.9 * LAM, LAM, 1.1 * LAM]),
    ]:
        focal = hc.make_focal_grid(q=q, num_airy=num_airy, spatial_resolution=LAM / d)
        ref = 0.0
        for lam in lams:
            wf = hc.Wavefront(field * np.exp(2j * np.pi * opd.ravel() / lam), lam)
            ref = ref + hc.FraunhoferPropagator(grid, focal)(wf).power.shaped
        shape = tuple(int(s) for s in focal.shape)
        model = sp.FocalPlaneModel(
            pupil,
            lams,
            shape,
            pixel_scale=float(focal.delta[0]),
            offset=-0.5 if shape[0] % 2 == 0 else 0.0,
        )
        ours = sp.to_numpy(model.images(opd))[0]
        a = np.asarray(ref) / np.max(ref)
        b = ours / ours.max()
        diff = float(np.max(np.abs(a - b)))
        label = f"hcipy q={q} {len(lams)} wavelength(s) [{model.propagator.method}]"
        out.append(check(label, diff, 1e-10))
        figures.setdefault("hcipy", (a, b))
    return out, figures


# --------------------------------------------------------------------------- 2
def validate_airy() -> list[dict[str, Any]]:
    from scipy.special import j1  # noqa: F401  (documents the reference function)

    # A binary (not anti-aliased) pupil approximates the continuous disk, whose
    # on-axis intensity is A theta^2 / lambda^2 per pixel for a unit-flux image.
    pupil = sp.Pupil.circular(512, 1.0, supersample=1)
    sampling = 8.0
    model = sp.FocalPlaneModel(pupil, LAM, 64, sampling=sampling, offset=-0.5)
    img = sp.to_numpy(model.images())[0]
    theta = model.pixel_scale
    peak_theory = (np.pi / 4) * theta**2 / LAM**2
    out = [
        check("airy peak normalization (relative error)", abs(img.max() / peak_theory - 1), 2e-3)
    ]
    # First dark ring: minimum of the radial profile near 1.22 lambda/D.
    cut = img[32, 32:]
    r = np.arange(cut.size) / sampling
    window = (r > 0.9) & (r < 1.6)
    first_zero = r[window][np.argmin(cut[window])]
    out.append(
        check("airy first dark ring [lambda/D] error", abs(first_zero - 1.2197), 1.0 / sampling)
    )
    return out


# --------------------------------------------------------------------------- 3
def validate_cramer_rao(quick: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pupil = sp.Pupil.circular(48, 1.0, obscuration=0.15)
    basis = sp.Basis.zernike(pupil, 12, start=4)
    opd = sp.random_aberration(pupil, 0.05 * LAM, n_modes=12, start=4, seed=11)
    div = sp.zernike_diversity(pupil, 4, [0.0, 0.25 * LAM])
    model = sp.FocalPlaneModel(pupil, LAM, 40, sampling=2.0, diversity=div)
    photons = 2e5
    expected = sp.simulate_images(model, opd, photons=photons, background=2.0, noise=False)
    truth_problem = sp.FocalPlaneProblem(model, expected, basis=basis, loss="poisson")
    x_true = truth_problem.initial(opd)
    _, _, fisher = truth_problem.gauss_newton(x_true)
    crb = np.linalg.inv(np.asarray(fisher))  # internal units: radians at LAM, plus log-flux
    k_wave = 2 * np.pi / LAM
    crb_coeff = np.diag(crb)[: basis.n_modes] / k_wave**2
    trials = 40 if quick else 200
    rng = np.random.default_rng(5)
    estimates = []
    for _ in range(trials):
        data = rng.poisson(expected).astype(float)
        prob = sp.FocalPlaneProblem(
            model, data, basis=basis, loss="poisson", fit_background=False, background=2.0
        )
        estimates.append(sp.solve(prob, method="lm", start=opd).coefficients)
    est = np.asarray(estimates)
    var = est.var(axis=0, ddof=1)
    ratio = var / crb_coeff
    bias = np.abs(est.mean(axis=0) - basis.fit(opd)) / np.sqrt(crb_coeff)
    # The ratio of a sample variance to its expectation has relative spread sqrt(2/(N-1)).
    tol = 1 + 4 * np.sqrt(2 / (trials - 1))
    out = [
        check("CRB efficiency: mean variance/CRB", float(np.mean(ratio)), tol),
        check(
            "CRB efficiency: mean variance/CRB (lower)", float(np.mean(ratio)), 1 / tol, below=False
        ),
        check(
            "estimator bias [CRB sigma], max over modes",
            float(np.max(bias)),
            4.0 * np.sqrt(1 / trials) * 3,
        ),
    ]
    return out, {"cramer_rao": (np.sqrt(crb_coeff) * 1e9, np.sqrt(var) * 1e9, basis.labels)}


# --------------------------------------------------------------------------- 4
def validate_capture(quick: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pupil = sp.Pupil.circular(48, 1.0, obscuration=0.15)
    model = sp.FocalPlaneModel(
        pupil, LAM, 48, sampling=2.0, diversity=sp.zernike_diversity(pupil, 4, [0.0, 0.3 * LAM])
    )
    levels = [0.05, 0.15, 0.25] if quick else [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
    trials = 3 if quick else 10
    rates = []
    for level in levels:
        ok = 0
        for t in range(trials):
            opd = sp.random_aberration(pupil, level * LAM, n_modes=20, start=4, seed=100 + t)
            imgs = sp.simulate_images(model, opd, photons=1e6, seed=200 + t)
            res = sp.retrieve(imgs, pupil, LAM, sampling=2.0, diversity=[0.0, 0.3 * LAM], basis=24)
            ok += sp.wavefront_error(res.opd, opd, pupil, remove="tiptilt") < 0.05 * level * LAM
        rates.append(ok / trials)
        print(f"  capture at {level:.2f} waves RMS: {ok}/{trials}")
    out = [
        check(
            "capture success rate at <= 0.3 waves RMS",
            min(r for lv, r in zip(levels, rates) if lv <= 0.3),
            1.0,
            below=False,
        )
    ]
    return out, {"capture": (levels, rates)}


# --------------------------------------------------------------------------- 5
def validate_precision() -> list[dict[str, Any]]:
    pupil = sp.Pupil.circular(48, 1.0, obscuration=0.15)
    basis = sp.Basis.zernike(pupil, 20)
    opd = sp.random_aberration(pupil, 0.08 * LAM, n_modes=20, start=4, seed=3)
    div = sp.zernike_diversity(pupil, 4, [0.0, 0.25 * LAM])
    data = sp.simulate_images(
        sp.FocalPlaneModel(pupil, LAM, 40, sampling=2.0, diversity=div), opd, photons=1e7, seed=1
    )
    results = {}
    for precision in ("single", "double"):
        model = sp.FocalPlaneModel(pupil, LAM, 40, sampling=2.0, diversity=div, precision=precision)
        results[precision] = sp.solve(
            sp.FocalPlaneProblem(model, data, basis=basis, loss="poisson"), method="lm"
        )
    diff = sp.wavefront_error(results["single"].opd, results["double"].opd, pupil)
    return [check("single vs double precision solution difference [nm]", diff * 1e9, 0.05)]


# --------------------------------------------------------------------------- 6
def validate_extensions(quick: bool) -> list[dict[str, Any]]:
    try:
        from extensions import run  # type: ignore[import-not-found]
    except ImportError:
        return []
    return run(check, quick)


def save_figure(path: Path, figures: dict[str, Any]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    if "hcipy" in figures:
        a, b = figures["hcipy"]
        axes[0].imshow(np.log10(b + 1e-8), origin="lower", cmap="magma")
        axes[0].set_title("solvephase image (log10, normalized)")
        im = axes[1].imshow(a - b, origin="lower", cmap="RdBu_r")
        axes[1].set_title("HCIPy - solvephase")
        fig.colorbar(im, ax=axes[1], fraction=0.046)
    if "cramer_rao" in figures:
        crb, std, labels = figures["cramer_rao"]
        x = np.arange(len(crb))
        axes[2].bar(x - 0.2, crb, 0.4, label="Cramer-Rao bound")
        axes[2].bar(x + 0.2, std, 0.4, label="Monte-Carlo std")
        axes[2].set_xticks(x, labels, rotation=90)
        axes[2].set_ylabel("coefficient std [nm]")
        axes[2].legend()
        axes[2].set_title("Poisson ML efficiency")
    if "capture" in figures:
        levels, rates = figures["capture"]
        axes[3].plot(levels, rates, "o-")
        axes[3].set_xlabel("aberration RMS [waves]")
        axes[3].set_ylabel("success rate")
        axes[3].set_ylim(-0.05, 1.05)
        axes[3].set_title("retrieve() capture range (2 images)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", default="validation-artifacts")
    args = parser.parse_args(argv)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    checks: list[dict[str, Any]] = []
    figures: dict[str, Any] = {}
    hc_checks, hc_fig = validate_hcipy(args.quick)
    checks += hc_checks
    figures.update(hc_fig)
    checks += validate_airy()
    crb_checks, crb_fig = validate_cramer_rao(args.quick)
    checks += crb_checks
    figures.update(crb_fig)
    cap_checks, cap_fig = validate_capture(args.quick)
    checks += cap_checks
    figures.update(cap_fig)
    checks += validate_precision()
    checks += validate_extensions(args.quick)
    report = {
        "solvephase": sp.__version__,
        "quick": args.quick,
        "seconds": time.perf_counter() - t0,
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
    }
    (out_dir / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_figure(out_dir / "validation.png", figures)
    n_fail = sum(not c["passed"] for c in checks)
    print(
        f"{len(checks) - n_fail}/{len(checks)} checks passed in {report['seconds']:.0f} s -> {out_dir}"
    )
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
