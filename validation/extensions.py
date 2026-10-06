"""Validation checks for the algorithm families beyond focal-plane retrieval.

Each check runs a reference problem with known truth and compares the error
with a documented threshold. ``run(check, quick)`` is called by
``validate.py``.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

import solvephase as sp

LAM = 1.0e-6
Check = Callable[..., dict[str, Any]]


def _phase_diversity(check: Check, quick: bool) -> list[dict[str, Any]]:
    pupil = sp.Pupil.circular(64, 1.0)
    truth = sp.random_aberration(pupil, 0.075 * LAM, n_modes=20, start=4, seed=1)
    model = sp.FocalPlaneModel(
        pupil, LAM, 64, sampling=2.0, diversity=sp.zernike_diversity(pupil, 4, [0.0, LAM / (2 * math.sqrt(3))])
    )
    rng = np.random.default_rng(2)
    yy, xx = np.mgrid[:64, :64] - 31.5
    scene = np.full((64, 64), 1e-3)
    for _ in range(8):
        cy, cx = rng.uniform(-12, 12, 2)
        scene += rng.uniform(0.3, 1.0) * np.exp(-0.5 * ((yy - cy) ** 2 + (xx - cx) ** 2) / rng.uniform(1, 3) ** 2)
    psf = sp.to_numpy(model.images(truth))
    clean = np.fft.irfft2(np.fft.rfft2(scene) * np.fft.rfft2(np.fft.ifftshift(psf, axes=(-2, -1))), s=(64, 64))
    images = rng.poisson(clean * 1000 / clean[0].mean()).astype(float)
    res = sp.phase_diversity(model, images, basis=sp.Basis.zernike(pupil, 20, start=4))
    rel = sp.wavefront_error(res.opd, truth, pupil, remove="tiptilt") / sp.rms(truth, pupil, "tiptilt")
    return [check("phase diversity (extended scene) relative wavefront error", rel, 0.1)]


def _lift(check: Check, quick: bool) -> list[dict[str, Any]]:
    pupil = sp.Pupil.circular(32, 8.0, obscuration=0.1)
    sensor = sp.LIFT(pupil, 1.6e-6, 32, sampling=2.0, n_modes=10, read_noise=1.0)
    truth = sp.random_aberration(pupil, 0.08 * 1.6e-6, n_modes=10, seed=4)
    coeffs = sensor.basis.fit(truth)
    bound = np.diag(sensor.crlb(coeffs, photons=1e5))[: sensor.basis.n_modes]
    rng = np.random.default_rng(6)
    expected = sp.simulate_images(sensor.model, truth, photons=1e5, noise=False)[0]
    trials = 40 if quick else 150
    est = []
    for _ in range(trials):
        img = rng.poisson(expected) + rng.normal(0, 1.0, expected.shape)
        est.append(sensor.estimate(img, start=truth).coefficients)
    ratio = float(np.mean(np.var(np.asarray(est), axis=0, ddof=1) / bound))
    tol = 1 + 4 * math.sqrt(2 / (trials - 1))
    return [check("LIFT variance / Cramer-Rao bound", ratio, tol), check("LIFT variance / CRB (lower)", ratio, 1 / tol, below=False)]


def _fast_furious(check: Check, quick: bool) -> list[dict[str, Any]]:
    pupil = sp.Pupil.circular(64, 1.0)
    ff = sp.FastAndFurious(pupil, LAM, 64, sampling=2.0)
    ncpa = sp.random_aberration(pupil, 0.1 * LAM, n_modes=20, start=2, seed=6)
    loop = sp.simulate_closed_loop(ff, ncpa, 20, gain=0.5)
    ratio = float(loop.residual_rms[-1] / loop.residual_rms[0])
    return [check("Fast & Furious residual after 20 steps / initial", ratio, 0.2)]


def _cdi(check: Check, quick: bool) -> list[dict[str, Any]]:
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:48, :48] - 23.5
    outline = np.hypot(yy, xx) < 18 * (1 + 0.25 * np.cos(3 * np.arctan2(yy, xx)))
    trials = 3 if quick else 8
    ok = 0
    for t in range(trials):
        obj = np.where(outline, 0.3 + gaussian_filter(rng.uniform(size=(48, 48)), 1.5), 0.0)
        data = sp.simulate_cdi(obj, oversampling=2)
        support = np.pad(outline, 24)
        res = sp.cdi(np.sqrt(data.intensity), support, schedule="hio:500,er:100", starts=4, constraint="positive", seed=t)
        _, err = sp.align_object(res.object, data.object)
        ok += err < 1e-3
    return [check("CDI (HIO+ER, 4 starts) noise-free success rate", ok / trials, 1.0, below=False)]


def _tie(check: Check, quick: bool) -> list[dict[str, Any]]:
    n, pitch, wl = 256, 10e-6, 633e-9
    x = (np.arange(n) - (n - 1) / 2) * pitch
    xx, yy = np.meshgrid(x, x)
    r = np.hypot(xx, yy)
    radius = 0.8e-3
    rho, theta = r / radius, np.arctan2(yy, xx)
    phase = 1.5 * (2 * rho**2 - 1) + 1.0 * rho**2 * np.cos(2 * theta)
    amplitude = 0.5 * (1 - np.tanh((r - radius) / (2 * pitch)))
    z = [-1e-3, 0.0, 1e-3]
    stack = sp.simulate_defocus_stack(amplitude * np.exp(1j * phase), z, pitch, wl)
    res = sp.tie(stack, z, pitch=pitch, wavelength=wl, method="pcg")
    inner = r < 0.9 * radius
    err = float(np.std((sp.to_numpy(res.phase) - phase)[inner]))
    return [check("TIE (masked PCG) phase error [rad RMS], soft-edged aperture", err, 0.02)]


def _wirtinger(check: Check, quick: bool) -> list[dict[str, Any]]:
    try:
        from solvephase.algorithms.wirtinger import relative_error, wirtinger
        from solvephase.operators import MatrixOperator
    except ImportError:
        return []
    n = 64
    trials = 3 if quick else 10
    ok = 0
    for t in range(trials):
        op = MatrixOperator.gaussian(8 * n, n, seed=t)
        rng = np.random.default_rng(100 + t)
        truth = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        y = np.abs(sp.to_numpy(op.forward(truth))) ** 2
        res = wirtinger(op, y, method="raf", seed=t)
        ok += relative_error(res.x, truth) < 1e-5
    return [check("Reweighted amplitude flow (m = 8n, Gaussian) exact-recovery rate", ok / trials, 1.0, below=False)]


def run(check: Check, quick: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for fn in (_phase_diversity, _lift, _fast_furious, _cdi, _tie, _wirtinger):
        out += fn(check, quick)
    return out
