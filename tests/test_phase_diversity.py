"""Phase diversity with an unknown extended object (Gonsalves / Paxman reduced metric)."""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest

from conftest import devices
from solvephase.algorithms.phase_diversity import PhaseDiversityProblem, phase_diversity
from solvephase.backend import to_numpy
from solvephase.basis import Basis
from solvephase.focal import FocalPlaneModel, zernike_diversity
from solvephase.metrics import rms, wavefront_error
from solvephase.pupil import Pupil
from solvephase.retrieval import FocalPlaneProblem, solve

WL = 1.6e-6
PV_DEFOCUS = WL / (2.0 * math.sqrt(3.0))  # 1 wave peak-to-valley of unit-RMS Noll Z4


def _blobs(rng: np.random.Generator, n: int, half: float, count: int = 8) -> np.ndarray:
    """Gaussian blobs within ``half`` pixels of the centre of an ``n x n`` grid."""
    y, x = np.mgrid[:n, :n] - (n - 1) / 2.0
    scene = np.full((n, n), 1e-3)
    for _ in range(count):
        cy, cx = rng.uniform(-half, half, 2)
        width = rng.uniform(1.0, 3.0)
        scene += rng.uniform(0.3, 1.0) * np.exp(-0.5 * ((y - cy) ** 2 + (x - cx) ** 2) / width**2)
    return scene


def _convolve(scene: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """Images of ``scene`` through PSFs whose optical axis sits at pixel ``n // 2``."""
    n = scene.shape[-1]
    big = psf.shape[-1]
    pad = np.zeros((psf.shape[0], n, n))
    s0 = n // 2 - big // 2
    pad[:, s0 : s0 + big, s0 : s0 + big] = psf
    kernel = np.fft.rfft2(np.fft.ifftshift(pad, axes=(-2, -1)))
    return np.fft.irfft2(np.fft.rfft2(scene) * kernel, s=(n, n))


@functools.cache
def _simulate(
    n_img: int = 64,
    n_pupil: int = 64,
    n_modes: int = 20,
    rms_waves: float = 0.075,
    photons: float = 1e3,
    seed: int = 1,
    tilt: float = 0.0,
) -> dict:
    """Two images (in focus, 1 wave PV defocus) of a blob scene with Poisson noise.

    The scene is three times wider than the detector and the PSF twice as
    wide, so the data are not periodic and the PSF wings are not truncated.
    ``photons`` is the mean count per pixel; ``tilt`` adds a differential
    x-tilt (RMS OPD in waves) to channel 1.
    """
    rng = np.random.default_rng(seed)
    pupil = Pupil.circular(n_pupil)
    basis = Basis.zernike(pupil, n_modes, start=4)
    coeffs = rng.standard_normal(n_modes) / np.arange(4, 4 + n_modes) ** 0.8
    coeffs *= rms_waves * WL / np.linalg.norm(coeffs)
    opd = basis.synthesize(coeffs)
    div = zernike_diversity(pupil, 4, [0.0, PV_DEFOCUS])
    true_div = div.copy()
    if tilt:
        true_div[1] += tilt * WL * Basis.zernike(pupil, 1, start=2).mode_maps()[0]
    big = 2 * n_img
    # The model's offset moves the window, so -0.5 puts the axis on pixel big // 2.
    wide = FocalPlaneModel(pupil, WL, big, sampling=2.0, diversity=true_div, offset=(-0.5, -0.5))
    n = 3 * n_img
    scene = _blobs(rng, n, half=12.0 * n_img / 64)
    clean = _convolve(scene, np.asarray(wide.images(opd)))
    c0 = n // 2 - n_img // 2
    crop = (slice(None), slice(c0, c0 + n_img), slice(c0, c0 + n_img))
    clean = clean[crop]
    scale = photons * n_img**2 / clean[0].sum()
    images = rng.poisson(np.maximum(clean * scale, 0.0)).astype(np.float64)
    return {
        "pupil": pupil,
        "basis": basis,
        "coeffs": coeffs,
        "opd": opd,
        "diversity": div,
        "images": images,
        "scene": scene[c0 : c0 + n_img, c0 : c0 + n_img] * scale,
    }


def _model(sim: dict, n_img: int = 64, **kw: object) -> FocalPlaneModel:
    return FocalPlaneModel(sim["pupil"], WL, n_img, sampling=2.0, diversity=sim["diversity"], **kw)


def _relative_error(result_opd: object, sim: dict) -> float:
    pupil = sim["pupil"]
    truth = sim["opd"]
    return wavefront_error(result_opd, truth, pupil, remove="tiptilt") / rms(
        truth, pupil, "tiptilt"
    )


# --------------------------------------------------------------------- gradient
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"regularization": 1e-3, "window": "hann"},
        {"fit_tilt": True, "frequency_mask": 0.8},
        {"basis": None, "fit_amplitude": True, "window": None},
    ],
)
def test_gradient_matches_finite_differences(options: dict) -> None:
    sim = _simulate(n_img=32, n_pupil=32, n_modes=10)
    kw = {"basis": sim["basis"], **options}
    problem = PhaseDiversityProblem(_model(sim, 32), sim["images"], **kw)
    rng = np.random.default_rng(0)
    x = 0.2 * rng.standard_normal(problem.n_params)
    if problem.fit_amplitude:
        x[problem._amp_sl] *= 0.05  # keep the amplitude positive
    _, grad = problem.objective(x)
    for _ in range(3):
        d = rng.standard_normal(problem.n_params)
        h = 1e-5
        fd = (problem.value(x + h * d) - problem.value(x - h * d)) / (2 * h)
        assert float(grad @ d) == pytest.approx(fd, rel=1e-6, abs=1e-8 * abs(problem.value(x)))


def test_metric_equals_reduced_form() -> None:
    """The residual form on the real-FFT half plane equals Gonsalves' full-plane metric."""
    sim = _simulate(n_img=32, n_pupil=32, n_modes=10)
    problem = PhaseDiversityProblem(_model(sim, 32), sim["images"], basis=sim["basis"])
    x = 0.1 * np.random.default_rng(2).standard_normal(problem.n_params)
    otf = np.fft.fft2(np.asarray(problem.model.forward(problem._channel_phase(x)).images))
    d = np.fft.fft2(np.asarray(problem.apodized))
    q = np.sum(d * otf.conj(), axis=0)
    p = np.sum(np.abs(otf) ** 2, axis=0) + problem.regularization
    terms = np.sum(np.abs(d) ** 2, axis=0) - np.abs(q) ** 2 / p
    rho = np.hypot(np.fft.fftfreq(32)[:, None], np.fft.fftfreq(32)[None, :])
    mask = (rho < problem.cutoff) & (rho > 0)
    assert np.count_nonzero(mask) > 400
    reduced = np.sum(terms[mask]) / problem._norm
    assert problem.value(x) == pytest.approx(reduced, rel=1e-9)


def test_metric_invariant_to_scene_shift() -> None:
    """A common shift of the data moves the object, not the wavefront."""
    sim = _simulate(n_img=32, n_pupil=32, n_modes=10)
    model = _model(sim, 32)
    x = 0.1 * np.random.default_rng(3).standard_normal(10)
    shifted = np.roll(sim["images"], (3, -5), axis=(-2, -1))
    a = PhaseDiversityProblem(model, sim["images"], basis=sim["basis"], window=None)
    b = PhaseDiversityProblem(model, shifted, basis=sim["basis"], window=None)
    assert b.value(x) == pytest.approx(a.value(x), rel=1e-10)


# --------------------------------------------------------------------- recovery
def test_recovers_aberration_and_object_from_extended_scene() -> None:
    sim = _simulate()
    result = phase_diversity(_model(sim), sim["images"], basis=sim["basis"])
    assert result.method == "phase_diversity"
    assert result.converged
    assert _relative_error(result.opd, sim) < 0.1
    # Coefficients are RMS OPD in metres, defocus and up.
    assert np.linalg.norm(result.coefficients - sim["coeffs"]) < 0.1 * np.linalg.norm(sim["coeffs"])
    obj = np.asarray(result.extra["object"])
    assert obj.shape == sim["images"].shape[1:]
    assert np.corrcoef(obj.ravel(), sim["scene"].ravel())[0, 1] > 0.97
    assert obj.sum() == pytest.approx(sim["scene"].sum(), rel=0.02)  # data units
    assert result.extra["otf"].shape == (2, 64, 33)
    assert result.history[-1] < result.history[0]
    # The model of the apodized data fits it to the noise level.
    resid = np.asarray(result.extra["apodized_images"]) - np.asarray(result.model_images)
    assert np.std(resid) < 1.5 * math.sqrt(np.mean(sim["images"]))


def test_point_source_agrees_with_focal_plane_problem() -> None:
    sim = _simulate(n_img=64, n_pupil=64)
    model = _model(sim)
    rng = np.random.default_rng(7)
    images = rng.poisson(1e6 * np.asarray(model.images(sim["opd"]))).astype(np.float64)
    pd = phase_diversity(model, images, basis=sim["basis"])
    ref = solve(FocalPlaneProblem(model, images, basis=sim["basis"], loss="poisson"))
    scale = rms(sim["opd"], sim["pupil"], "tiptilt")
    assert _relative_error(ref.opd, sim) < 0.02
    assert _relative_error(pd.opd, sim) < 0.03
    assert wavefront_error(pd.opd, ref.opd, sim["pupil"], remove="tiptilt") < 0.03 * scale
    # The object is a band-limited point at the optical axis (31.5, 31.5) carrying the flux.
    obj = np.asarray(pd.extra["object"])
    peak = np.unravel_index(np.argmax(obj), obj.shape)
    assert peak[0] in (31, 32) and peak[1] in (31, 32)
    assert obj.sum() == pytest.approx(images.mean(axis=0).sum(), rel=0.01)


def test_object_is_registered_with_the_optical_axis() -> None:
    """With a window offset the point-source object lands on the axis pixel."""
    pupil = Pupil.circular(32)
    div = zernike_diversity(pupil, 4, [0.0, PV_DEFOCUS])
    model = FocalPlaneModel(pupil, WL, 33, sampling=2.0, diversity=div, offset=(3.0, -2.0))
    images = 1e5 * np.asarray(model.images())
    result = PhaseDiversityProblem(model, images, basis=5, window=None)
    obj = np.asarray(result.object_estimate(np.zeros(result.n_params)))
    assert np.unravel_index(np.argmax(obj), obj.shape) == (16 - 3, 16 + 2)


def test_fit_tilt_registers_channels() -> None:
    sim = _simulate(tilt=0.05)
    plain = phase_diversity(_model(sim), sim["images"], basis=sim["basis"])
    fitted = phase_diversity(_model(sim), sim["images"], basis=sim["basis"], fit_tilt=True)
    assert _relative_error(fitted.opd, sim) < 0.1 < _relative_error(plain.opd, sim)
    assert fitted.tilts is not None
    np.testing.assert_allclose(fitted.tilts[1], [0.05 * WL, 0.0], atol=0.005 * WL)


def test_equalize_flux_makes_exposure_irrelevant() -> None:
    sim = _simulate(n_img=32, n_pupil=32, n_modes=10)
    model = _model(sim, 32)
    brighter = sim["images"] * np.array([1.0, 3.0])[:, None, None]
    x = 0.1 * np.random.default_rng(4).standard_normal(10)
    a = PhaseDiversityProblem(model, sim["images"], basis=sim["basis"])
    b = PhaseDiversityProblem(model, brighter, basis=sim["basis"])
    assert b.value(x) == pytest.approx(a.value(x), rel=1e-9)
    c = PhaseDiversityProblem(model, brighter, basis=sim["basis"], equalize_flux=False)
    assert abs(c.value(x) - a.value(x)) > 0.1 * a.value(x)


def test_start_and_coarse_to_fine() -> None:
    sim = _simulate()
    model = _model(sim)
    staged = phase_diversity(model, sim["images"], basis=sim["basis"], coarse_to_fine=(5, 10))
    assert _relative_error(staged.opd, sim) < 0.1
    assert len(staged.history) > staged.n_iter  # one start value per stage
    restarted = phase_diversity(model, sim["images"], basis=sim["basis"], start=staged)
    assert restarted.n_iter < staged.n_iter
    assert _relative_error(restarted.opd, sim) < 0.1
    with pytest.raises(ValueError, match="modal basis"):
        phase_diversity(model, sim["images"], basis=None, coarse_to_fine=(5,))


def test_int_basis_starts_at_defocus() -> None:
    sim = _simulate(n_img=32, n_pupil=32, n_modes=10)
    problem = PhaseDiversityProblem(_model(sim, 32), sim["images"], basis=6)
    assert problem.basis.labels == ["Z4", "Z5", "Z6", "Z7", "Z8", "Z9"]


# ------------------------------------------------------------------- validation
def test_input_validation() -> None:
    sim = _simulate(n_img=32, n_pupil=32, n_modes=10)
    model = _model(sim, 32)
    images = sim["images"]
    single = FocalPlaneModel(sim["pupil"], WL, 32, sampling=2.0)
    with pytest.raises(ValueError, match="at least two images"):
        PhaseDiversityProblem(single, images[0])
    with pytest.raises(ValueError, match="model expects"):
        PhaseDiversityProblem(model, images[:, :16])
    with pytest.raises(ValueError, match="window"):
        PhaseDiversityProblem(model, images, window="gauss")
    with pytest.raises(ValueError, match="window has shape"):
        PhaseDiversityProblem(model, images, window=np.ones((3, 3)))
    with pytest.raises(ValueError, match="regularization"):
        PhaseDiversityProblem(model, images, regularization="large")
    with pytest.raises(ValueError, match="non-negative"):
        PhaseDiversityProblem(model, images, regularization=-1.0)
    with pytest.raises(ValueError, match="frequency_mask"):
        PhaseDiversityProblem(model, images, frequency_mask="nyquist")
    bad = images.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        PhaseDiversityProblem(model, bad)
    coarse = FocalPlaneModel(sim["pupil"], WL, 32, sampling=1.2, diversity=sim["diversity"])
    with pytest.warns(UserWarning, match="undersampled"), pytest.raises(ValueError, match="number"):
        PhaseDiversityProblem(coarse, images)
    with pytest.warns(UserWarning, match="undersampled"):
        problem = PhaseDiversityProblem(
            coarse, images, regularization=1e-4, object_regularization=1e-3
        )
    assert problem.noise_power is None


# ------------------------------------------------------------------------- GPU
@pytest.mark.parametrize("device", devices())
def test_device_parity(device: str) -> None:
    sim = _simulate()
    model = _model(sim, device=device)
    result = phase_diversity(model, sim["images"], basis=sim["basis"])
    assert result.device == device
    assert _relative_error(result.opd, sim) < 0.1
    host = result.to_numpy()
    reference = phase_diversity(_model(sim), sim["images"], basis=sim["basis"])
    np.testing.assert_allclose(host.coefficients, reference.coefficients, atol=2e-3 * WL)
    obj = np.asarray(to_numpy(host.extra["object"]))
    ref_obj = np.asarray(reference.extra["object"])
    assert np.corrcoef(obj.ravel(), ref_obj.ravel())[0, 1] > 0.999


# ------------------------------------------------------------------------ slow
@pytest.mark.slow
def test_larger_field_and_aberration() -> None:
    """128-pixel field, 35 modes, 0.15 waves RMS, coarse to fine."""
    sim = _simulate(n_img=128, n_pupil=64, n_modes=35, rms_waves=0.15, photons=300.0, seed=4)
    result = phase_diversity(
        _model(sim, 128), sim["images"], basis=sim["basis"], coarse_to_fine=(10, 20)
    )
    assert _relative_error(result.opd, sim) < 0.1
