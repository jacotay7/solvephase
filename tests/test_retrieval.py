from __future__ import annotations

import numpy as np
import pytest

import solvephase as sp
from conftest import devices
from solvephase import Basis, FocalPlaneModel, FocalPlaneProblem, Pupil, solve, zernike_diversity

LAM = 1e-6


@pytest.fixture(scope="module")
def setup():
    pupil = Pupil.circular(48, 1.0, obscuration=0.15)
    basis = Basis.zernike(pupil, 20)
    opd = sp.random_aberration(pupil, 0.06 * LAM, n_modes=18, start=4, seed=3)
    div = zernike_diversity(pupil, 4, [0.0, 0.25 * LAM])
    model = FocalPlaneModel(pupil, LAM, 40, sampling=2.0, diversity=div)
    images = sp.simulate_images(model, opd, photons=1e6, background=5.0, seed=4)
    return pupil, basis, opd, model, images


@pytest.mark.parametrize("loss", ["gaussian", "poisson", "amplitude"])
def test_objective_gradient_matches_finite_differences(setup, loss, rng) -> None:
    _, basis, _, model, images = setup
    prob = FocalPlaneProblem(
        model,
        images,
        basis=basis,
        loss=loss,
        fit_background=True,
        fit_tilt=True,
        regularization=1e-3,
    )
    x = prob.initial() + 1e-2 * rng.standard_normal(prob.n_params)
    _, g = prob.objective(x)
    d = rng.standard_normal(prob.n_params)
    h = 1e-6
    fd = (prob.value(x + h * d) - prob.value(x - h * d)) / (2 * h)
    assert np.isclose(g @ d, fd, rtol=1e-5)
    _, g_gn, hess = prob.gauss_newton(x)
    np.testing.assert_allclose(g_gn, g, rtol=1e-8, atol=1e-8 * np.abs(g).max())
    assert np.all(np.linalg.eigvalsh(hess) > -1e-6 * np.abs(hess).max())


@pytest.mark.parametrize("loss", ["gaussian", "poisson"])
def test_background_plane_gradients(setup, loss, rng) -> None:
    _, basis, _, model, images = setup
    prob = FocalPlaneProblem(
        model, images, basis=basis, loss=loss, fit_background="plane", fit_blur=True
    )
    assert prob.layout.background_slope.stop - prob.layout.background_slope.start == 4
    x = prob.initial() + 1e-2 * rng.standard_normal(prob.n_params)
    _, g = prob.objective(x)
    h = 1e-6
    for d in (
        rng.standard_normal(prob.n_params),
        np.eye(prob.n_params)[prob.layout.background_slope.start],
    ):
        fd = (prob.value(x + h * d) - prob.value(x - h * d)) / (2 * h)
        assert np.isclose(g @ d, fd, rtol=1e-5)
    _, g_gn, hess = prob.gauss_newton(x)
    np.testing.assert_allclose(g_gn, g, rtol=1e-8, atol=1e-8 * np.abs(g).max())
    assert np.all(np.linalg.eigvalsh(hess) > -1e-6 * np.abs(hess).max())
    with pytest.raises(ValueError, match="fit_background"):
        FocalPlaneProblem(model, images, basis=basis, fit_background="quadratic")


def test_background_ramp_is_recovered(setup, tmp_path) -> None:
    pupil, basis, opd, model, _ = setup
    rng = np.random.default_rng(11)
    mean = 1e6 * np.asarray(model.images(opd))
    yy, xx = np.meshgrid(np.linspace(-1, 1, 40), np.linspace(-1, 1, 40), indexing="ij")
    ramp = 50.0 + 30.0 * xx - 20.0 * yy  # strong stray-light gradient
    images = rng.poisson(mean + ramp[None]).astype(float)
    const = solve(
        FocalPlaneProblem(model, images, basis=basis, loss="poisson", fit_background=True),
        method="lm",
    )
    prob = FocalPlaneProblem(model, images, basis=basis, loss="poisson", fit_background="plane")
    res = solve(prob, method="lm")
    assert res.background_slope is not None and res.background_slope.shape == (2, 2)
    np.testing.assert_allclose(res.background_slope, [[-20.0, 30.0]] * 2, atol=1.5)
    np.testing.assert_allclose(res.background, 50.0, atol=1.5)
    err = sp.wavefront_error(res.opd, opd, pupil, remove="tiptilt")
    assert err < 0.02 * sp.rms(opd, pupil, "tiptilt")
    # A constant background soaks the ramp into the wavefront.
    assert sp.wavefront_error(const.opd, opd, pupil, remove="tiptilt") > 3 * err
    assert solve(prob, method="lm", start=res).n_iter <= 3
    saved = np.load(res.save(tmp_path / "res"))
    np.testing.assert_allclose(saved["background_slope"], res.background_slope)


def test_amplitude_and_zonal_gradients(setup, rng) -> None:
    _, _, _, model, images = setup
    prob = FocalPlaneProblem(
        model, images, basis=None, fit_amplitude=True, smoothness=0.1, loss="poisson"
    )
    x = prob.initial() + 1e-2 * rng.standard_normal(prob.n_params)
    _, g = prob.objective(x)
    d = rng.standard_normal(prob.n_params)
    h = 1e-6
    fd = (prob.value(x + h * d) - prob.value(x - h * d)) / (2 * h)
    assert np.isclose(g @ d, fd, rtol=1e-5)
    with pytest.raises(ValueError, match="Gauss-Newton"):
        prob.gauss_newton(x)


@pytest.mark.parametrize("method", ["lm", "lbfgs"])
def test_modal_retrieval_reaches_the_noise_floor(setup, method) -> None:
    pupil, basis, opd, model, images = setup
    prob = FocalPlaneProblem(model, images, basis=basis, loss="poisson", fit_background=True)
    res = solve(prob, method=method)
    err = sp.wavefront_error(res.opd, opd, pupil, remove="tiptilt")
    assert err < 0.01 * sp.rms(opd, pupil, "tiptilt")
    assert res.converged
    np.testing.assert_allclose(res.flux, 1e6, rtol=0.01)
    np.testing.assert_allclose(res.background, 5.0, atol=0.5)
    assert res.coefficients is not None and res.coefficients.shape == (20,)


def test_registration_tilt_is_recovered(setup) -> None:
    pupil, basis, opd, model, _ = setup
    ramps = Basis.zernike(pupil, 2).mode_maps()
    shifted = model.diversity_opd.copy()
    shifted[1] += 0.03 * LAM * ramps[0]
    truth_model = FocalPlaneModel(pupil, LAM, 40, sampling=2.0, diversity=shifted)
    images = sp.simulate_images(truth_model, opd, photons=1e7, noise=False)
    prob = FocalPlaneProblem(model, images, basis=basis, fit_tilt=True, loss="poisson")
    res = solve(prob, method="lm")
    assert res.tilts is not None
    assert abs(res.tilts[1, 0] - 0.03 * LAM) < 1e-3 * LAM


def test_kl_prior_regularizes_unseen_modes() -> None:
    # A flat wavefront seen with few photons: the maximum-likelihood fit chases
    # noise, while a tight turbulence prior (large r0) shrinks the estimate.
    pupil = Pupil.circular(32, 1.0)
    kl = Basis.kl(pupil, 30, r0=2.0, r0_wavelength=LAM)
    model = FocalPlaneModel(pupil, LAM, 32, sampling=2.0)
    ratios = []
    for seed in range(3):
        images = sp.simulate_images(model, None, photons=1e3, seed=seed)
        free = solve(FocalPlaneProblem(model, images, basis=kl, loss="poisson"), method="lm")
        prior = solve(
            FocalPlaneProblem(model, images, basis=kl, loss="poisson", prior=True), method="lm"
        )
        ratios.append(sp.rms(prior.opd, pupil) / sp.rms(free.opd, pupil))
    assert max(ratios) < 0.8, ratios


def test_start_from_result_and_validation(setup) -> None:
    _, basis, _, model, images = setup
    prob = FocalPlaneProblem(model, images, basis=basis, loss="poisson")
    first = solve(prob, method="lm")
    again = solve(prob, method="lm", start=first)
    assert again.n_iter <= 3
    with pytest.raises(ValueError, match="channel"):
        FocalPlaneProblem(model, images[0], basis=basis)
    with pytest.raises(ValueError, match="non-finite"):
        bad = images.copy()
        bad[0, 0, 0] = np.nan
        FocalPlaneProblem(model, bad, basis=basis)
    with pytest.raises(ValueError, match="prior"):
        FocalPlaneProblem(model, images, basis=basis, prior=True)
    with pytest.raises(ValueError, match="unknown method"):
        solve(prob, method="newton")


def test_masked_pixels_are_ignored(setup) -> None:
    pupil, basis, opd, model, images = setup
    corrupted = images.copy()
    corrupted[:, 18:22, 18:22] = 0.0  # a dead patch on the core
    weights = np.ones_like(images)
    weights[:, 18:22, 18:22] = 0.0
    res = solve(
        FocalPlaneProblem(
            model, corrupted, basis=basis, loss="poisson", weights=weights, fit_background=True
        )
    )
    assert sp.wavefront_error(res.opd, opd, pupil, remove="tiptilt") < 0.03 * sp.rms(
        opd, pupil, "tiptilt"
    )


@pytest.mark.parametrize("device", devices())
def test_device_parity(setup, device) -> None:
    pupil, basis, opd, _, images = setup
    div = zernike_diversity(pupil, 4, [0.0, 0.25 * LAM])
    model = FocalPlaneModel(
        pupil, LAM, 40, sampling=2.0, diversity=div, device=device, precision="single"
    )
    prob = FocalPlaneProblem(model, images, basis=basis, loss="poisson", fit_background=True)
    res = solve(prob, method="lm")
    assert res.device == device
    # Single precision reaches the same noise floor as double (~0.35 nm here).
    assert sp.wavefront_error(sp.to_numpy(res.opd), opd, pupil, remove="tiptilt") < 0.6e-9


# ------------------------------------------------- blur and diversity scale
@pytest.mark.parametrize("fit_blur", [True, "xy"])
@pytest.mark.parametrize("loss", ["gaussian", "poisson"])
def test_blur_and_diversity_scale_gradients(setup, fit_blur, loss, rng) -> None:
    _, basis, _, model, images = setup
    prob = FocalPlaneProblem(
        model,
        images,
        basis=basis,
        loss=loss,
        fit_background=True,
        fit_tilt=True,
        fit_blur=fit_blur,
        fit_diversity_scale=True,
    )
    assert prob.layout.log_blur.stop - prob.layout.log_blur.start == (2 if fit_blur == "xy" else 1)
    x = prob.initial() + 1e-2 * rng.standard_normal(prob.n_params)
    _, g = prob.objective(x)
    h = 1e-6
    for d in (rng.standard_normal(prob.n_params), np.eye(prob.n_params)[-1]):
        fd = (prob.value(x + h * d) - prob.value(x - h * d)) / (2 * h)
        assert np.isclose(g @ d, fd, rtol=1e-5)
    _, g_gn, hess = prob.gauss_newton(x)
    np.testing.assert_allclose(g_gn, g, rtol=1e-8, atol=1e-8 * np.abs(g).max())
    assert np.all(np.linalg.eigvalsh(hess) > -1e-6 * np.abs(hess).max())


def test_blur_and_diversity_scale_are_recovered(setup, tmp_path) -> None:
    pupil, basis, opd, model, _ = setup
    truth = FocalPlaneModel(
        pupil, LAM, 40, sampling=2.0, diversity=1.15 * model.diversity_opd, blur=0.45
    )
    images = sp.simulate_images(truth, opd, photons=1e6, background=5.0, seed=6)
    naive = solve(FocalPlaneProblem(model, images, basis=basis, loss="poisson"), method="lm")
    prob = FocalPlaneProblem(
        model,
        images,
        basis=basis,
        loss="poisson",
        fit_background=True,
        fit_blur=True,
        fit_diversity_scale=True,
    )
    res = solve(prob, method="lm")
    assert res.blur is not None and res.diversity_scale is not None
    np.testing.assert_allclose(res.blur, 0.45, rtol=0.03)
    assert abs(res.diversity_scale - 1.15) < 0.01
    err = sp.wavefront_error(res.opd, opd, pupil, remove="tiptilt")
    assert err < 0.02 * sp.rms(opd, pupil, "tiptilt")
    # Ignoring the blur and the scale error biases the wavefront badly.
    assert sp.wavefront_error(naive.opd, opd, pupil, remove="tiptilt") > 20 * err
    # A warm start keeps the fitted nuisance values.
    again = solve(prob, method="lm", start=res)
    assert again.n_iter <= 3
    saved = np.load(res.save(tmp_path / "res"))
    np.testing.assert_allclose(saved["blur"], res.blur)
    assert float(saved["diversity_scale"]) == res.diversity_scale


def test_runaway_blur_step_stays_finite(setup) -> None:
    # A trial step far out in log sigma used to overflow exp() and raise inside
    # the objective; it must give a finite cost the optimizer can reject.
    _, basis, _, model, images = setup
    prob = FocalPlaneProblem(model, images, basis=basis, loss="poisson", fit_blur=True)
    x = prob.initial()
    good = prob.value(x)
    x[prob.layout.log_blur] = 1e3
    value, grad = prob.objective(x)
    assert np.isfinite(value) and value > good
    assert np.all(np.isfinite(grad))
    sigma = prob._blur_sigma(x)
    assert sigma is not None and max(sigma) <= max(model.image_shape) * (1 + 1e-12)


def test_fixed_blur_and_validation(setup) -> None:
    pupil, basis, opd, model, _ = setup
    truth = FocalPlaneModel(pupil, LAM, 40, sampling=2.0, diversity=model.diversity_opd, blur=0.3)
    images = sp.simulate_images(truth, opd, photons=1e6, seed=7)
    fixed = solve(FocalPlaneProblem(model, images, basis=basis, loss="poisson", blur=0.3))
    np.testing.assert_allclose(fixed.blur, [0.3, 0.3])
    assert sp.wavefront_error(fixed.opd, opd, pupil, remove="tiptilt") < 0.02 * sp.rms(
        opd, pupil, "tiptilt"
    )
    plain = solve(FocalPlaneProblem(model, images, basis=basis, loss="poisson"), max_iter=2)
    np.testing.assert_allclose(plain.blur, [0.0, 0.0])
    assert plain.diversity_scale is None
    with pytest.raises(ValueError, match="fit_blur"):
        FocalPlaneProblem(model, images, basis=basis, fit_blur="diagonal")
    with pytest.raises(ValueError, match="non-zero diversity"):
        FocalPlaneProblem(
            FocalPlaneModel(pupil, LAM, 40, sampling=2.0),
            images[0],
            basis=basis,
            fit_diversity_scale=True,
        )
    with pytest.raises(ValueError, match="positive starting width"):
        FocalPlaneProblem(model, images, basis=basis, fit_blur="xy", blur=(0.0, 0.3))
