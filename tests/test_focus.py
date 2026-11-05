"""Focus sensing with a focus mechanism (solvephase.algorithms.focus)."""

from __future__ import annotations

import copy

import numpy as np
import pytest

import solvephase as sp
from solvephase import Basis, FocalPlaneModel, FocusSensor, Pupil

LAM = 1e-6
NM = 1e-9  # the test mechanism moves 1 nm RMS of focus per unit


def _pupil(vanes: int) -> Pupil:
    return Pupil.circular(48, 1.0, obscuration=0.3, spiders=vanes, spider_width=0.06)


@pytest.fixture(scope="module")
def three_vane() -> Pupil:
    # Three vanes: not centro-symmetric, so one image fixes the sign of defocus.
    return _pupil(3)


@pytest.fixture(scope="module")
def four_vane() -> Pupil:
    # Four vanes: centro-symmetric, so the twin is exactly as likely.
    return _pupil(4)


def _defocus(pupil: Pupil) -> np.ndarray:
    return Basis.zernike(pupil, 1, start=4).mode_maps()[0]


def _truth(pupil: Pupil, focus_nm: float, seed: int = 2) -> np.ndarray:
    rest = sp.random_aberration(pupil, 0.02 * LAM, n_modes=6, start=5, seed=seed)
    return focus_nm * NM * _defocus(pupil) + rest


def _images(pupil, opd, positions, *, scale=1.0, blur=0.0, shape=32, photons=1e6, seed=3):
    resp = _defocus(pupil) * NM
    model = FocalPlaneModel(
        pupil,
        LAM,
        shape,
        sampling=2.0,
        diversity=[resp * p * scale for p in positions],
        blur=blur,
    )
    return sp.simulate_images(model, opd, photons=photons, read_noise=1.0, seed=seed)


def _sensor(pupil, **kwargs) -> FocusSensor:
    kwargs.setdefault("fit_blur", False)
    shape = kwargs.pop("shape", 32)
    return FocusSensor(
        pupil, LAM, shape, _defocus(pupil) * NM, sampling=2.0, basis=10, read_noise=1.0, **kwargs
    )


def test_best_focus_and_twin(three_vane) -> None:
    # On a plain annulus defocus is orthogonal to coma and trefoil (Noll 7-10).
    pupil = Pupil.circular(48, 1.0, obscuration=0.3)
    dfoc = _defocus(pupil)
    rest = sp.random_aberration(pupil, 30e-9, n_modes=4, start=7, seed=1)
    # A pure-focus response of 2 nm RMS per unit; 50 nm of defocus needs -25 units.
    offset = sp.best_focus(50e-9 * dfoc + rest, 2e-9 * dfoc, pupil)
    assert np.isclose(offset, -25.0, atol=0.3)
    # Tip, tilt and piston are ignored; a response with only those cannot focus.
    tilt = Basis.zernike(pupil, 2, start=2).mode_maps()
    assert np.isclose(sp.best_focus(50e-9 * dfoc + 1e-7 * tilt[0], 2e-9 * dfoc, pupil), -25.0)
    with pytest.raises(ValueError, match="excluded modes"):
        sp.best_focus(dfoc, tilt[0], pupil)
    # The twin is an involution that flips even modes and keeps odd ones.
    opd = _truth(three_vane, 40.0)
    np.testing.assert_allclose(sp.twin(sp.twin(opd)), opd)
    basis = Basis.zernike(Pupil.circular(48, 1.0, obscuration=0.3), 10, start=2)
    coef = basis.fit(np.where(basis.pupil.mask, opd, 0.0))
    coef_twin = basis.fit(np.where(basis.pupil.mask, sp.twin(opd), 0.0))
    even = np.array([j in (4, 5, 6, 11) for j in range(2, 12)])
    np.testing.assert_allclose(coef_twin[even], -coef[even], atol=1e-12)
    np.testing.assert_allclose(coef_twin[~even], coef[~even], atol=1e-12)


@pytest.mark.parametrize("focus_nm", [60.0, -60.0, 150.0])
def test_one_image_fixes_the_sign_on_an_asymmetric_pupil(three_vane, focus_nm) -> None:
    pupil = three_vane
    opd = _truth(pupil, focus_nm)
    sensor = _sensor(pupil)
    est = sensor.estimate(_images(pupil, opd, [0.0])[0], [0.0])
    truth = sensor.best_position(opd, 0.0)
    assert np.isclose(truth, -focus_nm, atol=1.0)
    assert abs(est.position - truth) < 5 * est.sigma + 0.5
    assert est.probability > 0.999
    # The twin start found the mirror solution, and it is far less likely.
    assert est.alternatives and abs(est.alternatives[0][0] + est.position) < 0.1 * abs(focus_nm)
    assert 0.8 < est.chi2 < 1.2
    assert est.n_images == 1 and est.reference == est.current == 0.0
    assert np.isclose(est.move, est.position)


def test_symmetric_pupil_reports_the_twin_ambiguity(four_vane) -> None:
    pupil = four_vane
    opd = _truth(pupil, 60.0)
    sensor = _sensor(pupil)
    est = sensor.estimate(_images(pupil, opd, [0.0])[0], [0.0])
    truth = sensor.best_position(opd, 0.0)
    # Both signs fit equally: the estimate is +/-60 and the other is the alternative.
    assert np.isclose(abs(est.position), abs(truth), atol=1.0)
    assert 0.4 < est.probability < 0.6
    candidates = [est.position] + [p for p, _ in est.alternatives]
    assert min(abs(c - truth) for c in candidates) < 1.0
    # A second image at a known position removes the ambiguity.
    images = _images(pupil, opd, [0.0, -80.0])
    two = sensor.estimate(images, [0.0, -80.0])
    assert abs(two.position - truth) < 5 * two.sigma + 0.5
    assert two.probability > 0.999
    assert two.current == -80.0 and np.isclose(two.move, two.position + 80.0)


def test_unmodelled_blur_biases_focus_and_fitting_it_removes_the_bias(three_vane) -> None:
    pupil = three_vane
    opd = _truth(pupil, 40.0)
    image = _images(pupil, opd, [0.0], blur=0.4)[0]
    truth = _sensor(pupil).best_position(opd, 0.0)
    naive = _sensor(pupil).estimate(image, [0.0])
    fitted = _sensor(pupil, fit_blur=True).estimate(image, [0.0])
    assert abs(naive.position - truth) > 10.0  # blur reads as extra defocus
    assert abs(fitted.position - truth) < 1.5
    np.testing.assert_allclose(fitted.result.blur, 0.4, rtol=0.05)


def test_warm_start_and_rereference(three_vane) -> None:
    pupil = three_vane
    opd = _truth(pupil, 50.0)
    sensor = _sensor(pupil)
    first = sensor.estimate(_images(pupil, opd, [0.0])[0], [0.0])
    # Move to the estimate, take a second image; the first estimate seeds the solve.
    p1 = first.position
    second = sensor.estimate(_images(pupil, opd, [0.0, p1]), [0.0, p1], start=first)
    truth = sensor.best_position(opd, 0.0)
    assert second.reference == p1
    assert abs(second.position - truth) < 5 * second.sigma + 0.5
    assert abs(second.move) < 2.0
    with pytest.raises(ValueError, match="position"):
        sensor.estimate(_images(pupil, opd, [0.0, 10.0]), [0.0])


def test_calibrate_recovers_the_response_scale(three_vane) -> None:
    pupil = three_vane
    opd = _truth(pupil, 30.0)
    positions = [-150.0, -75.0, 0.0, 75.0, 150.0]
    images = _images(pupil, opd, positions, scale=1.1)
    sensor = _sensor(pupil)
    cal = sensor.calibrate(images, positions)
    assert abs(cal.scale - 1.1) < 4 * cal.sigma + 1e-3
    assert cal.sigma < 0.01
    np.testing.assert_array_equal(cal.positions, positions)
    calibrated = sensor.with_response_scale(cal.scale)
    np.testing.assert_allclose(calibrated.response, sensor.response * cal.scale)
    assert np.isclose(calibrated.focus_rms_per_unit, sensor.focus_rms_per_unit * cal.scale)
    # The calibrated sensor puts the best position where the true response does.
    true_best = sensor.best_position(opd, 0.0) / 1.1
    est = calibrated.estimate(images[2], [0.0])
    assert abs(est.position - true_best) < 1.0
    with pytest.raises(ValueError, match="distinct positions"):
        sensor.calibrate(images[:1], [0.0])


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # the flat start diverges
def test_capture_scan_extends_the_capture_range(three_vane) -> None:
    pupil = three_vane
    opd = _truth(pupil, 900.0)  # 0.9 wave RMS of defocus
    image = _images(pupil, opd, [0.0], shape=48, photons=1e7, seed=5)[0]
    truth = _sensor(pupil).best_position(opd, 0.0)
    lost = _sensor(pupil, shape=48, capture=0.0).estimate(image, [0.0])
    found = _sensor(pupil, shape=48, capture=1.2e-6).estimate(image, [0.0])
    assert lost.chi2 > 10.0  # a flat start fails, and the fit statistic says so
    assert abs(found.position - truth) < 1.0
    assert found.chi2 < 1.3
    with pytest.raises(ValueError, match="capture"):
        _sensor(pupil, capture=-1.0)


def test_combine_estimates_is_the_joint_least_squares_optimum(three_vane) -> None:
    pupil = three_vane
    w = pupil.amplitude**2
    stars = []
    for focus_nm, scale, ref, seed in ((40.0, 1.0, 0.0, 2), (-20.0, 1.3, 15.0, 4)):
        opd = _truth(pupil, focus_nm, seed=seed)
        sensor = _sensor(pupil).with_response_scale(scale)
        stars.append((sensor, sensor.estimate(_images(pupil, opd, [0.0])[0], [0.0])))
        stars[-1][1].reference = ref  # pretend the second star was measured elsewhere
    weights = [1.0, 3.0]
    position, sigma = sp.combine_estimates([e for _, e in stars], weights)

    def cost(p: float) -> float:
        total = 0.0
        for (s, e), wt in zip(stars, weights, strict=True):
            res = sp.remove_modes(
                np.asarray(e.result.opd) + s.response * (p - e.reference), pupil, "tiptilt"
            )
            total += wt * np.sum(w * res**2)
        return total

    grid = np.linspace(position - 5, position + 5, 2001)
    assert abs(grid[np.argmin([cost(p) for p in grid])] - position) < 0.01
    single, single_sigma = sp.combine_estimates([stars[0][1]])
    assert np.isclose(single, stars[0][1].position) and np.isclose(single_sigma, stars[0][1].sigma)
    assert sigma < max(e.sigma for _, e in stars)
    # A star that converged far from the others is dropped with ``reject``.
    copies = []
    for shift in (0.0, 2.0, -3.0, 4.0, -1.0, 600.0):
        e = copy.copy(stars[0][1])
        e.reference, e.position = e.reference + shift, e.position + shift
        copies.append(e)
    mask = sp.outlier_mask(copies, reject=5.0, floor=30.0)
    assert mask.tolist() == [False] * 5 + [True]
    clean, _ = sp.combine_estimates(copies[:5])
    assert np.isclose(sp.combine_estimates(copies, reject=5.0, floor=30.0)[0], clean)
    assert sp.combine_estimates(copies)[0] > clean + 50
    assert not sp.outlier_mask(copies[:2]).any()  # too few stars to judge
    assert sp.outlier_mask(copies[3:], floor=30.0).tolist() == [False, False, True]  # 1 of 3
    with pytest.raises(ValueError, match="no estimates"):
        sp.combine_estimates([])
    with pytest.raises(ValueError, match="weights"):
        sp.combine_estimates([stars[0][1]], [-1.0])


def test_should_move_and_summary(three_vane) -> None:
    pupil = three_vane
    est = _sensor(pupil).estimate(_images(pupil, _truth(pupil, 30.0), [0.0])[0], [0.0])
    assert est.should_move(deadband=10.0)  # 30 units away, well measured
    assert not est.should_move(deadband=40.0)
    est.sigma = 20.0  # a noisy estimate: do not chase it
    assert not est.should_move(deadband=1.0, k_sigma=2.0)
    assert est.should_move(deadband=1.0, k_sigma=1.0)
    text = est.summary()
    assert "best position" in text and "probability" in text and "blur" in text


def test_sensor_validation(three_vane) -> None:
    pupil = three_vane
    with pytest.raises(ValueError, match="does not match"):
        FocusSensor(pupil, LAM, 32, np.zeros((8, 8)), sampling=2.0)
    with pytest.raises(ValueError, match="non-finite"):
        FocusSensor(pupil, LAM, 32, np.full(pupil.shape, np.nan), sampling=2.0)
    tilt = Basis.zernike(pupil, 1, start=2).mode_maps()[0]
    with pytest.raises(ValueError, match="excluded modes"):
        FocusSensor(pupil, LAM, 32, tilt, sampling=2.0)


@pytest.mark.slow
def test_fisher_sigma_matches_the_scatter(three_vane) -> None:
    pupil = three_vane
    opd = _truth(pupil, 50.0)
    sensor = _sensor(pupil)
    truth = sensor.best_position(opd, 0.0)
    estimates = [
        sensor.estimate(_images(pupil, opd, [0.0], photons=1e5, seed=s)[0], [0.0], twin_check=False)
        for s in range(30)
    ]
    positions = np.array([e.position for e in estimates])
    sigma = np.mean([e.sigma for e in estimates])
    assert 0.7 < np.std(positions) / sigma < 1.4
    assert abs(np.mean(positions) - truth) < 3 * sigma / np.sqrt(len(positions)) + 0.3
