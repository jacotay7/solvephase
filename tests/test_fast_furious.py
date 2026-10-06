"""Fast & Furious: weak-phase odd/even solution, sign from diversity, closed loop, GPU."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import devices
from solvephase import Basis, FocalPlaneModel, Pupil, rms
from solvephase.algorithms.fast_furious import FastAndFurious, simulate_closed_loop
from solvephase.backend import to_numpy

WL = 1.6e-6


def _aberration(pupil: Pupil, rms_waves: float, n_modes: int = 10, seed: int = 2) -> np.ndarray:
    basis = Basis.zernike(pupil, n_modes)
    opd = basis.synthesize(np.random.default_rng(seed).normal(size=n_modes))
    return opd * rms_waves * WL / rms(opd, pupil)


def _odd(opd: np.ndarray) -> np.ndarray:
    return 0.5 * (opd - opd[::-1, ::-1])


def _match(estimate: object, truth: np.ndarray, pupil: Pupil) -> tuple[float, float]:
    """Relative RMS error and least-squares gain of ``estimate`` against ``truth``."""
    est = np.asarray(to_numpy(estimate), dtype=np.float64)
    w = pupil.amplitude**2
    gain = float(np.sum(w * est * truth) / np.sum(w * truth * truth))
    return rms(est - truth, pupil) / rms(truth, pupil), gain


@pytest.fixture(scope="module")
def pupil64() -> Pupil:
    return Pupil.circular(64)


@pytest.fixture(scope="module")
def pupil32() -> Pupil:
    return Pupil.circular(32, obscuration=0.1)


def test_first_step_odd_part_matches_truth(pupil64: Pupil) -> None:
    truth = _aberration(pupil64, 0.01)
    model = FocalPlaneModel(pupil64, WL, 64, sampling=2)
    ff = FastAndFurious(pupil64, WL, 64, sampling=2)
    assert not ff.has_previous
    estimate = ff.step(model.images(truth)[0])
    assert ff.has_previous and ff.n_steps == 1
    err, gain = _match(ff.last_odd_opd, _odd(truth), pupil64)
    assert err < 0.25
    assert 0.8 < gain < 1.05
    np.testing.assert_allclose(
        to_numpy(ff.last_odd_opd) + to_numpy(ff.last_even_opd), to_numpy(estimate)
    )
    # Odd and even parts have the right symmetry and vanish outside the pupil.
    odd = to_numpy(ff.last_odd_opd)
    np.testing.assert_allclose(odd, -odd[::-1, ::-1], atol=1e-12 * WL)
    assert np.all(to_numpy(estimate)[~pupil64.mask] == 0)


def test_tilt_sign(pupil64: Pupil) -> None:
    """A ramp a * x is purely odd and is recovered with its sign (analytic check)."""
    _, x = pupil64.coordinates()
    ramp = np.where(pupil64.mask, 0.02 * WL * np.asarray(x) / 0.5, 0.0)
    model = FocalPlaneModel(pupil64, WL, 64, sampling=2)
    ff = FastAndFurious(pupil64, WL, 64, sampling=2, first_even=False)
    estimate = ff.step(model.images(ramp)[0])
    err, gain = _match(estimate, ramp, pupil64)
    assert 0.85 < gain < 1.05 and err < 0.2
    # first_even=False: odd part only (up to FFT round-off)
    assert np.abs(to_numpy(ff.last_even_opd)).max() < 1e-10 * np.abs(ramp).max()


def test_even_sign_from_previous_image(pupil64: Pupil) -> None:
    """With a known even DM change between frames the even part gets its sign."""
    model = FocalPlaneModel(pupil64, WL, 64, sampling=2)
    even_modes = Basis.zernike(pupil64, 3, start=4)  # defocus and astigmatism
    truth = even_modes.synthesize(np.array([1.0, -0.7, 0.5]))
    truth *= 0.01 * WL / rms(truth, pupil64)
    ff = FastAndFurious(pupil64, WL, 64, sampling=2, first_even=False)
    ff.step(model.images(truth)[0])
    probe = even_modes.synthesize(np.array([0.0, 0.0, 1.0])) * 0.01 * WL  # known DM change
    current = truth + probe
    estimate = ff.step(model.images(current)[0], probe)
    err, gain = _match(ff.last_even_opd, current, pupil64)
    assert gain > 0.7 and err < 0.4
    # Flipping the even aberration flips the estimate: the sign is resolved.
    ff.reset()
    ff.step(model.images(-truth)[0])
    flipped = ff.step(model.images(-truth + probe)[0], probe)
    _, gain_flipped = _match(flipped, -truth + probe, pupil64)
    assert gain_flipped > 0.7
    assert rms(to_numpy(estimate) - to_numpy(flipped), pupil64) > rms(truth, pupil64)


def test_closed_loop_noise_free(pupil32: Pupil) -> None:
    truth = _aberration(pupil32, 0.1)
    ff = FastAndFurious(pupil32, WL, 32, sampling=2)
    loop = simulate_closed_loop(ff, truth, 20, gain=0.5)
    assert loop.residual_rms[0] == pytest.approx(rms(truth, pupil32), rel=1e-9)
    assert loop.residual_rms[-1] < 0.2 * loop.residual_rms[0]
    assert loop.residual_rms[10] < 0.3 * loop.residual_rms[0]
    assert loop.strehl[-1] > 0.98
    assert loop.n_iter == 20 and len(loop.step_times) == 20
    final = rms(to_numpy(loop.residual_opd), pupil32)
    assert final == pytest.approx(loop.residual_rms[-1], rel=1e-6)


def test_closed_loop_photon_noise(pupil32: Pupil) -> None:
    truth = _aberration(pupil32, 0.1, seed=5)
    ff = FastAndFurious(pupil32, WL, 32, sampling=2)
    loop = simulate_closed_loop(ff, truth, 20, gain=0.5, photons=1e6, read_noise=1.0, seed=3)
    assert loop.residual_rms[-1] < 0.2 * loop.residual_rms[0]
    # Same seed, same noise, same trajectory.
    again = simulate_closed_loop(ff, truth, 20, gain=0.5, photons=1e6, read_noise=1.0, seed=3)
    np.testing.assert_allclose(again.residual_rms, loop.residual_rms)


def test_closed_loop_modal_dm_and_tol(pupil32: Pupil) -> None:
    truth = _aberration(pupil32, 0.1, n_modes=10)
    ff = FastAndFurious(pupil32, WL, 32, sampling=2)
    dm = Basis.zernike(pupil32, 20)
    loop = simulate_closed_loop(ff, truth, 30, gain=0.5, leak=0.99, basis=dm, tol=0.03 * WL)
    assert loop.converged and loop.n_iter < 30
    assert loop.residual_rms[-1] < 0.03 * WL
    # The DM only made shapes in its basis.
    dm_opd = to_numpy(loop.dm_opd)
    np.testing.assert_allclose(
        dm.project(dm_opd)[pupil32.mask], dm_opd[pupil32.mask], atol=1e-6 * WL
    )


def test_missing_dm_change_leaves_even_part_zero(pupil32: Pupil) -> None:
    truth = _aberration(pupil32, 0.02)
    model = FocalPlaneModel(pupil32, WL, 32, sampling=2)
    ff = FastAndFurious(pupil32, WL, 32, sampling=2)
    ff.step(model.images(truth)[0])
    ff.step(model.images(truth)[0])  # no change given: signs unresolvable
    odd = np.abs(to_numpy(ff.last_odd_opd)).max()
    assert np.abs(to_numpy(ff.last_even_opd)).max() < 1e-10 * odd
    ff.reset()
    assert not ff.has_previous and ff.n_steps == 0


def test_errors(pupil32: Pupil) -> None:
    with pytest.raises(ValueError, match="integer"):
        FastAndFurious(pupil32, WL, 32, sampling=2.3)
    with pytest.raises(ValueError, match="exactly one"):
        FastAndFurious(pupil32, WL, 32)
    amp = pupil32.amplitude.copy()
    amp[:4, :] = 0.0
    with pytest.raises(ValueError, match="centro-symmetric"):
        FastAndFurious(Pupil.from_array(amp, diameter=1.0), WL, 32, sampling=2)
    ff = FastAndFurious(pupil32, WL, 32, sampling=2)
    with pytest.raises(ValueError, match="image shape"):
        ff.step(np.ones((16, 16)))
    ff.step(np.ones((32, 32)))
    with pytest.raises(ValueError, match="dm_change_opd"):
        ff.step(np.ones((32, 32)), np.zeros((8, 8)))
    with pytest.raises(ValueError, match="gain"):
        simulate_closed_loop(ff, np.zeros(pupil32.shape), 5, gain=0.0)


@pytest.mark.parametrize("device", devices())
def test_device_parity(pupil32: Pupil, device: str) -> None:
    truth = _aberration(pupil32, 0.1)
    model = FocalPlaneModel(pupil32, WL, 32, sampling=2)
    image = to_numpy(model.images(truth)[0])
    ref = FastAndFurious(pupil32, WL, 32, sampling=2)
    ff = FastAndFurious(pupil32, WL, 32, sampling=2, device=device)
    e_ref = to_numpy(ref.step(image))
    e_dev = ff.step(image)
    assert type(e_dev).__module__.split(".")[0] == ("cupy" if device == "gpu" else "numpy")
    np.testing.assert_allclose(to_numpy(e_dev), e_ref, atol=1e-4 * rms(e_ref, pupil32))
    loop_ref = simulate_closed_loop(ref, truth, 15)
    loop_dev = simulate_closed_loop(ff, truth, 15)
    assert loop_dev.device == device
    np.testing.assert_allclose(
        loop_dev.residual_rms, loop_ref.residual_rms, atol=0.02 * loop_ref.residual_rms[0]
    )
