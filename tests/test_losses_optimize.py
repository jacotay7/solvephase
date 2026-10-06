from __future__ import annotations

import numpy as np
import pytest

from solvephase.backend import get_backend
from solvephase.losses import AmplitudeLoss, GaussianLoss, PoissonLoss, make_loss
from solvephase.optimize import adam, lbfgs, levenberg_marquardt

BE = get_backend("cpu")


@pytest.mark.parametrize("loss", [GaussianLoss(), PoissonLoss(read_noise=1.5), AmplitudeLoss()])
def test_loss_gradients_match_finite_differences(loss, rng) -> None:
    data = rng.poisson(50.0, size=(2, 8, 8)).astype(float)
    model = data + rng.normal(0, 3, size=data.shape) + 5
    w = rng.uniform(0.5, 1.5, size=data.shape)
    _, grad, curv = loss.evaluate(BE, model, data, w)
    d = rng.standard_normal(model.shape)
    h = 1e-6
    vp = float(loss.evaluate(BE, model + h * d, data, w)[0])
    vm = float(loss.evaluate(BE, model - h * d, data, w)[0])
    assert np.isclose(np.sum(grad * d), (vp - vm) / (2 * h), rtol=1e-6)
    assert np.all(curv >= 0)


def test_poisson_deviance_is_zero_at_a_perfect_fit_and_positive_elsewhere(rng) -> None:
    data = rng.poisson(20.0, size=(4, 4)).astype(float)
    loss = PoissonLoss()
    assert abs(float(loss.evaluate(BE, data, data, np.ones_like(data))[0])) < 1e-6
    assert float(loss.evaluate(BE, data + 1, data, np.ones_like(data))[0]) > 0


def test_make_loss_names() -> None:
    assert isinstance(make_loss("ls"), GaussianLoss)
    assert isinstance(make_loss("poisson", read_noise=2.0), PoissonLoss)
    with pytest.raises(ValueError, match="unknown loss"):
        make_loss("huber")


def _rosen(x):
    f = np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
    g = np.zeros_like(x)
    g[:-1] += -400 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2 * (1 - x[:-1])
    g[1:] += 200 * (x[1:] - x[:-1] ** 2)
    return float(f), g


def test_lbfgs_solves_rosenbrock() -> None:
    res = lbfgs(_rosen, np.full(20, -1.2), BE, max_iter=500, ftol=1e-15)
    assert res.converged
    np.testing.assert_allclose(res.x, 1.0, atol=1e-6)
    assert res.history[0] > res.history[-1] and len(res.history) == res.n_iter + 1


def test_lbfgs_callback_can_stop() -> None:
    res = lbfgs(_rosen, np.full(4, -1.2), BE, callback=lambda it, x, f: it >= 3)
    assert res.n_iter == 3 and res.message == "stopped by callback"


def test_levenberg_marquardt_fits_an_exponential() -> None:
    t = np.linspace(0, 1, 50)
    y = 2 * np.exp(-3 * t)

    def res(p):
        return p[0] * np.exp(-p[1] * t) - y

    def jac(p):
        return np.stack([np.exp(-p[1] * t), -p[0] * t * np.exp(-p[1] * t)], 1)

    def system(p):
        r = res(p)
        return 0.5 * float(r @ r), jac(p).T @ r, jac(p).T @ jac(p)

    out = levenberg_marquardt(
        system, lambda p: 0.5 * float(res(p) @ res(p)), np.array([1.0, 1.0]), BE
    )
    np.testing.assert_allclose(out.x, [2.0, 3.0], atol=1e-8)


def test_adam_reduces_a_quadratic() -> None:
    def quad(x):
        return float(np.sum((x - 3) ** 2)), 2 * (x - 3)

    out = adam(quad, np.zeros(5), BE, max_iter=600, learning_rate=0.1)
    np.testing.assert_allclose(out.x, 3.0, atol=1e-2)
