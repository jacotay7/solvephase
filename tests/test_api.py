from __future__ import annotations

import numpy as np
import pytest

import solvephase as sp
from solvephase import Pupil

LAM = 1.6e-6


@pytest.fixture(scope="module")
def data():
    pupil = Pupil.circular(48, 8.0, obscuration=0.14)
    opd = sp.random_aberration(pupil, 0.1 * LAM, n_modes=20, seed=7)
    model = sp.FocalPlaneModel(
        pupil, LAM, 48, sampling=2.0, diversity=sp.zernike_diversity(pupil, 4, [0.0, 0.3 * LAM])
    )
    images = sp.simulate_images(model, opd, photons=2e6, background=10, read_noise=3.0, seed=8)
    return pupil, opd, images


def test_retrieve_auto(data) -> None:
    pupil, opd, images = data
    res = sp.retrieve(images, pupil, LAM, sampling=2.0, diversity=[0.0, 0.3 * LAM], read_noise=3.0)
    assert sp.wavefront_error(res.opd, opd, pupil, remove="tiptilt") < 0.02 * sp.rms(
        opd, pupil, "tiptilt"
    )
    stages = [s["stage"] for s in res.extra["stages"]]
    assert stages == ["lm-amplitude", "gs->lm-amplitude", "polish"]
    assert "solvephase lm" in res.summary()


def test_retrieve_methods_and_zonal_refinement(data) -> None:
    pupil, _, images = data
    kw = {"sampling": 2.0, "diversity": [0.0, 0.3 * LAM]}
    lm = sp.retrieve(images, pupil, LAM, method="lm", **kw)
    assert lm.method == "lm"
    gs = sp.retrieve(images, pupil, LAM, method="gs", basis="zonal", **kw)
    assert gs.method == "misell"
    refined = sp.retrieve(images, pupil, LAM, zonal_refinement=True, max_iter=60, **kw)
    assert refined.basis is None and refined.extra["stages"][-1]["stage"] == "zonal"


def test_retrieve_input_validation(data) -> None:
    pupil, _, images = data
    with pytest.raises(ValueError, match="diversity"):
        sp.retrieve(images, pupil, LAM, sampling=2.0)
    with pytest.raises(ValueError, match="2 images"):
        sp.retrieve(images, pupil, LAM, sampling=2.0, diversity=[0.0, 1e-7, 2e-7])
    with pytest.raises(ValueError, match="unknown method"):
        sp.retrieve(images, pupil, LAM, sampling=2.0, diversity=[0.0, 1e-7], method="magic")
    with pytest.raises(TypeError, match="pupil"):
        sp.retrieve(images, "vlt", LAM, sampling=2.0, diversity=[0.0, 1e-7])


def test_retrieve_with_int_pupil_and_single_image() -> None:
    pupil = Pupil.circular(32, 1.0)
    model = sp.FocalPlaneModel(pupil, 1e-6, 32, sampling=2.0)
    images = sp.simulate_images(model, None, photons=1e5, seed=1)
    res = sp.retrieve(images[0], 32, 1e-6, sampling=2.0, basis=6, method="lm")
    assert res.rms() < 5e-9


def test_result_roundtrip(tmp_path, data) -> None:
    pupil, _, images = data
    res = sp.retrieve(images, pupil, LAM, sampling=2.0, diversity=[0.0, 0.3 * LAM], method="lm")
    path = res.save(tmp_path / "result")
    loaded = np.load(path)
    np.testing.assert_allclose(loaded["opd"], res.opd)
    assert str(loaded["method"]) == "lm"
    assert res.residuals(images).shape == images.shape
    host = res.to_numpy()
    assert isinstance(host.opd, np.ndarray)


def test_result_plot(tmp_path, data) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    pupil, _, images = data
    res = sp.retrieve(images, pupil, LAM, sampling=2.0, diversity=[0.0, 0.3 * LAM], method="lm")
    fig = res.plot(images, path=tmp_path / "r.png")
    assert (tmp_path / "r.png").exists() and len(fig.axes) >= 4
