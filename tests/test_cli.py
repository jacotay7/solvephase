from __future__ import annotations

import json

import numpy as np
import pytest

import solvephase as sp
from solvephase.cli import main


def test_info_prints_versions(capsys) -> None:
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "solvephase" in out and "numpy" in out and "gpu" in out


def test_retrieve_command_round_trip(tmp_path, capsys) -> None:
    pupil = sp.Pupil.circular(32, 1.0)
    model = sp.FocalPlaneModel(
        pupil, 1e-6, 32, sampling=2.0, diversity=sp.zernike_diversity(pupil, 4, [0.0, 0.3e-6])
    )
    truth = sp.random_aberration(pupil, 50e-9, n_modes=10, seed=1)
    images = sp.simulate_images(model, truth, photons=1e6, seed=2)
    np.save(tmp_path / "images.npy", images)
    code = main(
        [
            "retrieve",
            str(tmp_path / "images.npy"),
            "--wavelength",
            "1e-6",
            "--sampling",
            "2",
            "--defocus",
            "0",
            "0.3e-6",
            "--pupil-pixels",
            "32",
            "--modes",
            "12",
            "--method",
            "lm",
            "--output",
            str(tmp_path / "result.npz"),
            "--json",
            str(tmp_path / "summary.json"),
        ]
    )
    assert code == 0
    assert "saved" in capsys.readouterr().out
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["method"] == "lm" and len(summary["coefficients_m"]) == 12
    opd = np.load(tmp_path / "result.npz")["opd"]
    assert sp.wavefront_error(opd, truth, pupil, remove="tiptilt") < 2e-9


def test_retrieve_rejects_unknown_formats(tmp_path) -> None:
    (tmp_path / "images.txt").write_text("1 2 3")
    with pytest.raises(SystemExit):
        main(["retrieve", str(tmp_path / "images.txt"), "--wavelength", "1e-6", "--sampling", "2"])


def test_metric_mode_removal_options() -> None:
    keck = sp.Pupil.keck(64)
    pistons = np.zeros(keck.shape)
    for s in range(1, keck.n_segments + 1):
        pistons[keck.segments == s] = 1e-8 * s
    assert sp.rms(pistons, keck, "segment_piston") < 1e-20
    pupil = sp.Pupil.circular(32, 1.0)
    _, x = pupil.coordinates()
    assert sp.rms(np.where(pupil.mask, x, 0.0), pupil, ["piston", "tip"]) < 1e-12
    assert sp.rms(np.where(pupil.mask, 1.0, 0.0), pupil, None) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="unknown mode"):
        sp.rms(np.zeros(pupil.shape), pupil, "coma")
    with pytest.raises(ValueError, match="segmented"):
        sp.rms(np.zeros(pupil.shape), pupil, "segment_piston")
    with pytest.raises(ValueError, match="does not match"):
        sp.remove_modes(np.zeros((3, 3)), pupil)
