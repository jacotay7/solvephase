from __future__ import annotations

import numpy as np
import pytest

from solvephase import Basis, Pupil


def test_circular_area_and_antialiasing() -> None:
    p = Pupil.circular(128, 2.0, obscuration=0.3, supersample=8)
    expected = np.pi * (1.0**2 - 0.3**2)
    assert abs(np.sum(p.amplitude) * p.pitch**2 - expected) / expected < 2e-3
    edge = p.amplitude[(p.amplitude > 0) & (p.amplitude < 1)]
    assert edge.size > 0  # grey rim pixels
    assert p.shape == (128, 128) and p.n_valid == int(np.count_nonzero(p.amplitude))


def test_spiders_block_light_symmetrically() -> None:
    p = Pupil.circular(128, 1.0, spiders=4, spider_width=0.03)
    full = Pupil.circular(128, 1.0)
    blocked = full.amplitude.sum() - p.amplitude.sum()
    assert blocked > 0
    np.testing.assert_allclose(p.amplitude, p.amplitude[::-1, ::-1], atol=1e-12)


@pytest.mark.parametrize(("factory", "segments"), [(Pupil.keck, 36), (Pupil.jwst, 18)])
def test_segmented_presets_have_labelled_segments(factory, segments) -> None:
    p = factory(128)
    assert p.n_segments == segments
    assert set(np.unique(p.segments)) == set(range(segments + 1))
    assert np.all(p.segments[p.amplitude == 0] == 0)


def test_pupil_validation_messages() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        Pupil(-np.ones((4, 4)), pitch=1.0, diameter=1.0)
    with pytest.raises(ValueError, match="zero everywhere"):
        Pupil(np.zeros((4, 4)), pitch=1.0, diameter=1.0)
    with pytest.raises(ValueError, match="obscuration"):
        Pupil(np.ones((4, 4)), pitch=1.0, diameter=1.0, obscuration=1.2)


def test_from_array_infers_pitch_from_diameter() -> None:
    amp = np.zeros((10, 10))
    amp[2:8, 2:8] = 1
    p = Pupil.from_array(amp, diameter=3.0)
    assert np.isclose(p.pitch, 0.5)


def test_padding_and_downsampling_preserve_light() -> None:
    p = Pupil.circular(64, 1.0)
    assert np.isclose(p.padded(96).amplitude.sum(), p.amplitude.sum())
    d = p.downsampled(2)
    assert d.shape == (32, 32) and np.isclose(d.pitch, 2 * p.pitch)
    assert np.isclose(d.amplitude.sum() * 4, p.amplitude.sum())


def test_zernike_modes_are_unit_rms_and_fit_inverts_synthesize(rng) -> None:
    p = Pupil.circular(96, 1.0)
    z = Basis.zernike(p, 20)
    assert z.labels[:3] == ["Z2", "Z3", "Z4"]
    w = p.amplitude**2
    for m in z.mode_maps([0, 2, 10]):
        assert abs(np.sqrt(np.sum(w * m**2) / w.sum()) - 1.0) < 0.02
    c = rng.standard_normal(20) * 1e-7
    np.testing.assert_allclose(z.fit(z.synthesize(c)), c, atol=1e-18)


def test_zernike_tip_is_a_positive_x_ramp() -> None:
    p = Pupil.circular(64, 1.0)
    tip = Basis.zernike(p, 1).mode_maps()[0]
    _, x = p.coordinates()
    assert np.corrcoef(tip[p.mask], x[p.mask])[0, 1] > 0.999


def test_orthonormalized_basis_is_orthonormal_over_the_pupil() -> None:
    p = Pupil.circular(64, 1.0, obscuration=0.2, spiders=3, spider_width=0.05)
    z = Basis.zernike(p, 15, orthonormalize=True)
    w = (p.amplitude**2)[p.mask]
    gram = (z.modes * w) @ z.modes.T / w.sum()
    np.testing.assert_allclose(gram, np.eye(15), atol=1e-10)


def test_kl_variances_match_kolmogorov_theory() -> None:
    p = Pupil.circular(32, 8.0)
    kl = Basis.kl(p, 20, r0=0.2, outer_scale=np.inf)
    lam = 500e-9
    scale = (8.0 / 0.2) ** (5 / 3) * (lam / (2 * np.pi)) ** 2
    # Noll (1976): tip + tilt carry 0.896 (D/r0)^(5/3) rad^2; KL captures slightly more.
    ratio = kl.variances[:2].sum() / (0.896 * scale)
    assert 0.95 < ratio < 1.2
    assert np.all(np.diff(kl.variances) <= 1e-9 * kl.variances[0])


def test_segment_and_dm_and_fourier_bases() -> None:
    keck = Pupil.keck(96)
    seg = Basis.segments(keck)
    assert seg.n_modes == 3 * 36 and seg.labels[0] == "S1-piston"
    p = Pupil.circular(48, 1.0)
    dm = Basis.gaussian_dm(p, 9, stroke=1e-6)
    assert dm.kind == "dm" and dm.n_modes > 40
    assert np.isclose(dm.modes.max(), 1e-6, rtol=0.2)
    four = Basis.fourier(p, 10)
    assert four.n_modes == 10


def test_concatenate_and_subset() -> None:
    p = Pupil.circular(32, 1.0)
    a = Basis.zernike(p, 5)
    b = Basis.fourier(p, 3)
    c = a + b
    assert c.n_modes == 8 and c.labels[:5] == a.labels
    assert c.subset([0, 6]).labels == [a.labels[0], c.labels[6]]
    with pytest.raises(ValueError, match="zonal"):
        a.concatenate(Basis.zonal(p))


def test_zonal_basis_is_identity_on_valid_pixels(rng) -> None:
    p = Pupil.circular(16, 1.0)
    z = Basis.zonal(p)
    v = rng.standard_normal(p.n_valid)
    np.testing.assert_allclose(z.fit(z.synthesize(v)), v)
