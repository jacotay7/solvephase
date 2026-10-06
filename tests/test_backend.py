from __future__ import annotations

import numpy as np
import pytest

import solvephase as sp
from conftest import devices
from solvephase.backend import Backend, backend_of, get_backend, to_numpy


def test_cpu_defaults_to_double() -> None:
    be = get_backend("cpu")
    assert be.device == "cpu" and be.precision == "double"
    assert be.real_dtype == np.float64 and be.complex_dtype == np.complex128
    assert be.xp is np


def test_backends_are_cached_and_hashable() -> None:
    assert get_backend("cpu", "single") is get_backend("cpu", "float32")
    assert {get_backend("cpu"): 1}[get_backend("cpu", "double")] == 1


def test_existing_backend_passes_through_and_overrides() -> None:
    be = get_backend("cpu", "single")
    assert get_backend(be) is be
    assert get_backend(be, "double").precision == "double"


@pytest.mark.parametrize("bad", ["tpu", "cuda:7x"])
def test_unknown_device_is_rejected(bad: str) -> None:
    with pytest.raises(ValueError, match="device"):
        get_backend(bad)


def test_unknown_precision_is_rejected() -> None:
    with pytest.raises(ValueError, match="precision"):
        get_backend("cpu", "half")


def test_auto_resolves_to_an_available_device() -> None:
    be = get_backend("auto")
    assert be.device == ("gpu" if sp.gpu_available() else "cpu")


def test_asarray_promotes_to_working_precision() -> None:
    be = get_backend("cpu", "single")
    assert be.asarray(np.ones(3)).dtype == np.float32
    assert be.asarray(np.ones(3, complex)).dtype == np.complex64
    assert be.asarray(np.ones(3, int)).dtype.kind == "i"
    assert be.asarray([1, 2], dtype="complex").dtype == np.complex64


@pytest.mark.parametrize("device", devices())
def test_unitary_fft_round_trip(device: str, rng: np.random.Generator) -> None:
    be = get_backend(device, "double")
    x = be.asarray(rng.standard_normal((3, 16, 12)) + 1j * rng.standard_normal((3, 16, 12)))
    y = be.fft2(x)
    assert np.isclose(be.dot(y, y), be.dot(x, x))
    np.testing.assert_allclose(to_numpy(be.ifft2(y)), to_numpy(x), atol=1e-12)


def test_dot_matches_numpy(rng: np.random.Generator) -> None:
    be = get_backend("cpu")
    a = rng.standard_normal(20000)
    b = rng.standard_normal(20000)
    assert np.isclose(be.dot(a, b), float(a @ b))
    za = a + 1j * b
    assert np.isclose(be.dot(za, za), float(np.sum(np.abs(za) ** 2)))


def test_backend_of_infers_precision() -> None:
    assert backend_of(np.zeros(2, np.float32)).precision == "single"
    assert backend_of(np.zeros(2)).precision == "double"
    assert isinstance(backend_of(np.zeros(2)), Backend)


@pytest.mark.gpu
def test_gpu_defaults_to_single_and_round_trips() -> None:
    be = get_backend("gpu")
    assert be.precision == "single"
    x = be.asarray(np.arange(4.0))
    assert type(x).__module__.startswith("cupy")
    np.testing.assert_array_equal(to_numpy(x), np.arange(4.0, dtype=np.float32))
    assert backend_of(x).device == "gpu"
