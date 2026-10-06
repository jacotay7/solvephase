"""Shared pytest options and fixtures."""

from __future__ import annotations

import numpy as np
import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-gpu", action="store_true", help="run CuPy/CUDA tests")
    parser.addoption("--run-slow", action="store_true", help="run slow tests")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    from solvephase.backend import gpu_available

    skip_gpu = pytest.mark.skip(reason="needs --run-gpu and a CUDA device with CuPy")
    skip_slow = pytest.mark.skip(reason="needs --run-slow")
    run_gpu = config.getoption("--run-gpu") and gpu_available()
    for item in items:
        if "gpu" in item.keywords and not run_gpu:
            item.add_marker(skip_gpu)
        if "slow" in item.keywords and not config.getoption("--run-slow"):
            item.add_marker(skip_slow)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


def devices() -> list[object]:
    """Parametrization helper: CPU always, GPU marked."""
    return ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)]
