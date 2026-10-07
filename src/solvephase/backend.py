"""Array backends: re-exported from :mod:`aocore.backend` (the stack's shared backend).

See :mod:`aocore.backend` for the full documentation.
"""

from aocore.backend import (
    Backend,
    BackendLike,
    _cpu_workers,
    backend_of,
    get_backend,
    gpu_available,
    to_numpy,
)

__all__ = [
    "Backend",
    "BackendLike",
    "_cpu_workers",
    "backend_of",
    "get_backend",
    "gpu_available",
    "to_numpy",
]
