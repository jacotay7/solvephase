"""solvephase: fast, GPU-optional phase retrieval.

The most common entry points:

* :func:`retrieve` - one call from focal-plane images to a wavefront;
* :class:`Pupil`, :class:`Basis`, :class:`FocalPlaneModel` - describe the optics;
* :class:`FocalPlaneProblem` + :func:`solve` - full control of the
  nonlinear focal-plane solver;
* :func:`phase_diversity`, :func:`lift`, :class:`FastAndFurious`,
  :func:`cdi`, :func:`wirtinger`, :func:`tie`, :func:`gerchberg_saxton` - the algorithm
  families (see :mod:`solvephase.algorithms`).
"""

from .__about__ import __version__
from .algorithms.cdi import (
    CDIResult,
    ShrinkwrapConfig,
    align_object,
    autocorrelation_support,
    cdi,
    simulate_cdi,
)
from .algorithms.fast_furious import ClosedLoopResult, FastAndFurious, simulate_closed_loop
from .algorithms.gerchberg_saxton import gerchberg_saxton
from .algorithms.lift import LIFT, lift, lift_crlb
from .algorithms.phase_diversity import PhaseDiversityProblem, phase_diversity
from .algorithms.tie import TIEResult, simulate_defocus_stack, tie
from .algorithms.wirtinger import GenericResult, relative_error, wirtinger
from .api import retrieve
from .backend import Backend, backend_of, get_backend, gpu_available, to_numpy
from .basis import Basis
from .focal import FocalPlaneModel, zernike_diversity
from .losses import AmplitudeLoss, GaussianLoss, Loss, PoissonLoss
from .metrics import remove_modes, rms, strehl_from_rms, wavefront_error
from .operators import (
    CodedDiffractionOperator,
    LinearOperator,
    MatrixOperator,
    OversampledFourierOperator,
)
from .propagation import (
    AngularSpectrumPropagator,
    FFTPropagator,
    FocalPlanePropagator,
    MFTPropagator,
)
from .pupil import Pupil
from .result import Result
from .retrieval import FocalPlaneProblem, noise_weights, solve
from .simulate import random_aberration, simulate_images
from .unwrap import unwrap_phase, wrap

__all__ = [
    "LIFT",
    "AmplitudeLoss",
    "AngularSpectrumPropagator",
    "Backend",
    "Basis",
    "CDIResult",
    "ClosedLoopResult",
    "CodedDiffractionOperator",
    "FFTPropagator",
    "FastAndFurious",
    "FocalPlaneModel",
    "FocalPlaneProblem",
    "FocalPlanePropagator",
    "GaussianLoss",
    "GenericResult",
    "LinearOperator",
    "Loss",
    "MFTPropagator",
    "MatrixOperator",
    "OversampledFourierOperator",
    "PhaseDiversityProblem",
    "PoissonLoss",
    "Pupil",
    "Result",
    "ShrinkwrapConfig",
    "TIEResult",
    "__version__",
    "align_object",
    "autocorrelation_support",
    "backend_of",
    "cdi",
    "gerchberg_saxton",
    "get_backend",
    "gpu_available",
    "lift",
    "lift_crlb",
    "noise_weights",
    "phase_diversity",
    "random_aberration",
    "relative_error",
    "remove_modes",
    "retrieve",
    "rms",
    "simulate_cdi",
    "simulate_closed_loop",
    "simulate_defocus_stack",
    "simulate_images",
    "solve",
    "strehl_from_rms",
    "tie",
    "to_numpy",
    "unwrap_phase",
    "wavefront_error",
    "wirtinger",
    "wrap",
    "zernike_diversity",
]
