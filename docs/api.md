# API reference

## High level

::: solvephase.retrieve

::: solvephase.Result

## Optics

::: aocore.Pupil

::: solvephase.Basis

::: solvephase.FocalPlaneModel

::: solvephase.zernike_diversity

## Focal-plane solvers

::: solvephase.FocalPlaneProblem

::: solvephase.solve

::: solvephase.gerchberg_saxton

::: solvephase.noise_weights

## Algorithm families

::: solvephase.algorithms.phase_diversity

::: solvephase.algorithms.lift

::: solvephase.algorithms.fast_furious

::: solvephase.algorithms.cdi

::: solvephase.algorithms.wirtinger

::: solvephase.operators

::: solvephase.algorithms.tie

## Losses and optimizers

::: solvephase.losses

::: solvephase.optimize

## Shared primitives (aocore)

The backend, pupils, propagators, wavefront metrics and phase unwrapping come
from [aocore](https://github.com/jacotay7/aocore), the stack's shared core.
solvephase re-exports them unchanged (`solvephase.Pupil`,
`solvephase.FocalPlanePropagator`, `solvephase.rms`, ...).

::: aocore.propagation

::: aocore.metrics

::: aocore.unwrap

::: aocore.backend

## Simulation

::: solvephase.simulate
