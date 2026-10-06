# Roadmap

This file tracks only what is **not done yet**. Ground rules: every algorithm
lands with tests that assert accuracy against a known truth or theory, a docs
section with its primary references, and a benchmark entry. Every speed claim
lands with a benchmark artifact.

## Algorithms

- [ ] **Ptychography** (PIE/ePIE/rPIE/mPIE, difference-map and LSQ-ML
      ptychography, position refinement, mixed states). Deferred until the
      status of the Phase Focus PIE patents has been checked for an MIT
      release. Difference-map and maximum-likelihood ptychography may be
      unencumbered and could land first.
- [ ] **Extended-scene phase diversity for field-filling scenes** (solar
      granulation). The Fourier-domain reduced metric with apodization is
      only approximate there. A spatial-domain joint object/wavefront
      estimator with a field mask would fix it.
- [ ] **Near-field (Fresnel) multi-plane retrieval**: Misell/GS and the
      nonlinear solver over angular-spectrum propagated planes (the
      `AngularSpectrumPropagator` already has an exact adjoint).
- [ ] **Differential OTF (dOTF)**, the **asymmetric-pupil Fourier WFS** and
      **kernel phase**: linear focal-plane sensing methods (Codona 2012;
      Martinache 2010, 2013).
- [ ] **Multi-wavelength segment phasing** with $2\pi$-ambiguity resolution
      for Keck/JWST/ELT-like apertures.
- [ ] **Vector (high-NA) PSF models** for microscopy.
- [ ] **PhaseMax/PhaseLift and Gauss-Newton** for generic measurement
      models.

## Performance

- [ ] **Batched independent focal-plane problems** (many field points or
      time steps) in one GPU solve.
- [ ] **Fused CUDA kernels and CUDA-graph capture** for small problems,
      where the GPU is bound by kernel launch overhead.
- [ ] **Optional autodiff backends** (PyTorch, JAX) for user-defined
      forward models, kept as alternatives to the hand-written adjoints.

## Integration

- [ ] **Real-time NCPA loops**: a `shmpipeline`/pyRTC plugin running Fast &
      Furious or LIFT on streamed frames.
- [ ] **A PhasePack-style benchmark harness** comparing algorithm families
      across noise levels, sampling ratios and measurement models.

## Non-goals

- Simulating atmospheres, wavefront sensors or detectors. pyturb, makewfs
  and getframes own those, and solvephase consumes their outputs.
- Closed-loop AO control, which belongs to pyRTC and shmpipeline-ao.
