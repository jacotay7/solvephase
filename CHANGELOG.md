# Changelog

All notable changes to `solvephase` are documented here. The project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

First release.

### Focal-plane phase retrieval

- `FocalPlaneModel`: pupil-to-detector image formation for any number of
  diversity channels.
  - Broadband light (exact per-wavelength sampling).
  - Any detector sampling, including undersampled, with pixel integration
    (`oversample`).
  - Reverse-mode (`vjp`) and forward-mode (`jvp`) derivatives, verified
    against finite differences.
- Propagators with exact adjoints:
  - zero-padded FFT;
  - matrix Fourier transform, stacked over wavelengths;
  - angular spectrum (Fresnel).

  An automatic engine choice picks FFT or MFT per model. The model matches
  HCIPy to machine precision.
- `FocalPlaneProblem` + `solve`: maximum-likelihood retrieval.
  - Losses: Poisson deviance with read noise, weighted Gaussian, or the
    amplitude metric.
  - Nuisance fits: per-image flux, background and registration tilt, and an
    optional zonal pupil amplitude.
  - Regularization: Tikhonov, a KL-turbulence prior (MAP), and zonal
    smoothness.
  - Optimizers: Levenberg-Marquardt with an exact Gauss-Newton/Fisher
    matrix, L-BFGS with a strong-Wolfe line search, or Adam. All are
    device-resident.
  - Single precision reaches the double-precision noise floor, because
    likelihoods and normal equations accumulate in float64.
- `gerchberg_saxton`: Gerchberg-Saxton and the multi-plane Misell algorithm,
  with exact modulus projection on the full FFT grid.
- `unwrap_phase`: weighted least-squares unwrapping over obscured, spidered
  and segmented pupils (DCT-preconditioned CG), with 2π bridging of
  disconnected regions.
- `retrieve`: one-call retrieval with a robust default strategy.
  - Amplitude-metric Levenberg-Marquardt from a flat start and from
    Gerchberg-Saxton; the better fit is kept.
  - Then a Poisson maximum-likelihood polish and optional zonal refinement.

### Algorithm families

- Phase diversity with an unknown extended object
  (`phase_diversity`, `PhaseDiversityProblem`).
  - Gonsalves/Paxman reduced metric with an exact gradient.
  - Apodization and a diffraction-cutoff frequency mask.
  - Automatic noise and regularization estimates.
  - Wiener object estimate.
- LIFT (`lift`, `LIFT`, `lift_crlb`): maximum-likelihood low-order sensing
  from one astigmatic image, with a twin check and Cramér-Rao bounds.
- Fast & Furious (`FastAndFurious`, `simulate_closed_loop`): sequential
  weak-phase focal-plane sensing with the Korkiakoski et al. (2014)
  modifications. It takes one FFT pair per step.
- Coherent diffraction imaging (`cdi`).
  - Algorithms: ER, HIO, DM, RAAR, RRR, ASR/Douglas-Rachford, HPR and OSS.
  - Constraints: real, positive, amplitude bounds; beamstop masks.
  - Chainable schedules, shrinkwrap, and batched multi-start (one batched
    FFT per iteration).
  - Helpers: `autocorrelation_support`, `align_object` (ambiguity-aware) and
    `simulate_cdi`.
- Generic measurement models (`solvephase.algorithms.wirtinger`,
  `solvephase.operators`).
  - Algorithms: Wirtinger flow, truncated Wirtinger flow, truncated and
    reweighted amplitude flows, and an L-BFGS solver.
  - Initializations: spectral, truncated, orthogonality-promoting and
    optimal-preprocessing spectral.
  - Operators: Gaussian matrix, coded diffraction and oversampled Fourier,
    each with an exact adjoint.
- Transport of intensity (`tie`, `simulate_defocus_stack`).
  - Solvers: FFT and DCT Poisson, uniform intensity and Teague's non-uniform
    solution, and an exact masked conjugate-gradient solver.
  - Axial derivative: central differences or a multi-plane polynomial fit.

### Optics and integration

- `Pupil`: anti-aliased circular, annular and spidered pupils; segmented
  hexagonal apertures with segment labels; Keck, JWST, VLT and Hubble
  presets.
- `Basis`:
  - Zernike (annular), KL (with prior variances), Fourier and any other
    aobasis generator;
  - segment piston/tip/tilt;
  - Gaussian and measured DM influence functions;
  - zonal.
- `solvephase.interop`: pyturb turbulence (`turbulence_opd`) and getframes
  detector frames (`expose`). `expose` converts ADU to electrons and masks
  saturated pixels.
- Ambiguity-aware metrics: `rms`, `wavefront_error` (with twin handling) and
  `strehl_from_rms`.
- A `solvephase` command line with `info` and `retrieve`.
- CPU threading tuned for contended and hyper-threaded machines:
  - FFT worker counts scale with the transform size.
  - BLAS threads are limited, per call, for matrix Fourier transforms and
    Gauss-Newton products (via threadpoolctl). This made broadband
    gradients up to 10x faster.
  - Scalar products avoid threaded level-1 BLAS.

### Comparison

- `benchmarks/methods.py` runs every method on one reference problem. All
  focal-plane methods see the same wavefront. It reports:
  - the data each method needs;
  - accuracy at a standard SNR;
  - warm CPU and GPU time;
  - capture range.

  Its output feeds the comparison tables on the "Choosing an algorithm" docs
  page, which also has a decision flowchart.
- The README showcase races every focal-plane method on one clock, with a
  live error-versus-time chart, plus a gallery of CDI, coded diffraction and
  TIE.
- The Keck preset now follows the documented geometry: 10.95 m across the
  corners, pointy-top segments, six 26 mm support arms, and a 2.57 m central
  obscuration.

### Quality

- More than 280 tests: adjoints, gradients, recovery accuracy, Cramér-Rao
  efficiency, and CPU/GPU parity.
- A validation suite against theory and HCIPy.
- Benchmark and comparison suites with versioned artifacts.
- An mkdocs documentation site whose code examples are executed in CI.
