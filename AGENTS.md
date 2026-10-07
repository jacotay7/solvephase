# AGENTS.md

Operating guide for AI agents (and humans) working in `solvephase`. Keep it
accurate: update it in the same change that alters the layout, commands,
contracts or conventions below, and add a Gotchas entry whenever something led
you astray.

## What this repository is

`solvephase` is a phase-retrieval library: given intensity measurements, it
recovers the phase (wavefront) that produced them. It covers

- **focal-plane wavefront sensing** for adaptive optics and optical metrology:
  Gerchberg-Saxton/Misell, nonlinear (gradient / Gauss-Newton) retrieval with
  Gaussian, Poisson or amplitude likelihoods, phase diversity (point source
  and extended object), LIFT, Fast & Furious;
- **coherent diffraction imaging**: ER, HIO, DM, RAAR, RRR, ASR, HPR, OSS,
  shrinkwrap, batched multi-start;
- **generic measurement models**: Wirtinger flow family with spectral
  initialization;
- **transport of intensity** (TIE).

It runs on NumPy/SciPy, or on CUDA through CuPy with the same code.

It belongs to an AO simulation family and reuses rather than copies it:
[`aobasis`](https://github.com/jacotay7/aobasis) supplies modal bases (a core
dependency); [`pyturb`](https://github.com/jacotay7/pyturb) (atmospheric OPD)
and [`getframes`](https://github.com/jacotay7/getframes) (detector noise) are
optional `interop` dependencies used for realistic data; HCIPy is an optional
independent reference for validation only. Never hard-code a filesystem path
to a sibling repository.

## Layout

```text
src/solvephase/
  backend.py        Backend (NumPy | CuPy, single | double), to_numpy, get_backend
  propagation.py    FFT, MFT, focal-plane and angular-spectrum propagators (exact adjoints)
  pupil.py          Pupil: anti-aliased analytic, segmented, telescope presets
  basis.py          Basis: Zernike/KL/Fourier (via aobasis), zonal, segments, DM
  focal.py          FocalPlaneModel: forward, vjp (reverse mode), jvp (forward mode)
  losses.py         Gaussian, Poisson (deviance), amplitude losses with curvature
  optimize.py       device-resident L-BFGS (strong Wolfe), Levenberg-Marquardt, Adam
  retrieval.py      FocalPlaneProblem + solve(): the nonlinear focal-plane solver
  unwrap.py         weighted least-squares phase unwrapping (DCT-preconditioned CG)
  metrics.py        rms, wavefront_error (ambiguity-aware), strehl_from_rms
  result.py         Result returned by every focal-plane solver
  algorithms/       one module per algorithm family (gerchberg_saxton, cdi, ...)
  api.py            retrieve(): the one-call high-level entry point
tests/              pytest; mirrors the module names
benchmarks/         speed suite (run.py), head-to-head baselines (compare.py), method
                    comparison feeding docs/choosing.md (methods.py), JSON artifacts
validation/         physics/statistics evidence (Cramer-Rao, independent references)
docs/               mkdocs-material site
examples/           headless, deterministic scripts
```

## Quality gate (run before handing off)

```bash
ruff check .
ruff format --check .
python -m mypy
python -m pytest -q --cov=solvephase --cov-report=term-missing
python -m pytest -q --run-slow -m slow          # convergence/statistics tests
python -m pytest -q --run-gpu -m gpu            # needs CuPy + a CUDA device
mkdocs build --strict
python -m build
```

CI has no GPU, so `--run-gpu` locally is the only check of the CuPy paths;
note the GPU and CuPy version when you report GPU results.

## Conventions (stable contracts)

- **Arrays are `(y, x)`**; coordinates sit on pixel centres, centred:
  `(i - (n - 1) / 2) * pitch` (`propagation.centered_coordinates`). The image
  of an on-axis point source is centred in the window (between pixels for
  even sizes).
- **Sign:** Fraunhofer kernel `exp(-2 pi i x.alpha / lambda)`; a pupil OPD ramp
  `a * x` moves the image to angle `+a` (towards larger column index). Any
  sign/transpose fix needs an analytic ramp test, never trial and error.
- **Units:** SI. OPD in metres is the user-facing wavefront; modal
  coefficients are RMS OPD in metres (modes are unit RMS). Internally,
  solvers work in radians at the reference wavelength. Pixel scales are
  radians, or `sampling` in pixels per lambda/D (2 = Nyquist).
- **Normalization:** propagators are unitary (Parseval holds when the window
  captures all light). `FocalPlaneModel` images sum to 1 per channel for an
  unbounded detector; a finite window loses light and is never renormalized.
- **Adjoints are exact.** Every linear operator ships `forward` and
  `adjoint`, and has an adjoint test `<A x, y> == <x, A^H y>`. Every analytic
  gradient has a finite-difference test.
- **Backends:** write numerical code against `backend.xp` and `Backend`
  methods (`fft2`, `asarray`, `to_numpy`, ...). Never call NumPy allocation or
  FFT directly in a hot path. Host transfers happen only at named boundaries
  (`to_numpy`, scalar convergence checks every `check_every` iterations).
  GPU results stay on the GPU.
- **Precision:** double on CPU and single on GPU by default; float32 and
  float64 must both work. Don't silently promote float32 hot paths to float64.
- **Randomness:** draw on the host from a `numpy.random.Generator`
  (`Backend.random(seed)`) and move to the device, so one seed gives the same
  starts on CPU and GPU. Never use global `np.random` state.
- **Results:** focal-plane solvers return `result.Result`; other families
  return `Result` too when they produce a pupil wavefront, otherwise a small
  dataclass documented in their module. Every result records `history`,
  `n_iter`, `converged`, `message` and `elapsed`.
- **Public API:** export from `solvephase/__init__.py` only what users need;
  internal helpers start with `_`.

## Writing code here

- Start from a primary reference; cite it in the module docstring and the
  docs page.
- A test that only checks "it ran" is not a test. Assert recovery accuracy
  against a known truth, adjointness, gradients, invariants (piston, flux,
  shift), or statistics against theory with ensemble tolerances.
- Mark tests over ~2 s `@pytest.mark.slow`, GPU tests `@pytest.mark.gpu`
  (parametrize devices with `tests/conftest.py::devices()`).
- Public functions have type hints and NumPy-style docstrings with units.
- Error messages say why, and what to do instead.
- Comments and docstrings describe current behaviour; history goes in
  `CHANGELOG.md` (under Unreleased).
- Every user-visible feature lands with its docs page section and a
  CHANGELOG entry.
- Docs code fenced as ` ```python ` is executed by `tests/test_docs.py`
  (each page's blocks in order, as one script). Fence illustrative fragments
  that cannot run alone as ` ```py `.

## Maintainer rules

- Minor issues worked around rather than fixed get a GitHub issue in the
  affected repository (when credentials allow), linked from a comment at the
  workaround. Bugs in the maintainer's own repos (aobasis, pyturb, getframes,
  makewfs) are fixed at the source, not worked around here. Never file issues
  on third-party repositories; record them in a solvephase issue instead.
- No planning or status-tracker files in the repo (ROADMAP.md is the one
  forward-looking document).
- Everything must be usable under the MIT license: never copy code from
  GPL/LGPL/non-commercial/CeCILL sources, and never add patent-encumbered
  algorithms. PIE-family ptychography (Phase Focus patents) is excluded on
  these grounds; other ptychography only if it is clearly unencumbered.

## Gotchas

- `aobasis` modes are columns of an `(n_points, n_modes)` matrix;
  `Basis.modes` stores the transpose `(n_modes, n_valid)`.
- aobasis warns when points lie outside `pupil_radius`; anti-aliased rim
  pixels always do, so `Basis.zernike` silences that warning on purpose.
- The Poisson loss is evaluated as a deviance (non-negative). The raw
  negative log-likelihood carries a huge data constant that made relative
  `ftol` stops fire after a few iterations.
- Unweighted Gaussian least squares on raw PSFs is badly conditioned for
  zonal retrieval (the core dominates); prefer Poisson or amplitude losses.
- Spider vanes split a pupil into disconnected regions; `unwrap_phase`
  bridges their 2 pi multiples with a smooth-surface fit (`bridge=True`).
- The angular-spectrum propagator removes evanescent waves; with a pitch
  below the wavelength most of the spectrum is evanescent and the field is
  not preserved at zero distance.
- `offset` on `FocalPlaneModel`/propagators shifts the detector *window*:
  pixel `j` samples angle `(j - (m-1)/2 + offset) * pixel_scale`, so the
  optical axis lands on pixel `(m-1)/2 - offset`. `offset=-0.5` reproduces
  HCIPy / `fftshift` centring.
- CPU threading traps: threaded OpenBLAS level-1 calls (`np.vdot`,
  `np.dot`, `np.linalg.norm` on >10k elements) stall ~1 ms each when the
  cores are busy - use `Backend.dot`; and many-thread SciPy FFTs are slower
  than one thread for small transforms - `backend._cpu_workers(size)` scales
  the thread count with the transform size.
- Float32 losses: evaluate data terms and accumulate normal equations in
  float64 (`FocalPlaneProblem` does). Float32 loss values are too noisy for
  line-search and LM acceptance and stall solvers above ~1e5 counts/pixel.
- The Poisson and amplitude losses continue quadratically below a small
  positive floor; clamping instead gave gradients inconsistent with the
  value whenever a line search probed negative model values.
- Background parameters are scaled by robust border noise, not the image
  standard deviation (a bright PSF core inflates it by orders of magnitude).
- Simulated data must exercise what you fit: tests once failed because data
  had a background the problem did not fit, or modes the basis lacked.
- Extended-scene phase diversity needs compact scenes (blurred scene inside
  the field); apodization subtracts the edge level, not the mean; the
  reduced metric's regularization must be tiny (~1/peak SNR^2).
- CDI: a square or symmetric support invites twin/translation stagnation in
  single-start tests (use irregular outlines or several starts); RAAR needs
  beta close to 1 on noise-free data.
- TIE: masked/Neumann CG in float32 must project the per-region constant out
  of the preconditioned residual and make right-hand sides consistent;
  samples must vanish well inside the window (periodic propagation edges
  otherwise corrupt dI/dz).
- LIFT: an astigmatism coefficient in the basis produces an exact twin
  `a' = -a - 2d`; this is physics, not a solver bug (`LIFT` reports the
  smaller one).
- Fast & Furious needs an integer FFT size and a centro-symmetric pupil, and
  its first step must apply even diversity (`first_even=True`).
- OpenBLAS level-2/3 calls (`matmul`, gemv) on small operands stall like
  level-1 under load (~3 ms vs 80 us for `np.einsum`); `MatrixOperator`
  uses einsum on the CPU for small products.
- Octanary coded-diffraction spectral initializations need `diag(A^H A)`
  whitening beyond ~1e4 pixels, otherwise the leading eigenvector locks onto
  single pixels. Wirtinger flow with the paper's `mu_max = 0.2` diverges for
  n >= 128 (default 0.1). Stop slowly contracting first-order methods on
  residual stagnation, not on iterate change.
- Fourier magnitudes alone defeat the Wirtinger family (shift and twin
  ambiguities); that model belongs to the CDI projection solvers.
- Always pass `encoding="utf-8"` to `read_text`/`write_text`/`open` for text:
  Windows defaults to cp1252 and the docs contain non-ASCII characters.
- Chaotic projection algorithms (DM, RAAR, HIO) land in platform-dependent
  places after a fixed number of iterations (FFT rounding differs across
  OS/BLAS builds); assert robust reductions, not tight ratios.
- Tests import shared helpers as `from conftest import devices` (`tests/`
  is not a package).
- GPU timings on a shared device (or CPU timings under load) vary 2-5x;
  record the load and hardware with any performance claim.
