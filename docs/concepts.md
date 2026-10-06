# Concepts

## The problem

A detector records intensity $I = |E|^2$, never the field $E$ itself. When
the field comes from a known optical system acting on an unknown phase, it
is often possible to recover the phase from intensities alone. The
measurements must be oversampled or diverse enough, and the ambiguities
handled.

In solvephase every problem is a forward model

$$
I_k = \left| \mathcal{A}_k\, u(\theta) \right|^2 + \text{noise},
$$

where $u$ is the unknown complex field (for example a pupil
$a\, e^{i\theta}$), $\mathcal{A}_k$ is a known linear propagator (a
Fourier transform, Fresnel propagation, a coded mask), and $k$ indexes
*channels*: diversity images, wavelengths, scan positions or masks.

## Two algorithm families

**Projections** (Gerchberg-Saxton, ER, HIO, RAAR, ...) alternate between
enforcing the measured moduli and enforcing what is known about $u$
(support, pupil amplitude). They are cheap per iteration, have a wide
capture range and are robust to bad starts. They are hard to combine with
noise models or nuisance parameters, and they stagnate.

**Optimization** (nonlinear least squares, maximum likelihood, Wirtinger
flows) minimizes a data misfit such as the Poisson negative log-likelihood
directly. It needs gradients. solvephase computes them analytically by
propagating residuals back through the exact adjoint of each operator. It
can also fit flux, background, registration and noise weights, and it
reaches the Cramér-Rao bound. A good start helps, which is why solvephase
often seeds it with a projection algorithm.

## Ambiguities

Phase retrieval only determines the phase up to transformations that leave
every measured intensity unchanged:

- **Piston** (a constant phase) is always invisible.
- **Tip/tilt** is invisible when the image position is not known
  (registration).
- **The twin.** For one in-focus image of a centro-symmetric pupil,
  $\theta(x)$ and $-\theta(-x)$ give the same image, so the even part of the
  phase has an undetermined sign. A known **diversity**, such as a defocused
  second image, an astigmatism bias (LIFT) or a previous DM step (Fast &
  Furious), breaks the symmetry.
- **CDI** adds translation and complex-conjugate inversion.

[`wavefront_error`][solvephase.wavefront_error] and the CDI helper
`align_object` compare results modulo exactly these ambiguities.

## Sampling

A focal-plane image must sample the PSF at least at Nyquist (2 pixels per
$\lambda/D$) to capture all spatial frequencies of the pupil autocorrelation.
Coarser detectors are handled exactly by the matrix Fourier transform and
pixel integration (`oversample`). Retrieval still works, but higher-order
modes become harder to determine. CDI needs an oversampling ratio above 2 in
each dimension (the object support must be smaller than half the array).

## Diversity design

With two images, a defocus diversity of about 0.5–1 wave peak-to-valley
($\approx 0.15$–$0.3\,\lambda$ RMS) is a good default for wavefronts up to
about $0.3\,\lambda$ RMS. Larger aberrations need larger diversity or more
planes; very small aberrations are best sensed with smaller diversity. See
[Focal-plane retrieval](guide/focal-plane.md#choosing-the-diversity).
