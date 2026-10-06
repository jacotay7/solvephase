# Focal-plane retrieval

Recover a pupil wavefront from images of a point source taken through a known
optical system. This is the workhorse of non-common-path aberration (NCPA)
calibration, optical testing, image-based telescope phasing and focal-plane
wavefront sensing.

## The forward model

[`FocalPlaneModel`][solvephase.FocalPlaneModel] maps a pupil phase
$\theta$ (radians at the reference wavelength $\lambda_0$) to $K$ images,
one per diversity *channel*:

$$
I_k = \sum_{\ell} w_\ell \left| \mathcal{F}_{\lambda_\ell}\!\left[ a\,
e^{i \frac{\lambda_0}{\lambda_\ell} (\theta + \delta_k)} \right] \right|^2
\Big/ \textstyle\sum a^2,
$$

where $a$ is the pupil amplitude, $\delta_k$ the known diversity phase of
channel $k$, $w_\ell$ the spectral weights and $\mathcal{F}_\lambda$ the
Fraunhofer propagator at wavelength $\lambda$ onto the detector grid. It is a
zero-padded FFT when the sampling allows, and otherwise an exact matrix
Fourier transform (Soummer et al., Opt. Express 15, 15935, 2007). With
`oversample > 1` each detector pixel integrates
`oversample x oversample` model samples.

The data model adds per-channel flux $F_k$ and background $b_k$:
$M_k = F_k I_k + b_k$.

## Likelihoods

| `loss=` | Data term | When |
|---|---|---|
| `"poisson"` | Poisson deviance with read noise $\sigma$ (shifted Poisson, $d+\sigma^2 \sim \mathcal{P}(m+\sigma^2)$) | photon-counting data in electrons: statistically optimal |
| `"gaussian"` | $\tfrac12 \sum w (m-d)^2$ | known Gaussian errors (pass `weights=1/variance`, see [`noise_weights`][solvephase.noise_weights]) |
| `"amplitude"` | $\tfrac12 \sum w (\sqrt{m}-\sqrt{d})^2$ | widest capture range from poor starts; used by `retrieve` to explore |

All three come with exact gradients and Fisher curvature. Pixels with
`weights=0` (bad, saturated, outside a field stop) are ignored.

## Solvers

[`solve`][solvephase.solve] minimizes a
[`FocalPlaneProblem`][solvephase.FocalPlaneProblem]:

- `method="lm"`: Levenberg-Marquardt using the exact Gauss-Newton (Fisher)
  matrix, built with forward-mode derivatives. It converges in about 5–10
  iterations, which makes it the best choice for modal bases up to a few
  hundred modes.
- `method="lbfgs"`: limited-memory BFGS with a strong-Wolfe line search,
  for zonal (pixel) wavefronts, fitted amplitude, or thousands of modes.
- `method="adam"`: first-order, for experiments.

Gradients come from one adjoint propagation per channel and wavelength. This
is the reverse-mode differentiation of Jurling & Fienup (JOSA A 31, 1348,
2014), written out by hand for each operator.

[`gerchberg_saxton`][solvephase.gerchberg_saxton] implements GS (one image)
and Misell's multi-plane algorithm (several diversity images). It is a
projection method that is fast per iteration with a wide capture range. Its
wrapped phase is unwrapped with weighted least squares
([`unwrap_phase`][solvephase.unwrap_phase]) and can be fitted to a basis.

## Nuisance parameters

| Option | Fits |
|---|---|
| `fit_flux=True` (default) | total flux per image |
| `fit_background=True` | constant background per image |
| `fit_tilt=True` | registration tip/tilt of each image relative to the first |
| `fit_amplitude=True` | the pupil amplitude itself (zonal), e.g. for apodization or segment reflectivity |

## Regularization and priors

- `regularization=` adds Tikhonov damping on the coefficients (radians⁻²).
- `prior=True` with a KL basis (`Basis.kl(pupil, n, r0=...)`) adds the
  turbulence prior, giving a maximum a posteriori estimate.
- `smoothness=` penalizes first differences of a zonal phase.

## Choosing the diversity

With two images, one in focus and one defocused:

- Defocus of 0.15–0.3 $\lambda$ RMS (0.5–1 wave peak-to-valley) is a robust
  default for aberrations up to about 0.3 $\lambda$ RMS.
- Larger aberrations need larger diversity, or three planes ($-d$, 0, $+d$).
- Very small aberrations at high SNR are sensed more precisely with less
  diversity.
- Any known OPD works as diversity, such as astigmatism or DM probes: pass
  `(K, ny, nx)` maps.

The [validation page](../validation.md) reports the capture range of
[`retrieve`][solvephase.retrieve] measured on random aberrations.

## Broadband light, sampling and pixels

```python
import numpy as np
import solvephase as sp

pupil = sp.Pupil.hst(64)
band = np.linspace(1.5e-6, 1.8e-6, 5)  # five wavelengths across H band
model = sp.FocalPlaneModel(
    pupil,
    band,
    48,
    pixel_scale=0.06 / 206265,  # 60 mas pixels: undersampled at 1.5 um
    weights=[0.8, 1.0, 1.0, 0.9, 0.7],  # relative photon flux
    oversample=3,  # integrate each detector pixel
    diversity=sp.zernike_diversity(pupil, 4, [0.0, 0.4e-6]),
)
print(model)
truth = sp.random_aberration(pupil, 60e-9, n_modes=20, start=4, seed=3)
images = sp.simulate_images(model, truth, photons=3e6, seed=4)
result = sp.solve(
    sp.FocalPlaneProblem(model, images, basis=sp.Basis.zernike(pupil, 24), loss="poisson")
)
print(f"{sp.wavefront_error(result.opd, truth, pupil, remove='tiptilt') * 1e9:.2f} nm")
```

## Masks, saturation and detectors

Real frames saturate, have hot pixels and come in ADU. Convert to
photo-electrons, mask unusable pixels with `weights=0`, and pass the read
noise. [`solvephase.interop.expose`](interop.md) does all three for
getframes-simulated cameras.

## Segmented apertures

```python
keck = sp.Pupil.keck(96)
ptt = sp.Basis.segments(keck)  # piston, tip, tilt per segment
print(ptt.n_modes, ptt.labels[:3])
```

Segment pistons larger than half a wave are ambiguous at one wavelength
(modulo $\lambda$). Use broadband data or several wavelengths to resolve
them.

## References

- R. W. Gerchberg & W. O. Saxton, Optik 35, 237 (1972).
- D. L. Misell, J. Phys. D 6, L6 (1973).
- J. R. Fienup, Appl. Opt. 21, 2758 (1982); Appl. Opt. 32, 1737 (1993).
- R. A. Gonsalves, Opt. Eng. 21, 829 (1982).
- R. G. Paxman, T. J. Schulz & J. R. Fienup, JOSA A 9, 1072 (1992).
- A. S. Jurling & J. R. Fienup, JOSA A 31, 1348 (2014).
- B. H. Dean et al., Proc. SPIE 6265 (2006) (JWST Hybrid Diversity Algorithm).
- D. C. Ghiglia & L. A. Romero, JOSA A 11, 107 (1994) (least-squares unwrapping).
