# Phase diversity with an extended object

Phase diversity measures a wavefront from two or more images of the *same*
scene, each recorded with a different, known extra aberration (usually a
defocus). The scene does not need to be a star: granulation on the Sun, an
extended target on the ground, a resolved satellite or a planet all work,
because the object is estimated together with the wavefront and then
eliminated from the problem (Gonsalves 1982; Paxman, Schulz & Fienup 1992).

```python
from solvephase.algorithms.phase_diversity import PhaseDiversityProblem, phase_diversity
```

For a known point source, where the object is a delta function, use
`FocalPlaneProblem` instead: it can use a Poisson likelihood and fit flux and
background. With a point source both approaches give the same wavefront.

## The model

Channel $k$ records the scene $o$ blurred by the point-spread function $s_k$
of the unknown wavefront $\phi$ plus the known diversity $\theta_k$, with
noise $n_k$:

$$
d_k = o \ast s_k + n_k, \qquad
s_k = \left|\mathcal{F}\{A\, e^{i(\phi + \theta_k)}\}\right|^2 .
$$

In Fourier space the convolution is a product. With
$D_k = \mathcal{F}\{d_k\}$, $O = \mathcal{F}\{o\}$ and the optical transfer
functions $S_k = \mathcal{F}\{s_k\}$,

$$
D_k(\mathbf{f}) = O(\mathbf{f})\, S_k(\mathbf{f}) + N_k(\mathbf{f}).
$$

`solvephase` computes $s_k$ with a
`FocalPlaneModel` (one channel per image, broadband and pixel
integration included) whose `image_shape` is the data shape. Its PSFs are
normalized to sum to 1, so $S_k(0) = 1$ and the object comes out in data units
(photons per pixel, say).

## Eliminating the object: the reduced metric

For white Gaussian noise the joint maximum-likelihood estimate of $o$ and
$\phi$ minimizes

$$
L(o, \phi) = \sum_{\mathbf{f}} \left[ \sum_k \left| D_k - O\, S_k \right|^2
+ \gamma\, |O|^2 \right].
$$

For fixed $\phi$ this is quadratic in $O$ at every frequency separately, and
its minimizer is the multi-frame Wiener filter

$$
\hat{O}(\mathbf{f}) = \frac{\sum_k D_k\, S_k^*}{\sum_k |S_k|^2 + \gamma}.
$$

Substituting $\hat O$ back leaves a function of the wavefront alone, the
*reduced* (or object-independent) metric of Gonsalves (1982) and Paxman et al.
(1992):

$$
L(\phi) = \sum_{\mathbf{f}} \left[ \sum_k |D_k|^2 -
\frac{\left| \sum_k D_k\, S_k^* \right|^2}{\sum_k |S_k|^2 + \gamma} \right].
$$

With two channels it takes the familiar form
$\sum_{\mathbf{f}} |D_1 S_2 - D_2 S_1|^2 / (|S_1|^2 + |S_2|^2 + \gamma)$ (for
$\gamma \to 0$): the error metric vanishes when the two images are consistent
with one object.

`PhaseDiversityProblem` evaluates $L$ in the first form,
$\sum_k |D_k - \hat O S_k|^2 + \gamma |\hat O|^2$, a sum of non-negative terms
that keeps its accuracy in single precision on a GPU, and divides it by the
noise power per frequency so that it reads as a chi-square: about
$(K - 1)$ times the number of frequencies used at the solution.

### Gradient

Because $\hat O$ minimizes $L(o, \phi)$, its own variation drops out of the
derivative (the envelope theorem), and

$$
\frac{\partial L}{\partial S_k^*} = -\hat{O}^* \left( D_k - \hat{O}\, S_k \right).
$$

`solvephase` back-propagates this through the real FFT to the PSFs,
$\partial L / \partial s_k = 2 N\, \mathrm{irfft2}\{ -\hat O^* (D_k - \hat O S_k) \}$
on the frequency mask ($N$ pixels), then through
`FocalPlaneModel.vjp` to the pupil phase, and projects onto the modal basis.
The parameters are those of `FocalPlaneProblem`: modal coefficients in
radians at the reference wavelength internally, reported as RMS OPD in metres.
The gradient is exact (tested against finite differences) and
`phase_diversity` minimizes $L$ with L-BFGS. One evaluation costs one forward
and one adjoint propagation per channel plus a few real FFTs.

## Regularization

$\gamma$ is the inverse signal-to-noise power ratio of the Wiener filter. Two
values are used:

- **In the metric** (`regularization`), $\gamma$ only needs to keep the
  division finite where every transfer function vanishes (near the
  diffraction cutoff). A larger value biases the estimate towards sharper
  PSFs, because it penalizes the object power that blurred channels need.
  `"auto"` sets $\gamma = \sigma_N^2 / \max_{\mathbf f} \overline{|D|^2}$,
  the inverse of the best per-frequency signal-to-noise power ratio, which is
  tiny for good data.
- **For the reported object** (`object_regularization`), the Wiener filter
  needs real damping or it amplifies noise wherever the OTFs are small.
  `"auto"` uses the noise power over the *mean* in-band data power.

The noise power per frequency $\sigma_N^2$ is measured from the data
themselves, as the mean power at spatial frequencies beyond the diffraction
cutoff, where a Nyquist-sampled image contains only noise (Löfdahl & Scharmer
1994). It is reported in `result.extra["noise_power"]`. Undersampled data
have no such frequencies: give both regularizations as numbers.

## Frequency mask

An incoherent imaging system transmits nothing beyond the cutoff
$f_c = D/\lambda$ (cycles per radian), which is
$f_c\,p$ cycles per pixel for a pixel scale $p$ (0.5 at Nyquist sampling).
Beyond it $S_k = 0$ and the data are pure noise, so
`frequency_mask="diffraction"` sums the metric only inside $D/\lambda_{\min}$
(the shortest wavelength of a broadband model; the pupil's widest baseline
sets $D$). A float restricts it further to that fraction of the cutoff, and
`None` uses every frequency. The zero frequency, the total flux, carries no
wavefront information and is always excluded. Sample at Nyquist or better at
the shortest wavelength; the problem warns otherwise.

## Apodization and field edges

The Fourier-domain model is a *periodic* convolution, but a real detector
sees a window cut out of a larger scene: light from outside the field blurs
in, and the field edges are discontinuous. `PhaseDiversityProblem` therefore
apodizes each image before the FFT (Löfdahl & Scharmer 1994):

$$
d_k^{w} = \ell_k + w\,(d_k - \ell_k),
$$

where $w$ is the window and $\ell_k$ the image's level at the field edge
(its mean weighted by $1 - w$). The frame becomes periodic, a uniform
background passes untouched (a constant convolves to itself), and only the
scene's structure meets the taper. The default `window="tukey"` has a flat
centre and cosine edges over `window_alpha = 0.25` of each axis;
`"hann"` tapers the whole field, and `None` skips apodization.

Apodization is an approximation, $w\,(o \ast s_k) \ne (w\,o) \ast s_k$, and
the error grows with the curvature of the window times the second moment of
the PSF. A 1-wave defocused PSF is about 16 pixels wide at Nyquist sampling,
so for a scene that fills the field the edge error can exceed the photon
noise at high signal-to-noise ratio and bias the wavefront. In practice:

- **Compact scenes** (a satellite, a planet, a star cluster on a dark sky)
  whose blurred images stay inside the flat part of the window are modelled
  exactly; this is where the method is most accurate.
- **Field-filling scenes** (solar granulation, ground scenes) need fields
  much wider than the defocused PSF (128 pixels or more), wide tapers, and
  ideally a smaller diversity; check the result on simulations matched to
  your data. `result.extra["apodized_images"]` and `result.model_images`
  show what was fitted.

## Choosing the diversity

The classic choice is about **one wave peak-to-valley of defocus** in one
channel and an in-focus image in the other (Paxman et al. 1992; Löfdahl &
Scharmer 1994). Smaller defocus loses sensitivity to even modes, larger
defocus spreads the light and lowers the per-pixel SNR (and worsens edge
effects). For unit-RMS Noll Z4, 1 wave PV is
$\lambda / (2\sqrt 3) \approx 0.29\,\lambda$ RMS:

```py
div = zernike_diversity(pupil, 4, [0.0, wavelength / (2 * math.sqrt(3))])
```

Other practical points:

- **Tip and tilt.** A common tip/tilt only moves the unknown object and is
  not measurable, so `basis=n` builds `Basis.zernike(pupil, n, start=4)`
  (defocus upwards); a basis that includes tip/tilt leaves them at their
  starting values. Compare results with `remove="tiptilt"`.
- **Registration.** The images must be co-registered. `fit_tilt=True` fits a
  tip/tilt per channel relative to channel 0 (`result.tilts`).
- **Exposure.** `equalize_flux=True` (default) scales every image to the
  mean total flux, so unequal exposures do not bias the metric.
- **Large aberrations.** `coarse_to_fine=(5, 10)` solves with the first 5,
  then 10, then all modes, widening the capture range.
- **More channels.** Any number $K \ge 2$ of diversities works; the metric
  and the object estimate use all of them.
- **Solar and extended-scene wavefront sensing.** Split a large field into
  subfields (isoplanatic patches, typically 64-128 pixels), run
  `phase_diversity` on each, and average or map the wavefronts; this is the
  standard approach for solar telescopes (Löfdahl & Scharmer 1994) and for
  calibrating non-common-path aberrations on extended sources.

## Example

Two images (in focus and 1 wave PV defocus) of a compact random scene with
1000 photons per pixel; 20 Zernike modes, 0.075 waves RMS of aberration.

```python
import math

import numpy as np

from solvephase import Basis, FocalPlaneModel, Pupil, rms, wavefront_error, zernike_diversity
from solvephase.algorithms.phase_diversity import phase_diversity

wl = 1.6e-6
rng = np.random.default_rng(1)
pupil = Pupil.circular(64)

# Truth: 0.075 waves RMS over Zernike modes Z4-Z23.
basis = Basis.zernike(pupil, 20, start=4)
coeffs = rng.standard_normal(20) / np.arange(4, 24) ** 0.8
coeffs *= 0.075 * wl / np.linalg.norm(coeffs)
opd = basis.synthesize(coeffs)

# Channel 0 in focus, channel 1 with 1 wave peak-to-valley of defocus
# (unit-RMS Z4 has a peak-to-valley of 2 sqrt(3)).
div = zernike_diversity(pupil, 4, [0.0, wl / (2 * math.sqrt(3))])
model = FocalPlaneModel(pupil, wl, 64, sampling=2.0, diversity=div)

# A compact scene: Gaussian blobs within 12 pixels of the centre.
y, x = np.mgrid[:64, :64] - 31.5
scene = np.full((64, 64), 1e-3)
for _ in range(8):
    cy, cx = rng.uniform(-12, 12, 2)
    width = rng.uniform(1.0, 3.0)
    scene += rng.uniform(0.3, 1.0) * np.exp(-0.5 * ((y - cy) ** 2 + (x - cx) ** 2) / width**2)

# Images: scene * PSF (a periodic convolution is exact here because the
# blurred scene stays inside the field), 1000 photons per pixel, Poisson noise.
psf = np.asarray(model.images(opd))
kernel = np.fft.rfft2(np.fft.ifftshift(psf, axes=(-2, -1)))
clean = np.fft.irfft2(np.fft.rfft2(scene) * kernel, s=(64, 64))
images = rng.poisson(clean * 1000 / clean[0].mean()).astype(float)

result = phase_diversity(model, images, basis=basis)
print(result.summary())
err = wavefront_error(result.opd, opd, pupil, remove="tiptilt")
print(f"error {err / wl:.4f} waves RMS of {rms(opd, pupil, 'tiptilt') / wl:.4f}")
obj = result.extra["object"]  # Wiener estimate of the scene, in photons per pixel
```

This recovers the wavefront to about 0.003 waves RMS (4% of the aberration)
in about 50 L-BFGS iterations and 0.3 s on a CPU. On a GPU pass
`device="gpu"` to the model; the result stays on the GPU until
`result.to_numpy()`.

### What the result holds

| Field | Content |
|---|---|
| `opd`, `phase`, `coefficients` | the wavefront (metres, radians at the reference wavelength, RMS metres per mode) |
| `extra["object"]` | Wiener object estimate on the data grid, in data units, registered with the optical axis, band-limited to the frequency mask, apodized |
| `extra["otf"]` | the transfer functions $S_k$, real-FFT layout `(K, my, mx // 2 + 1)` |
| `extra["apodized_images"]`, `model_images` | the apodized data and their model $\hat O S_k$ |
| `extra["regularization"]`, `extra["object_regularization"]`, `extra["noise_power"]` | the values used |
| `extra["window"]`, `extra["frequency_mask"]` | the apodization window and the frequency mask |
| `tilts` | per-channel registration tip/tilt (with `fit_tilt=True`) |

## References

- R. A. Gonsalves, "Phase retrieval and diversity in adaptive optics",
  *Opt. Eng.* **21**, 829 (1982).
- R. G. Paxman, T. J. Schulz & J. R. Fienup, "Joint estimation of object and
  aberrations by using phase diversity", *J. Opt. Soc. Am. A* **9**, 1072
  (1992).
- M. G. Löfdahl & G. B. Scharmer, "Wavefront sensing and image restoration
  from focused and defocused solar images", *Astron. Astrophys. Suppl.* **107**,
  243 (1994).
- L. M. Mugnier, A. Blanc & J. Idier, "Phase diversity: a technique for
  wave-front sensing and for diffraction-limited imaging", *Adv. Imaging
  Electron Phys.* **141**, 1 (2006).
