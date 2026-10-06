# Focal-plane wavefront sensing for adaptive optics: LIFT and Fast & Furious

Two focal-plane wavefront sensors built for adaptive-optics (AO) loops, where
the sensor runs on every frame and the light is often scarce:

| | LIFT | Fast & Furious (F&F) |
|---|---|---|
| Measures | a few low-order modes (tip, tilt, focus, astigmatism, coma, ...) | the full pupil phase, zonal |
| Data per estimate | one image with a known astigmatism | the current image and the previous one |
| Breaks the even/odd ambiguity with | a static astigmatism | the known DM change between frames |
| Estimator | maximum likelihood (iterated linearization) | closed-form weak-phase solution, two FFTs |
| Phase range | up to a few tenths of a wave RMS | weak phase (well below one radian), closed loop |
| Typical use | low-order sensing on faint stars (LGS tip/tilt/focus) | NCPA and quasi-static speckle control |

Both are in `solvephase.algorithms`: `lift`, `LIFT`, `lift_crlb` in
`solvephase.algorithms.lift`, and `FastAndFurious`, `simulate_closed_loop` in
`solvephase.algorithms.fast_furious`.

## The even/odd ambiguity

Split the pupil phase into parts that are even and odd under a 180-degree
rotation, $\phi = \phi_e + \phi_o$ with $\phi_e(-\mathbf{x}) = \phi_e(\mathbf{x})$ and
$\phi_o(-\mathbf{x}) = -\phi_o(\mathbf{x})$. For a real pupil amplitude $A$ that is
itself centro-symmetric, the fields $A e^{i\phi(\mathbf{x})}$ and
$A e^{-i\phi(-\mathbf{x})}$ are complex conjugates of each other up to a
reflection, so their Fourier transforms differ only by conjugation and the
images are identical:

$$
\left|\mathcal{F}\{A e^{i\phi(\mathbf{x})}\}\right|^2
= \left|\mathcal{F}\{A e^{-i\phi(-\mathbf{x})}\}\right|^2 .
$$

The twin $-\phi(-\mathbf{x}) = -\phi_e + \phi_o$ keeps the odd part and flips the
even part. One in-focus image therefore fixes the odd part (tip, tilt, coma,
trefoil) but not the *sign* of the even part (defocus, astigmatism, spherical).
Something even and known must be added to tell the twins apart: a static
astigmatism (LIFT), a defocused second image (classical phase diversity), or
the DM change between two frames (F&F).

## LIFT

LIFT, the LInearized Focal-plane Technique (Meimon, Fusco & Mugnier 2010;
Plantet et al. 2013), adds a known astigmatism $\phi_d$ to the beam. For the
total phase $\phi + \phi_d$ the twin is $-(\phi + \phi_d)(-\mathbf{x})$, so the
equivalent solution for the unknown part is

$$
\phi' = -\phi_e + \phi_o - 2\phi_d ,
$$

which lies a distance $2\phi_d$ away from the truth in the astigmatism mode. A
local search started from a flat wavefront converges to the true solution as
long as the astigmatism component of the aberration stays above $-\phi_d$.

### Estimator

With modal coefficients $\mathbf{c}$ (Zernike RMS OPD in metres), flux $F$ and
the image model $m(\mathbf{c}) = F\, \mathrm{PSF}(\mathbf{c}) + b$, LIFT
maximizes the likelihood of the image $d$ under photon and read noise. Each
iteration linearizes the model, $m(\mathbf{c} + \delta) \approx m(\mathbf{c}) + J\delta$,
and solves the weighted normal equations

$$
\left(J^{\mathsf T} R^{-1} J\right)\delta = J^{\mathsf T} R^{-1}\,\bigl(d - m(\mathbf{c})\bigr),
\qquad R = \mathrm{diag}\bigl(m + \sigma_\mathrm{ron}^2\bigr),
$$

which is Gauss-Newton on the (shifted) Poisson likelihood. solvephase does
exactly this with `FocalPlaneProblem` + `solve(method="lm")` (damped
Gauss-Newton, Poisson loss, fitted flux); `LIFT` only sets the problem up,
keeps the model for repeated use and adds the twin check below. Use
`loss="gaussian"` for the original form with $R$ estimated from the image.

The default astigmatism is $\lambda/10$ RMS (about half a wave peak-to-valley;
VLT/IRLOS uses 170 nm RMS at 1.6 um). For a 32-pixel pupil, Nyquist
sampling and ten modes, the Cramer-Rao bound is flat between about
$0.05\lambda$ and $0.15\lambda$ RMS, while the fraction of random aberrations
of $0.15\lambda$ RMS recovered from a flat start rises from about 80 % at
$0.05\lambda$ to about 97 % at $0.1\lambda$.

**Twin check.** For a centro-symmetric pupil, a centred image window and a
basis closed under $\phi(\mathbf{x}) \to -\phi(-\mathbf{x})$ (Zernikes are), the
twin $\phi'$ above is exact: it gives *the same* image. After convergence
`LIFT` compares the solution with its twin and reports the smaller wavefront
(`result.extra["twin_swapped"]` says whether it swapped). Disable with
`resolve_twin=False`.

### Cramer-Rao bound

The Fisher information of the coefficients is
$I = J^{\mathsf T}\,\mathrm{diag}\bigl(1/(m + \sigma_\mathrm{ron}^2)\bigr)\,J$, built
from the same forward-mode Jacobian as the solver, with the flux (and the
background, when fitted) as nuisance parameters. `lift_crlb` returns the
coefficient block of $I^{-1}$ in m$^2$: the smallest covariance any unbiased
estimator can reach. LIFT reaches it at high flux: over 300 noise
realizations of a $0.1\lambda$ RMS ten-mode aberration (24-pixel pupil,
$10^5$ and $10^6$ photons, 1 e$^-$ read noise), the variance-to-bound ratios
average 0.98-1.0, all within 0.8-1.2 (`tests/test_lift.py`).

### Example

```python
import numpy as np
import solvephase as sp
from solvephase.algorithms.lift import LIFT, lift_crlb

wavelength = 1.6e-6
pupil = sp.Pupil.circular(32)
sensor = LIFT(pupil, wavelength, 32, sampling=2, n_modes=10, read_noise=2.0)

rng = np.random.default_rng(0)
truth = rng.normal(size=10)
truth *= 0.1 * wavelength / np.linalg.norm(truth)  # 0.1 wave RMS, Noll 2..11

photons = 1e4
image = sp.to_numpy(sensor.expected_image(truth, photons=photons))
image = rng.poisson(image) + rng.normal(0.0, 2.0, image.shape)

result = sensor.estimate(image)
crlb = np.sqrt(
    np.diag(lift_crlb(pupil, wavelength, 32, truth, photons=photons, read_noise=2.0, sampling=2))
)
print(np.round((result.coefficients - truth) * 1e9, 1))  # error, nm
print(np.round(crlb * 1e9, 1))  # bound, nm
```

In a loop, keep one `LIFT` (the model and the basis stay on the device) and
warm-start each estimate from the previous one:

```python
result = None
for frame in range(3):
    image = rng.poisson(sp.to_numpy(sensor.expected_image(truth, photons=photons)))
    result = sensor.estimate(image, start=result)
```

The one-call form is `lift(image, pupil, wavelength, sampling=2, n_modes=10)`.

## Fast & Furious

F&F (Keller et al. 2012; Korkiakoski et al. 2014; on sky, Bos et al. 2020)
is a sequential phase-diversity method built on the weak-phase solution of
Gonsalves (2001). It is fast enough for every frame of an AO loop and
estimates the phase zonally, so it can drive a high-order DM.

### Weak-phase solution

To first order in $\phi$, with
$a = \mathcal{F}\{A\}$ (real, even), $v = \mathcal{F}\{A\phi_e\}$ (real, even)
and $i y = \mathcal{F}\{A\phi_o\}$ ($y$ real, odd), the focal field is
$E \approx a + iv - y$ and the image

$$
p = a^2 + v^2 + y^2 - 2ay .
$$

Its odd part gives the odd phase directly and its even part the magnitude of
the even field:

$$
p_o = -2ay \;\Rightarrow\; y = \frac{-a\,p_o}{2a^2 + \epsilon},
\qquad
p_e = a^2 + v^2 + y^2 \;\Rightarrow\; |v| = \left|p_e - a^2 - y^2\right|^{1/2}.
$$

The sign of $v$ comes from the previous image. If that frame had the phase
$\phi + \phi_d$, where $\phi_d$ is minus the known DM change applied since,
its even part is $p_{e,2} = a^2 + (v + v_d)^2 + (y + y_d)^2$, and

$$
v_s = \frac{p_{e,2} - p_e - v_d^2 - y_d^2 - 2 y y_d}{2 v_d},
\qquad v = \operatorname{sign}(v_s)\,|v| .
$$

Following Korkiakoski et al. (2014), $v_s$ is used for the sign only (the
two-image difference is noisy); the images are normalized to the energy of
$a^2$ over the detector, a scaled $a^2$ is added so the image peak matches the
unaberrated peak (a first-order model that accounts for the Strehl loss),
and the field $v + iy$ is multiplied by a concave parabola before the inverse
FFT to damp noisy high spatial frequencies. Then

$$
A\phi = \mathcal{F}^{-1}\{w\,(v + iy)\}, \qquad \phi = A\phi / A ,
$$

and the DM is updated with a leaky integrator,
$\theta_k = g_l\,\theta_{k-1} - g\,\phi_k$.

The first frame has no predecessor. With `first_even=True` (the default) it
uses positive signs for $v$: a valid even guess whose correction supplies the
even DM change the next frame needs. An odd-only first correction would leave
$v_d = 0$ and the signs could never be resolved.

### Requirements

* A centro-symmetric pupil amplitude (circular, annular, symmetric spiders).
  Arbitrary pupils need F&F-GS (Korkiakoski et al. 2014), not implemented.
* The detector must be at least Nyquist-sampled and
  $\lambda / (\text{pupil pitch} \times \text{pixel scale})$ must be an integer
  FFT size, e.g. `sampling=2` with a pupil that fills its grid.
* Background-subtracted images; the optical axis at the window centre.
* Weak residual phase. Starting at 0.1 wave RMS (0.63 rad) works; much larger
  aberrations should first be reduced with a robust method (phase diversity,
  LIFT, Gerchberg-Saxton).

`epsilon` (default $10^{-4}$, in units of the unaberrated peak) regularizes
the division by $a$; Korkiakoski et al. recommend 50-500 times the per-pixel
noise of the normalized image. Smaller values recover more of the faint-ring
signal (higher spatial frequencies); larger values reject noise.

### Closed-loop example

`step(image, dm_change_opd)` returns the estimated OPD of the wavefront in the
image; apply `-gain` times it with the DM, and pass the resulting wavefront
change with the next image:

```python
import numpy as np
import solvephase as sp
from solvephase.algorithms.fast_furious import FastAndFurious, simulate_closed_loop

wavelength = 1.6e-6
pupil = sp.Pupil.circular(64, obscuration=0.1)
basis = sp.Basis.zernike(pupil, 20)
ncpa = basis.synthesize(np.random.default_rng(1).normal(size=20))
ncpa *= 0.1 * wavelength / sp.rms(ncpa, pupil)  # 0.1 wave RMS static NCPA

ff = FastAndFurious(pupil, wavelength, 64, sampling=2)
camera = sp.FocalPlaneModel(pupil, wavelength, 64, sampling=2)

dm = np.zeros(pupil.shape)
change = None
for k in range(20):
    image = sp.to_numpy(camera.images(ncpa + dm)[0])
    estimate = sp.to_numpy(ff.step(image, change))
    new_dm = dm - 0.5 * estimate  # gain 0.5, no leak
    change, dm = new_dm - dm, new_dm
print(sp.rms(ncpa + dm, pupil) / sp.rms(ncpa, pupil))  # about 0.1
```

`simulate_closed_loop` runs the same loop on the sensor's device, with
optional photon and read noise, leak, a modal DM (`basis=`) and a stopping
tolerance, and records the residual RMS and per-step latency:

```python
loop = simulate_closed_loop(ff, ncpa, 20, gain=0.5, photons=1e6, read_noise=1.0, seed=0)
print(loop.residual_rms / loop.residual_rms[0])
print(loop.strehl[-1], np.median(loop.step_times) * 1e3, "ms")
```

From 0.1 wave RMS of 10-20 Zernike modes the loop reaches about 10 % of the
initial RMS in 20 iterations without noise and 11-15 % with $10^6$ photons
per frame (gain 0.5). The residual floor comes from the parts of the field
where $a \approx 0$ (dark Airy rings), where the regularized division
discards the signal, and from the pupil edge.

Each step costs one forward FFT (of the DM change) and one inverse FFT on the
$N \times N$ grid, $N = $ sampling $\times$ pupil pixels, plus a few
element-wise operations, all on the backend.

## Which method when

* **LIFT** for a few low-order modes at very low flux, when one image per
  frame must suffice and the astigmatism can live permanently in the sensor
  arm. It is maximum likelihood and reaches the Cramer-Rao bound.
* **F&F** for many degrees of freedom in a closed loop with a DM whose
  changes are known: NCPA calibration, quasi-static speckle suppression. It
  needs no extra optics and almost no computation, but only works for weak
  phases and centro-symmetric pupils, and converges over several frames.
* **Full phase diversity** (`FocalPlaneProblem` with two or more diversity
  channels and `solve`) when accuracy matters more than speed, the
  aberration is large, the pupil is arbitrary, or the light is broadband or
  the detector undersampled.

## References

* S. Meimon, T. Fusco & L. M. Mugnier, "LIFT: a focal-plane wavefront sensor
  for real-time low-order sensing on faint sources", Opt. Lett. 35, 3036
  (2010).
* C. Plantet, S. Meimon, J.-M. Conan & T. Fusco, "Experimental validation of
  LIFT for estimation of low-order modes in low-flux wavefront sensing", Opt.
  Express 21, 16337 (2013).
* A. Kuznetsov, S. Oberti, B. Neichel & T. Fusco, "Striving towards robust
  phase diversity on-sky: implementing LIFT for VLT/MUSE-NFM", A&A 687, A221
  (2024).
* R. A. Gonsalves, "Small-phase solution to the phase-retrieval problem",
  Opt. Lett. 26, 684 (2001).
* C. U. Keller, V. Korkiakoski, N. Doelman, R. Fraanje, R. Andrei &
  M. Verhaegen, "Extremely fast focal-plane wavefront sensing for extreme
  adaptive optics", Proc. SPIE 8447, 844721 (2012).
* V. Korkiakoski, C. U. Keller, N. Doelman, M. Kenworthy, G. Otten &
  M. Verhaegen, "Fast & Furious focal-plane wavefront sensing", Appl. Opt.
  53, 4565 (2014).
* S. P. Bos, S. Vievard, M. J. Wilby et al., "On-sky verification of Fast and
  Furious focal-plane wavefront sensing", A&A 639, A52 (2020).
