# Quickstart

This page simulates focal-plane images of an aberrated telescope, then
retrieves the aberration. Every block runs as written; together they make
one script.

## 1. Describe the optics

```python
import numpy as np
import solvephase as sp

wavelength = 1.6e-6  # metres
pupil = sp.Pupil.circular(64, 8.0, obscuration=0.14)  # 64 px across an 8 m pupil
print(pupil)
```

A [`Pupil`][solvephase.Pupil] is a sampled amplitude map with its pixel
pitch and diameter. Presets exist for Keck, JWST, the VLT and Hubble, and
for any segmented hexagonal aperture.

## 2. Simulate data

Two images, one in focus and one with 0.4 µm RMS of defocus (the *diversity*
that makes the problem well posed):

```python
truth = sp.random_aberration(pupil, 120e-9, n_modes=30, start=4, seed=1)  # 120 nm RMS
model = sp.FocalPlaneModel(
    pupil,
    wavelength,
    64,
    sampling=2.0,  # 64x64 pixels, Nyquist sampled
    diversity=sp.zernike_diversity(pupil, 4, [0.0, 0.4e-6]),
)
images = sp.simulate_images(model, truth, photons=1e6, background=10, read_noise=3, seed=2)
print(images.shape)  # (2, 64, 64), photo-electrons
```

## 3. Retrieve

```python
result = sp.retrieve(
    images,
    pupil,
    wavelength,
    sampling=2.0,
    diversity=[0.0, 0.4e-6],  # defocus RMS per image, metres
    read_noise=3.0,
)
print(result.summary())
error = sp.wavefront_error(result.opd, truth, pupil, remove="tiptilt")
print(f"wavefront error: {error * 1e9:.2f} nm RMS")
```

`result.opd` is the wavefront in metres over the pupil, and
`result.coefficients` holds the Zernike coefficients (RMS metres, Noll
order from tip). The model images, flux, background and convergence history
are also on the result. [`retrieve`][solvephase.retrieve] used its default
robust strategy:

1. Levenberg-Marquardt on the amplitude metric from a flat start.
2. Gerchberg-Saxton/Misell followed by Levenberg-Marquardt.
3. Keep whichever fits better, then a Poisson maximum-likelihood polish.

## 4. On the GPU

Add `device="gpu"` (needs CuPy). The arrays on the result stay on the GPU;
use `sp.to_numpy` or `result.to_numpy()` to bring them back.

```python
if sp.gpu_available():
    gpu_result = sp.retrieve(
        images,
        pupil,
        wavelength,
        sampling=2.0,
        diversity=[0.0, 0.4e-6],
        read_noise=3.0,
        device="gpu",
    )
    print(gpu_result.device, sp.to_numpy(gpu_result.opd).shape)
```

## 5. Full control

[`retrieve`][solvephase.retrieve] is a thin layer over
[`FocalPlaneProblem`][solvephase.FocalPlaneProblem] and
[`solve`][solvephase.solve]. Use them directly to choose the basis, the
likelihood, the nuisance parameters and the optimizer:

```python
problem = sp.FocalPlaneProblem(
    model,
    images,
    basis=sp.Basis.zernike(pupil, 45),
    loss="poisson",
    read_noise=3.0,
    fit_background=True,
)
lm = sp.solve(problem, method="lm")  # Levenberg-Marquardt
zonal = sp.solve(  # pixel-by-pixel refinement
    sp.FocalPlaneProblem(
        model, images, basis=None, loss="poisson", read_noise=3.0, fit_background=True
    ),
    method="lbfgs",
    start=lm,
    max_iter=100,
)
print(lm.n_iter, zonal.n_iter)
```

## Where next

- [Choosing an algorithm](choosing.md) for the other families: CDI,
  extended-object phase diversity, LIFT, Fast & Furious, Wirtinger flows
  and TIE.
- [Focal-plane retrieval](guide/focal-plane.md) for diversity design,
  broadband light, undersampled detectors and pixel masks.
- [Simulation and interoperability](guide/interop.md) for pyturb
  turbulence and getframes detectors.
