# solvephase

**Fast, GPU-optional phase retrieval for adaptive optics, optical metrology and
coherent imaging.**

Phase retrieval recovers the phase of a light field (a wavefront, or an
object's complex transmission) from intensity measurements alone. solvephase
puts the main algorithm families behind one consistent API. They share their
propagators, likelihoods and optimizers, and the same code runs on NumPy or
on CUDA through CuPy.

```py
import solvephase as sp

pupil = sp.Pupil.vlt(128)  # 8 m, obscured, four spiders
result = sp.retrieve(
    images,
    pupil,
    1.65e-6,  # two images, in focus + defocused
    sampling=2.0,
    diversity=[0.0, 0.5e-6],
    read_noise=3.0,
    device="gpu",
)
result.opd  # retrieved wavefront [m]
result.coefficients  # Zernike coefficients [m RMS]
```

## What is in it

| Family | Algorithms | Typical use |
|---|---|---|
| Focal-plane retrieval | Gerchberg-Saxton, Misell (multi-plane), nonlinear modal/zonal retrieval with Gaussian, Poisson or amplitude likelihoods; Levenberg-Marquardt, L-BFGS, Adam | NCPA calibration, optical testing, telescope phasing |
| Phase diversity | Gonsalves/Paxman reduced metric for unknown extended objects | solar and extended-scene wavefront sensing |
| AO focal-plane sensors | LIFT, Fast & Furious, Cramér-Rao bounds | low-order and closed-loop focal-plane WFS |
| Coherent diffraction imaging | ER, HIO, DM, RAAR, RRR, ASR, HPR, OSS, shrinkwrap, batched multi-start | X-ray/optical CDI, lensless imaging |
| Generic models | Wirtinger flow, TWF, TAF, RAF, L-BFGS; spectral initializations; coded-diffraction and Gaussian operators | compressive / coded phase retrieval, research |
| Transport of intensity | FFT and DCT Poisson solvers, uniform and non-uniform intensity, multi-plane derivative estimation | microscopy, curvature sensing |

## Why solvephase

- **Fast.** Exact analytic gradients and Gauss-Newton Jacobians through every
  propagator (no autodiff tape), batched FFTs and matrix Fourier transforms,
  device-resident optimizers, and a CPU path that avoids thread-contention
  traps. See [Benchmarks](benchmarks.md).
- **Statistically right.** Poisson and read-noise likelihoods, nuisance
  parameters (flux, background, registration), masks for bad or saturated
  pixels, and estimators that reach the Cramér-Rao bound
  ([Validation](validation.md)).
- **Robust by default.** [`retrieve`][solvephase.retrieve] runs several
  capture strategies and keeps the best, so large aberrations converge
  without hand tuning.
- **Physically consistent.** SI units throughout, documented sign and
  sampling conventions, agreement with HCIPy to machine precision.
- **Part of an AO toolchain.** Modal bases come from
  [aobasis](https://github.com/jacotay7/aobasis), realistic aberrations from
  [pyturb](https://github.com/jacotay7/pyturb), detector frames from
  [getframes](https://github.com/jacotay7/getframes).

## Next steps

- [Installation](installation.md)
- [Quickstart](quickstart.md)
- [Choosing an algorithm](choosing.md)
- [Concepts](concepts.md), if you are new to phase retrieval
