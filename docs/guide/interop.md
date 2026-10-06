# Simulation and interoperability

solvephase is one part of an adaptive-optics simulation family and reuses its
siblings rather than copying them:

| Package | Role here | Install |
|---|---|---|
| [aobasis](https://github.com/jacotay7/aobasis) | Zernike, KL, Fourier and DM bases | core dependency |
| [pyturb](https://github.com/jacotay7/pyturb) | atmospheric OPD (von Kármán phase screens) | `solvephase[interop]` |
| [getframes](https://github.com/jacotay7/getframes) | detector frames: shot/read/dark noise, gain, bias, saturation | `solvephase[interop]` |
| [makewfs](https://github.com/jacotay7/makewfs) | wavefront-sensor images (shares solvephase's `(y, x)`, metres and centred-pixel conventions) | separately |
| [HCIPy](https://hcipy.org) | independent optics reference used by the validation suite | `solvephase[validation]` |

## Built-in simulation

```python
import solvephase as sp

pupil = sp.Pupil.circular(64, 8.0, obscuration=0.14)
model = sp.FocalPlaneModel(
    pupil, 1.6e-6, 64, sampling=2.0, diversity=sp.zernike_diversity(pupil, 4, [0.0, 0.4e-6])
)
truth = sp.random_aberration(pupil, 80e-9, n_modes=30, seed=1)
images = sp.simulate_images(model, truth, photons=1e6, background=5, read_noise=2, seed=2)
```

## Turbulence from pyturb

```python
from solvephase.interop import turbulence_opd

residual = turbulence_opd(pupil, r0=0.8, outer_scale=25.0, seed=3)  # OPD [m] over the pupil
print(f"{sp.rms(residual, pupil) * 1e9:.0f} nm RMS")
```

## Detector frames from getframes

[`expose`][solvephase.interop.expose] exposes normalized model images on a
getframes camera preset (or your own `getframes.Camera`). It returns the raw
ADU frames, frames converted back to photo-electrons, the read noise in
electrons, and weights that mask saturated pixels. These are exactly what the
Poisson likelihood needs.

```python
from solvephase.interop import expose

frames = expose(
    sp.to_numpy(model.images(truth)), "andor_ikon_m934", photons=3e5, background=20, seed=4
)
result = sp.retrieve(
    frames.electrons,
    pupil,
    1.6e-6,
    sampling=2.0,
    diversity=[0.0, 0.4e-6],
    read_noise=frames.read_noise,
    weights=frames.weights,
)
print(
    frames.n_saturated,
    f"{sp.wavefront_error(result.opd, truth, pupil, remove='tiptilt') * 1e9:.1f} nm",
)
```

!!! warning "Saturation"
    Bright PSF cores easily exceed a detector's full well or ADC range. A
    saturated pixel left in the fit biases the wavefront by tens of percent.
    Always pass `weights=frames.weights` (or your own mask).

## API

::: solvephase.interop.turbulence_opd

::: solvephase.interop.expose

::: solvephase.interop.DetectorImages
