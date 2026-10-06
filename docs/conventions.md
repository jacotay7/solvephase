# Conventions and units

These are stable contracts; every one of them is covered by a test.

## Units

| Quantity | Unit |
|---|---|
| OPD (the user-facing wavefront) | metres |
| Phase | radians at a stated wavelength (`result.wavelength`) |
| Modal coefficients | RMS OPD in metres (modes are unit-RMS over the pupil) |
| Wavelength, pupil diameter, pixel pitch | metres |
| Detector pixel scale | radians, or `sampling` = pixels per $\lambda/D$ |
| Images | any linear unit; photo-electrons for the Poisson likelihood |

Internally, solvers work in radians at the reference wavelength (the
spectrally weighted mean wavelength unless you pass
`reference_wavelength`).

## Arrays and coordinates

- Arrays are indexed `(y, x)`.
- Coordinates sit on pixel centres and are centred: pixel `i` of `n` is at
  $(i - (n-1)/2)\,\Delta$. For even `n`, the optical axis lies between the
  two central pixels. `offset=(dy, dx)` (in detector pixels) shifts the
  detector window: pixel `j` samples angle $(j - (n-1)/2 + \text{offset})\,\theta$,
  so the axis falls on pixel $(n-1)/2 - \text{offset}$. `offset=-0.5` puts it
  on pixel `n/2`, as HCIPy and `numpy.fft.fftshift` do.
- Pupil presets are symmetric about the optical axis.

## Sign of the propagation

The Fraunhofer kernel is $\exp(-2\pi i\, \mathbf{x}\cdot\boldsymbol{\alpha}/\lambda)$.
A pupil OPD ramp $\mathrm{OPD} = a\,x$ moves the image to angle $+a$, towards
larger column index. Zernike tip (Noll 2) is a $+x$ ramp and tilt (Noll 3) a
$+y$ ramp. The image is not flipped or transposed relative to HCIPy.

## Normalization

- Propagators are unitary: when the output window captures all the light,
  $\sum |E|^2 = \sum |u|^2$.
- [`FocalPlaneModel`][solvephase.FocalPlaneModel] images sum to 1 per
  channel for an unbounded detector. A finite window loses light and is
  never renormalized.
- The `flux` fitted by a solver is the total flux of the whole PSF, in data
  units.

## Devices and precision

- `device="cpu"` uses NumPy and multithreaded SciPy FFTs. `device="gpu"`
  uses CuPy. `device="auto"` picks the GPU when one is available.
- Precision defaults to double on CPU and single on GPU. Data terms and
  Gauss-Newton normal equations are always accumulated in double, so
  single-precision solves reach the same noise floor.
- Arrays returned from GPU solves stay on the GPU.
  [`to_numpy`][solvephase.to_numpy] is the explicit host boundary.
- Random numbers (initial guesses, simulated noise) are drawn on the host
  from a seed and moved to the device, so a seed gives the same start on
  every device.
