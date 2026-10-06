# Transport of intensity (TIE)

The transport-of-intensity equation recovers a phase map from a few intensity
images taken at small, known defocus distances. It is non-iterative (or a
single linear solve), needs no reference beam, and works on extended,
partially coherent illumination, which makes it a standard tool in
quantitative phase microscopy, X-ray and electron imaging, optical metrology,
and, in its curvature-sensing form, adaptive optics.

```python
from solvephase.algorithms.tie import TIEResult, simulate_defocus_stack, tie
```

## The equation

For a paraxial, monochromatic field $u = \sqrt{I}\, e^{i\phi}$ propagating
along $z$ (wavenumber $k = 2\pi/\lambda$), the imaginary part of the paraxial
wave equation $2ik\,\partial_z u + \nabla_\perp^2 u = 0$ is an energy
conservation law (Teague 1983):

$$
-k\,\frac{\partial I}{\partial z} = \nabla_\perp \cdot \left( I\, \nabla_\perp \phi \right).
$$

The intensity flows along the phase gradient: a phase bump (a converging
lens) concentrates light downstream. Given $I$ and $\partial_z I$ at one plane
($z = 0$) this is an elliptic equation for $\phi$, unique up to a constant
(piston) once boundary conditions are fixed. `tie` returns $\phi$ in radians at
$z = 0$ and the optical path difference $\mathrm{OPD} = \phi\,\lambda / 2\pi$
in metres, with piston removed.

For a fixed OPD the intensity derivative does not depend on the wavelength
($\partial_z I = -\nabla\cdot(I\nabla\,\mathrm{OPD})$), so the recovered OPD is
achromatic for non-dispersive objects; the phase in radians scales as
$1/\lambda$.

### Assumptions and caveats

- **Paraxial, scalar.** Phase gradients must be small angles. The simulator
  `simulate_defocus_stack` uses the exact angular spectrum by default; the
  TIE itself is the paraxial limit.
- **Weak defocus.** $\partial_z I$ is estimated by finite differences, which
  assumes intensity varies smoothly (polynomially) over the defocus range.
  Planes beyond the first caustic (the local focal length
  $k / |\nabla^2 \phi|$) break this completely.
- **Phase vortices.** The TIE assumes a continuous phase; it cannot represent
  phase singularities, and Teague's construction below also assumes the
  "transverse flux" $I\nabla\phi$ is curl free.
- **Partial coherence.** With partially coherent light the TIE still holds
  for a suitably generalized phase (Paganin & Nugent 1998). In practice a
  finite illumination aperture low-pass filters $\partial_z I$ (high spatial
  frequencies decay faster with defocus), and broadband light uses an
  effective wavelength; see the review by Zuo et al. (2020).
- **Unobservable boundary flux.** Light crossing the edge of the field of
  view carries information the images do not contain. A phase whose slope
  crosses the window edge (or a global tilt over uniform illumination) is not
  recoverable without boundary data; keep the object inside the field, or use
  an illuminated aperture whose edge is imaged (see
  [curvature sensing](#curvature-sensing)).

## Estimating the axial derivative

`distances` gives each image's defocus in metres relative to the plane where
the phase is wanted (positive downstream), and may include $z = 0$.

| `derivative` | Planes used | Accuracy |
| --- | --- | --- |
| `"central"` | nearest plane on each side of $z = 0$ (plus $z = 0$ if measured) | $O(\Delta z^2)$; exact central difference $(I_+ - I_-)/2\Delta z$ for a symmetric pair, three-point Lagrange slope otherwise |
| `"polyfit"` | all planes; least-squares polynomial of degree `order` (default $\min(N-1, 3)$) | with `order = N - 1` the $N$-point stencil, error $O(\Delta z^{N-1})$; with fewer coefficients than planes it also averages noise |

The polynomial fit is a Savitzky–Golay filter along $z$: the slope at $z = 0$
is a fixed linear combination of the planes, applied pixel by pixel. With
five planes at $0, \pm\Delta z, \pm 2\Delta z$ the default cubic fit is the
fourth-order stencil, far more accurate than the central difference at large
defocus (Waller et al. 2010); in the test suite it is 60 times more accurate
at $\Delta z = 8$ mm. With noisy data, a low-order fit over many planes
reduces the slope noise: over $z = d\,(-3 \dots 3)$ by $\sqrt{14}$ compared
with $\pm d$.

The in-focus intensity $I_0$ is the measured $z = 0$ plane when present,
otherwise the interpolant (or fit) at $z = 0$; pass `in_focus=` to override.
Use `derivative="polyfit"` for a one-sided stack (for example $z = 0, \Delta z,
2\Delta z$).

## Solvers

| `method` | Boundary | Non-uniform intensity | Cost |
| --- | --- | --- | --- |
| `"dct"` (default) | Neumann: zero normal slope at the window edge | Teague's auxiliary function | a few DCT/DST |
| `"fft"` | periodic | Teague's auxiliary function | a few real FFTs |
| `"pcg"` | Neumann on the boundary of `mask` | exact, by conjugate gradients | ~100–300 iterations of a DCT preconditioner |

**Uniform-intensity approximation** (`uniform=True`). With $I \approx
\bar I_0$ (the mean over the mask) the TIE is a Poisson equation,

$$
\nabla^2 \phi = -\frac{k}{\bar I_0}\,\frac{\partial I}{\partial z},
$$

solved with one forward and one inverse transform. It is accurate for weak
amplitude modulation and fails when the intensity varies (relative error 0.78
in the test with a 30–100 % intensity variation).

**Teague's auxiliary function** (default for `"fft"` and `"dct"`). Write
$I\nabla\phi = \nabla\psi$; then $\nabla^2\psi = -k\,\partial_z I$ and

$$
\phi = \nabla^{-2}\, \nabla \cdot \left( \frac{\nabla \psi}{I} \right),
$$

two Poisson solves and a division by the intensity, clipped below
`intensity_floor` $\times \max I_0$ (Teague 1983; Gureyev & Nugent 1996). The
neglected curl of $I\nabla\phi$ limits accuracy when the intensity gradients
are not aligned with the phase gradients (relative error ~0.03 in the test).

**Exact masked solve** (`"pcg"`). The discrete equation
$D^\top \mathrm{diag}(I_\mathrm{edge})\, D\, \phi = k\,\partial_z I$ (forward
differences $D$, edge intensity the mean of its two pixels) is solved inside
`mask` by conjugate gradients, preconditioned with the uniform Neumann
Poisson solver (a DCT, as in Zuo et al. 2014). There is no curl-free assumption, and edges leaving the mask
carry no flux, so the boundary of an illuminated aperture is handled
naturally. The default mask is $I_0 > $ `intensity_floor` $\times \max I_0$;
weights are clipped at the same floor, which bounds the condition number by
`1 / intensity_floor`. The mean of the right-hand side is removed per
connected region (the net flux out of the support is not observable), and the
phase piston is removed per region. Its five-point discretization is
second-order accurate (relative error ~1.4e-3 on smooth phase sampled at
~10 pixels per feature, versus ~1e-4 for the spectral solvers).

### Boundary conditions

The FFT solver assumes the field repeats across the window; any phase or
intensity that does not wrap smoothly couples opposite edges. The DCT solver
(Zuo et al. 2014)
expands everything in the cosine series of the mirror-extended field, so it
assumes only zero normal slope at the window edge. For a phase step across the
window the test suite measures a relative error of 2.7e-3 with `"dct"` and
0.66 with `"fft"`. Both spectral solvers differentiate exactly (DCT-II to
DST-II for gradients, back for the divergence), so on interior objects they
agree to round-off.

### Regularization

The inverse Laplacian amplifies low spatial frequencies, so noise in
$\partial_z I$ appears as smooth "cloud" artefacts. `"fft"` and `"dct"` replace
$1/\lambda_q$ (eigenvalues of $-\nabla^2$) by the Tikhonov filter
$\lambda_q / (\lambda_q^2 + \alpha)$ in the first Poisson solve, with
$\alpha$ set by the dimensionless `regularization` $r$ so that the lowest
non-zero frequency of the grid is attenuated by exactly $1/(1 + r)$. The
default $10^{-6}$ is negligible; values $10^{-3}$–$10^{-1}$ trade low-order
accuracy for noise suppression. `"pcg"` is not regularized; use `tol` or the
mask to control it.

## Choosing the defocus distance

The defocus distance trades nonlinearity against noise:

- the finite-difference truncation error grows as $\Delta z^2$ (central
  difference) — halving $\Delta z$ cuts the error four times in the
  noise-free tests;
- the noise in $\partial_z I$ grows as $\sigma_I / \Delta z$ and is then
  amplified at low frequencies by the inverse Laplacian.

So there is an optimum: in the test suite with $10^{-3}$ intensity noise,
$\Delta z = 4$ mm is seven times more accurate than $16$ mm and nine times
more accurate than $0.5$ mm. Useful rules of thumb: keep the intensity
contrast $|\Delta I|/I \approx \Delta z\,|\nabla^2\phi|/k$ to a few per cent,
keep the diffraction blur $\sqrt{\lambda \Delta z}$ below the smallest phase
feature, and stay well inside the first caustic. Several planes with
`derivative="polyfit"` relax the trade-off: higher order removes the
nonlinearity at larger $\Delta z$, more planes average the noise.

## Curvature sensing

Roddier's curvature wavefront sensor (Roddier 1988) is the TIE applied to an
illuminated pupil. With uniform illumination $I_0$ inside an aperture of
boundary $\partial\Omega$, the TIE becomes

$$
\frac{I(-\Delta z) - I(+\Delta z)}{I(-\Delta z) + I(+\Delta z)}
\approx \frac{\Delta z}{k}\left( \nabla^2 \phi - \frac{\partial \phi}{\partial n}\,
\delta_{\partial\Omega} \right),
$$

an interior Laplacian (curvature) signal plus a line signal along the edge
proportional to the outward normal slope. The edge signal is exactly the
Neumann boundary data the Poisson equation needs, and `method="pcg"` uses it
automatically: the aperture's soft edge is part of the non-uniform intensity
$I$, so no boundary condition has to be supplied. For a pupil with defocus,
astigmatism and coma (0.8 rad RMS) in the test suite, `"pcg"` recovers the
wavefront to 4e-3 rad RMS, while the window-boundary solvers miss the
aperture-edge information (0.28 rad).

## Example

A soft-edged circular pupil with defocus and astigmatism, three planes at
$\pm 1$ mm:

```python
import numpy as np

from solvephase.algorithms.tie import simulate_defocus_stack, tie

n, pitch, wavelength = 256, 10e-6, 633e-9  # grid, 10 um pixels, HeNe
x = (np.arange(n) - (n - 1) / 2) * pitch
xx, yy = np.meshgrid(x, x)
radius = 0.8e-3  # illuminated aperture radius [m]
r = np.hypot(xx, yy)
rho, theta = r / radius, np.arctan2(yy, xx)
phase = 1.5 * (2 * rho**2 - 1) + 1.0 * rho**2 * np.cos(2 * theta)  # [rad]
amplitude = 0.5 * (1 - np.tanh((r - radius) / (2 * pitch)))  # soft-edged pupil

z = [-1e-3, 0.0, 1e-3]  # defocus planes [m]
stack = simulate_defocus_stack(amplitude * np.exp(1j * phase), z, pitch, wavelength)

res = tie(stack, z, pitch=pitch, wavelength=wavelength, method="pcg")
inner = r < 0.9 * radius
err = (res.phase - phase)[inner]
opd_nm = np.std(err) * wavelength / (2 * np.pi) * 1e9
print(f"{res.message}; {res.n_iter} iterations")
print(f"residual: {np.std(err):.4f} rad RMS ({opd_nm:.2f} nm)")
```

which prints a residual of about 1e-3 rad (0.1 nm) RMS after 150 iterations.
Pass CuPy arrays, or `device="gpu"`, to run on the GPU; `res.to_numpy()` brings
the result to the host.

`simulate_defocus_stack` propagates with periodic boundaries: pad the field
when light would otherwise wrap around the window.

## API summary

`tie(intensities, distances, *, pitch, wavelength, method="dct", uniform=False,
derivative="central", order=None, in_focus=None, regularization=1e-6,
intensity_floor=1e-3, mask=None, tol=1e-6, max_iter=2000, check_every=10,
device=None, precision=None)` returns a `TIEResult` with

- `phase` [rad] and `opd` [m] at $z = 0$, piston-free, zero outside `mask`;
- `intensity` (the $I_0$ used) and `didz` (data units per metre);
- `mask`, `method`, `derivative`, `uniform`, `wavelength`;
- `history` (relative residuals, `"pcg"`), `n_iter`, `converged`, `message`,
  `elapsed`, `device`.

## References

- M. R. Teague, "Deterministic phase retrieval: a Green's function solution,"
  *J. Opt. Soc. Am.* **73**, 1434 (1983).
- T. E. Gureyev and K. A. Nugent, "Phase retrieval with the transport-of-intensity
  equation. II. Orthogonal series solution for nonuniform illumination,"
  *J. Opt. Soc. Am. A* **13**, 1670 (1996).
- D. Paganin and K. A. Nugent, "Noninterferometric phase imaging with partially
  coherent light," *Phys. Rev. Lett.* **80**, 2586 (1998).
- L. Waller, L. Tian and G. Barbastathis, "Transport of intensity phase-amplitude
  imaging with higher order intensity derivatives," *Opt. Express* **18**, 12552
  (2010).
- C. Zuo, Q. Chen and A. Asundi, "Boundary-artifact-free phase retrieval with the
  transport of intensity equation: fast solution with use of discrete cosine
  transform," *Opt. Express* **22**, 9220 (2014).
- F. Roddier, "Curvature sensing and compensation: a new concept in adaptive
  optics," *Appl. Opt.* **27**, 1223 (1988).
- C. Zuo, J. Li, J. Sun, Y. Fan, J. Zhang, L. Lu, R. Zhang, B. Wang, L. Huang and
  Q. Chen, "Transport of intensity equation: a tutorial," *Opt. Lasers Eng.*
  **135**, 106187 (2020).
