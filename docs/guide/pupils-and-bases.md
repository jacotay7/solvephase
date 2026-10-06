# Pupils and bases

## Pupils

A [`Pupil`][solvephase.Pupil] is a real, non-negative amplitude map on a
square grid with its pixel `pitch` (m), reference `diameter` (m, which sets
$\lambda/D$ and the Zernike radius) and central `obscuration` ratio.
Analytic constructors anti-alias edges by supersampling each pixel.

```python
import numpy as np
import solvephase as sp

circ = sp.Pupil.circular(128, 8.0, obscuration=0.14, spiders=4, spider_width=0.05)
seg = sp.Pupil.segmented_hexagonal(128, rings=2, segment_size=1.3, gap=0.01)
presets = [sp.Pupil.keck(128), sp.Pupil.jwst(128), sp.Pupil.vlt(128), sp.Pupil.hst(128)]
custom = sp.Pupil.from_array(np.ones((32, 32)), diameter=1.0)
print(circ, seg.n_segments, [p.name for p in presets], custom.pitch)
```

Segmented pupils carry a label image, `pupil.segments`, with values
`1..S`. The presets are approximate geometries meant for algorithm work.

## Bases

A [`Basis`][solvephase.Basis] maps coefficients to an OPD over the pupil's
illuminated pixels. Modes are unit RMS, so a coefficient is the RMS OPD of
that mode in metres.

| Constructor | Modes | Source |
|---|---|---|
| `Basis.zernike(pupil, n, start=2)` | Zernike (Noll/ANSI/Fringe), annular when obscured | [aobasis](https://github.com/jacotay7/aobasis) |
| `Basis.kl(pupil, n, r0=...)` | Karhunen-Loève of von Kármán turbulence, with prior `variances` | aobasis |
| `Basis.fourier(pupil, n)` | sines/cosines, orthonormalized | aobasis |
| `Basis.from_aobasis(pupil, Generator, n, ...)` | any aobasis generator | aobasis |
| `Basis.segments(pupil)` | piston/tip/tilt per segment | built in |
| `Basis.gaussian_dm(pupil, actuators_across)` | Gaussian influence functions (coefficients = DM commands) | aobasis |
| `Basis.influence_functions(pupil, ifs, m2c=...)` | your measured DM influence functions, optionally through an M2C | built in |
| `Basis.zonal(pupil)` | one value per pixel | built in |
| `Basis.from_maps(pupil, maps)` | any modes you supply | built in |

Bases combine with `+` (`concatenate`), select with `subset`, and
`orthonormalized()` makes them orthonormal over the sampled pupil (spiders,
gaps and grey edge pixels included), which improves solver conditioning.

```python
pupil = sp.Pupil.circular(64, 1.0, obscuration=0.2)
basis = sp.Basis.zernike(pupil, 10) + sp.Basis.fourier(pupil, 6)
coeffs = np.zeros(basis.n_modes)
coeffs[2] = 50e-9  # 50 nm RMS of defocus (Z4)
opd = basis.synthesize(coeffs)
print(basis.labels[:4], np.allclose(basis.fit(opd), coeffs))
```

Retrieving in a DM basis gives commands you can apply directly. With the
same influence functions, aobasis M2C matrices plug in through
`Basis.influence_functions(..., m2c=m2c)`, so results can feed a
reconstructor or a controller such as pyRTC.
