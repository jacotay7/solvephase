# Coherent diffraction imaging (CDI)

In coherent diffraction imaging a compact object is illuminated by a coherent
beam (X-rays, electrons, visible light) and a detector in the far field
records the intensity of its diffraction pattern, $I(\mathbf{k}) =
|\mathcal{F}o(\mathbf{k})|^2$. No lens forms an image: the phase of
$\mathcal{F}o$ is lost and must be recovered numerically. `solvephase`
implements the standard family of iterative projection algorithms, with
shrinkwrap support refinement and many random starts run simultaneously as
one batch (on the GPU if you like).

```python
from solvephase.algorithms.cdi import (
    CDIResult,
    ShrinkwrapConfig,
    align_object,
    autocorrelation_support,
    cdi,
    simulate_cdi,
)
```

## Oversampling: why the phase is recoverable

A diffraction pattern sampled at the Nyquist rate of the *intensity* holds
twice as many samples per axis as the object has pixels. Equivalently the
reconstruction window is larger than the object, and the region outside the
object is known to be zero. Miao, Sayre & Chapman (1998) phrased the
condition as an **oversampling ratio**

$$
\sigma = \frac{\text{window area}}{\text{object area}} > 2,
$$

which in 2-D makes the number of known intensities exceed the number of
unknown (complex) object values. `simulate_cdi(obj, oversampling=s)` pads an
object of $m_y \times m_x$ pixels into a window of $s\,m_y \times s\,m_x$
pixels, so $\sigma = s^2$ for an object that fills its box: $s = 2$
($\sigma = 4$) is comfortable, $s$ below $\sqrt{2}$ is not unique.

The far-field arrays are **centred** (DC at `(ny // 2, nx // 2)`, i.e. what
`numpy.fft.fftshift` produces) and the transform is unitary, so
$\sum |\mathcal{F}o|^2 = \sum |o|^2$. The object and its support are centred
too. The solver shifts once on entry and once on exit, never inside the loop.

## The two constraint sets

Write $x$ for the current iterate (a complex image on the window), $A$ for
the measured modulus and $S$ for the support. The two projections are

$$
P_M x = \mathcal{F}^{-1}\!\left[ A\, \frac{\mathcal{F}x}{|\mathcal{F}x|} \right]
\quad\text{(measured pixels; elsewhere } \mathcal{F}x \text{ is kept)},
$$

$$
P_S x = \begin{cases} c(x) & \mathbf{r} \in S \\ 0 & \mathbf{r} \notin S \end{cases},
$$

where $c$ is the nearest value allowed by the object constraint
(`constraint=`): `"complex"` keeps $x$, `"real"` keeps $\operatorname{Re} x$,
and `"positive"` keeps $\max(\operatorname{Re} x, 0)$. Optional `bounds=(lo,
hi)` also clip the modulus inside the support. Pixels with `measured=False`
(a beamstop, detector gaps, saturated pixels) and non-finite data are left
free by $P_M$, which is the exact projection onto the set of fields that
agree with the data where it exists. The reflectors are $R = 2P - I$.

## Algorithms

Each iteration below costs one batched FFT pair (DM and OSS: two). With
$y = P_M x$:

| name | update $x \to x'$ | reference |
| --- | --- | --- |
| `er` | $x' = P_S\, y$ | Gerchberg & Saxton 1972; Fienup 1978, 1982 |
| `hio` | $x' = y$ where $y$ satisfies the object constraints, $x - \beta y$ elsewhere | Fienup 1982 |
| `dm` | $x' = x + \beta\,[P_S f_M(x) - P_M f_S(x)]$, $f_M = (1+\gamma_M)P_M - \gamma_M I$, $f_S = (1+\gamma_S)P_S - \gamma_S I$, $\gamma_S = -1/\beta$, $\gamma_M = 1/\beta$ | Elser 2003 |
| `raar` | $x' = \tfrac{\beta}{2}(R_S R_M + I)x + (1-\beta)\,y = \beta x + \beta P_S(2y - x) + (1-2\beta)\,y$ | Luke 2005 |
| `rrr` | $x' = x + \beta\,[P_S(2y - x) - y]$ | Elser, Lan & Bendory 2018 |
| `asr` (`dr`) | $x' = \tfrac12 (R_S R_M + I)x = x + P_S(2y - x) - y$ | Bauschke, Combettes & Luke 2002 |
| `hpr` | $x' = \tfrac12[R_S(R_M + (\beta - 1)P_M) + I + (1-\beta)P_M]x = x - \beta y + P_S((1+\beta)y - x)$ | Bauschke, Combettes & Luke 2003 |
| `oss` | HIO, then outside $S$: $x' \leftarrow \mathcal{F}^{-1}[W_\alpha\,\mathcal{F}x']$, $W_\alpha(\mathbf{k}) = e^{-k^2/2\alpha^2}$ | Rodriguez et al. 2013 |

Notes:

- **ER** is a pure alternating projection: its modulus error never increases,
  but it stagnates in local minima. Use it to polish the result of another
  algorithm.
- **HIO** and **HPR** coincide for linear constraints (support only, or
  realness). With positivity HIO tests the constraint on $y$ (Fienup's
  rule), HPR on $(1+\beta)y - x$.
- **ASR** is Douglas-Rachford; it is RAAR with $\beta = 1$ and RRR with
  $\beta = 1$. RAAR with $\beta < 1$ adds a damping $P_M$ term that helps on
  inconsistent (noisy) data but slows convergence on consistent data.
- **OSS** steps the filter width $\alpha$ (in frequency pixels) linearly from
  $2N$ to $N/5$ over `oss_stages=10` equal segments, as in the authors'
  reference code. At the end of each segment the start's best iterate (lowest
  modulus error at a check) is kept and restarts the next segment; the stage
  ends on the best iterate of all segments.
- The solution estimate returned for every algorithm is $P_S(y)$ (for DM,
  $P_S f_M(x)$), which equals the fixed point's image on consistent data.

`beta=None` uses per-algorithm defaults (`DEFAULT_BETA`): 0.9 for HIO, HPR,
OSS and DM, 0.98 for RAAR, 0.5 for RRR. A stage may set its own value.

### Schedules

Algorithms are chained with a schedule, as a string or a list of tuples with
an optional per-stage $\beta$:

```py
cdi(mags, support, schedule="hio:400,er:100")
cdi(mags, support, schedule=[("raar", 300, 0.9), ("hio", 200), ("er", 50)])
```

### Which one?

- **Noise-free or high-count data, known support:** HIO $\to$ ER is the
  workhorse and the fastest per iteration; DM, RAAR ($\beta \approx 0.98$)
  and HPR have fewer stagnation modes. Try a few with `starts=8`.
- **Noisy data:** OSS (the smoothness constraint outside the support
  suppresses noise-driven oscillations), or RAAR with $\beta$ between 0.75
  and 0.9, followed by a short ER.
- **Unknown support:** HIO or RAAR with shrinkwrap, then ER with the support
  frozen (`shrinkwrap={"stop": ...}`).
- **Stagnation in a twin/translated superposition** (common for
  centro-symmetric supports such as squares): more random starts.

## Shrinkwrap

When the support is unknown, start from the autocorrelation: the inverse
transform of the intensity is $o \star o$, whose support is the difference
set $S - S$, twice the object's extent.
`autocorrelation_support(intensity, threshold=0.04)` thresholds it.
Shrinkwrap (Marchesini et al. 2003) then refines the support every
`every` iterations:

$$
S \leftarrow \{\mathbf{r} : (G_\sigma * |y|)(\mathbf{r}) > t \max (G_\sigma * |y|)\},
\qquad \sigma \leftarrow \max(\sigma_{\min}, d\,\sigma),
$$

with $G_\sigma$ a Gaussian of standard deviation $\sigma$ pixels and $y = P_M
x$ the current data-consistent image. Defaults (`ShrinkwrapConfig`) follow
the paper: `every=20`, `threshold=0.2`, `sigma=3`, `sigma_min=1.5`,
`decay=0.99`. Refined supports stay inside the initial one. Each start has its
own support; `CDIResult.support` is the best start's.

## Multi-start and the GPU

`starts=B` runs $B$ independent random starts (random Fourier phases on the
measured modulus, drawn on the host from `seed`) as a leading batch
dimension: every iteration is one batched FFT over `(B, ny, nx)`, which
keeps a GPU busy even for small patterns. The returned `object` is the start
with the lowest final modulus error; `start_errors` holds all of them and
`objects` all estimates (for averaging or a phase-retrieval transfer
function). The same seed gives the same starts on CPU and GPU; in double
precision the two agree to rounding.

Errors are evaluated only every `check_every` iterations (each check is a
device synchronization and one extra FFT):

$$
E_M = \sqrt{\frac{\sum_{\text{measured}} (|\mathcal{F}\hat o| - A)^2}{\sum_{\text{measured}} A^2}},
\qquad
E_S = \frac{\| u - P_S u \|}{\| u \|},
$$

for the estimate $\hat o = P_S u$. The run stops early when the best start's
$E_M \le$ `tol`.

Measured speed on a 256 x 256 window (128 x 128 object), iterations per
second; a "start-iteration" is one start advanced by one iteration:

| | HIO, 1 start | HIO, 16 starts |
| --- | --- | --- |
| CPU, 16 cores, double | ~270 it/s | ~13 it/s (~210 start-it/s) |
| CPU, single | ~650 it/s | ~32 it/s (~520 start-it/s) |
| GPU (Quadro P620), single | ~2100 it/s | ~210 it/s (~3300 start-it/s) |

## Ambiguities and `align_object`

The modulus $|\mathcal{F}o|$ is unchanged by

- a global phase $o \to e^{i\theta} o$;
- a translation $o(\mathbf{r}) \to o(\mathbf{r} - \mathbf{s})$ (a linear phase
  in Fourier space);
- the twin image $o(\mathbf{r}) \to \overline{o(-\mathbf{r})}$.

To compare a reconstruction with a known object, `align_object(estimate,
reference)` tries the estimate and its twin, finds the translation from the
cross-correlation peak refined to 1/`upsample` pixel with an upsampled matrix
DFT (Guizar-Sicairos, Thurman & Fienup 2008), and fits the complex factor
$c$:

$$
\text{error} = \min_{c,\ \mathbf{s},\ \text{twin}} \frac{\| c\, \hat o_{\mathbf{s}} - o_{\text{ref}} \|}{\| o_{\text{ref}} \|}.
$$

It returns `(aligned, error)` with `aligned` $= c\,\hat o_{\mathbf{s}}$. Pass
`scale=False` to remove only a global phase.

## Example

```python
import numpy as np
from scipy.ndimage import gaussian_filter

from solvephase.algorithms.cdi import align_object, autocorrelation_support, cdi, simulate_cdi

# A smooth positive 48 x 48 object with an irregular outline.
rng = np.random.default_rng(0)
yy, xx = np.mgrid[:48, :48] - 23.5
outline = np.hypot(yy, xx) < 18 * (1 + 0.25 * np.cos(3 * np.arctan2(yy, xx)))
obj = np.where(outline, 0.3 + gaussian_filter(rng.uniform(size=(48, 48)), 1.5), 0.0)

# Oversample 2x per axis (a 96 x 96 window) and add Poisson noise.
data = simulate_cdi(obj, oversampling=2, photons=1e8, seed=1)

# The support is unknown: start from the autocorrelation and shrinkwrap.
support = autocorrelation_support(data.intensity, threshold=0.04)
result = cdi(
    data.magnitudes,
    support,
    schedule="hio:800,er:100",
    constraint="positive",
    shrinkwrap={"every": 20, "threshold": 0.2, "stop": 800},
    starts=8,
    seed=0,
)
aligned, error = align_object(result.object, data.object)
print(result)
print(f"modulus error {result.modulus_error:.3g}, error vs truth {error:.3g}")
```

This prints a modulus error of about $5\times10^{-3}$ (the Poisson noise
floor) and an error against the truth of about $3\times10^{-3}$. Add
`device="gpu"` to run the same solve on CUDA; `result.to_numpy()` copies the
arrays to the host.

## References

- R. W. Gerchberg and W. O. Saxton, "A practical algorithm for the
  determination of phase from image and diffraction plane pictures," Optik
  35, 237 (1972).
- J. R. Fienup, "Reconstruction of an object from the modulus of its Fourier
  transform," Opt. Lett. 3, 27 (1978).
- J. R. Fienup, "Phase retrieval algorithms: a comparison," Appl. Opt. 21,
  2758 (1982).
- J. Miao, D. Sayre and H. N. Chapman, "Phase retrieval from the magnitude of
  the Fourier transforms of nonperiodic objects," JOSA A 15, 1662 (1998).
- H. H. Bauschke, P. L. Combettes and D. R. Luke, "Phase retrieval, error
  reduction algorithm, and Fienup variants: a view from convex
  optimization," JOSA A 19, 1334 (2002).
- H. H. Bauschke, P. L. Combettes and D. R. Luke, "Hybrid
  projection-reflection method for phase retrieval," JOSA A 20, 1025 (2003).
- V. Elser, "Phase retrieval by iterated projections," JOSA A 20, 40 (2003).
- S. Marchesini, H. He, H. N. Chapman, S. P. Hau-Riege, A. Noy, M. R.
  Howells, U. Weierstall and J. C. H. Spence, "X-ray image reconstruction
  from a diffraction pattern alone," Phys. Rev. B 68, 140101(R) (2003).
- D. R. Luke, "Relaxed averaged alternating reflections for diffraction
  imaging," Inverse Problems 21, 37 (2005).
- S. Marchesini, "A unified evaluation of iterative projection algorithms
  for phase retrieval," Rev. Sci. Instrum. 78, 011301 (2007).
- M. Guizar-Sicairos, S. T. Thurman and J. R. Fienup, "Efficient subpixel
  image registration algorithms," Opt. Lett. 33, 156 (2008).
- J. A. Rodriguez, R. Xu, C.-C. Chen, Y. Zou and J. Miao, "Oversampling
  smoothness: an effective algorithm for phase retrieval of noisy
  diffraction intensities," J. Appl. Cryst. 46, 312 (2013).
- V. Elser, T.-Y. Lan and T. Bendory, "Benchmark problems for phase
  retrieval," SIAM J. Imaging Sci. 11, 2429 (2018).
