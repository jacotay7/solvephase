# Focus sensing with a focus mechanism

A telescope that refocuses with a mechanism (a mirror or detector on a piston
stage, a refocusing lens) can measure its own focus from ordinary images of a
star. [`FocusSensor`][solvephase.FocusSensor] takes point-source images
recorded at known mechanism positions, retrieves the wavefront, and returns
the mechanism position that minimizes the RMS wavefront error, with its
uncertainty. It is built for mechanisms whose moves are expensive (wear,
time, disturbance): every move that corrects focus also serves as the
diversity for the next measurement, so a dedicated diversity move is never
needed, and often one image is enough.

## The mechanism makes the diversity

Each unit of mechanism travel adds a known pupil OPD, the *response*
$r(\mathbf{x})$, in metres per unit. For a pure focus mechanism it is a
defocus map times the sensitivity; a ray-traced response also carries the
mechanism's side effects (field-dependent astigmatism, image motion). An image
taken at position $p_k$ sees the wavefront $\phi$ at a reference position
$p_\text{ref}$ plus the known diversity $r\,(p_k - p_\text{ref})$. This is
phase diversity (Gonsalves 1982; Paxman, Schulz & Fienup 1992) in which the
diversity comes from the mechanism itself. The estimate is maximum likelihood
under photon and read noise, solved with Levenberg-Marquardt, with per-image
flux, background and registration (the mechanism may move the image), and by
default a fitted blur. Use `fit_background="plane"` when a star may sit on a
background gradient (scattered light, the wing of a brighter neighbour): a
ramp that only a constant models is absorbed into the wavefront and drags the
focus estimate.

## One image, and the twin

One in-focus image cannot tell the sign of the even part of the wavefront
(defocus, astigmatism, spherical aberration) when the pupil is
centro-symmetric: the twin $-\phi(-\mathbf{x})$ gives the same image (see
[the even/odd ambiguity](ao-wavefront-sensing.md#the-evenodd-ambiguity)).
Many real pupils are *not* centro-symmetric: three spider vanes, off-axis
vignetting, a mask. Then the twin does not fit the data and one image fixes
the sign. Hubble's focus monitoring has long used single-image phase retrieval
this way (Krist & Burrows 1995; Lallo et al. 2006).

[`FocusSensor.estimate`][solvephase.algorithms.focus.FocusSensor.estimate] always also solves
from the twin of its best solution ([`twin`][solvephase.twin]) and reports the
posterior probability of the chosen solution against any distinct alternative.
A probability near 1 means one image decided the sign. A probability near 0.5
means it did not: move to the estimate anyway, since the second image at a
known offset settles it either way.

## Blur reads as defocus

Pointing jitter during the exposure and detector charge diffusion blur the
image. Unmodelled, the blur is absorbed into the even aberrations and inflates
the estimated defocus, which biases the move. The sensor fits a Gaussian blur
width by default (`fit_blur=True`; see
[nuisance parameters](focal-plane.md#detector-blur-and-jitter)).

## The best position and its uncertainty

The best position minimizes the intensity-weighted RMS of
$\phi + r\,(p - p_\text{ref})$ after removing piston, tip and tilt
(`exclude`):

$$
p^\star = p_\text{ref} - \frac{\langle P r, P \phi\rangle}{\langle P r, P r\rangle},
$$

where $P$ removes the excluded modes. This is the Maréchal (maximum Strehl)
focus. It is not "Z4 = 0" when other even aberrations are present on an
obscured pupil, where Zernike defocus and spherical aberration are not
orthogonal. Since $p^\star$ is linear in the wavefront, its standard deviation
`sigma` follows from the Fisher matrix of the fit. It is a statistical error
only; model errors (pupil, sampling, response) add bias. The reduced fit
statistic `chi2` (about 1 for a good fit) flags a wrong local solution or a
model error.

## Capture range

From a flat start a single image converges for up to about 0.2 waves RMS of
defocus. Before solving, the sensor evaluates the objective on a grid of
pure-focus starting points (`capture`, default half a wave RMS) and starts
from the best ones, which extends the capture range to several waves.

## Deciding whether to move

[`FocusEstimate.should_move`][solvephase.algorithms.focus.FocusEstimate.should_move] moves only
when the required offset exceeds both a deadband (the focus error you accept)
and `k_sigma` times the estimate's own uncertainty, so the mechanism does not
chase noise. A minimal-move loop:

1. Take an image where the mechanism is; estimate.
2. If a move is warranted, move to the estimate and take another image.
3. Re-estimate from both images (warm-started from step 1). Stop if no
   further move is warranted; otherwise move again.

## Calibration and several stars

The response must be known: it sets both the known diversity and the
conversion from wavefront to travel.
[`FocusSensor.calibrate`][solvephase.algorithms.focus.FocusSensor.calibrate] fits one scale
factor on the response from a through-focus series of one star (all images
jointly, `fit_diversity_scale=True`). Spread the positions over about one wave
of defocus on both sides of focus.
[`combine_estimates`][solvephase.combine_estimates] gives the common position
that minimizes the weighted mean-square wavefront error over several stars
(field points) imaged at once. Pass `reject=5, floor=...` (mechanism units) to
drop a star whose own best position is far from the others'
([`outlier_mask`][solvephase.outlier_mask]): with a poor PSF or a bright
background gradient one star occasionally converges to a wrong solution, and a
weighted mean carries a large fraction of its error.

## Example

A three-vane pupil, a focus mechanism of 2 nm RMS defocus per micrometre, and
a star 60 nm RMS out of focus with some astigmatism and coma:

```python
import numpy as np
import solvephase as sp

pupil = sp.Pupil.circular(48, 0.5, obscuration=0.4, spiders=3, spider_width=0.03)
wavelength = 650e-9
defocus = sp.Basis.zernike(pupil, 1, start=4).mode_maps()[0]
response = 2e-9 * defocus  # metres of OPD per micrometre of travel

truth = 60e-9 * defocus + sp.random_aberration(pupil, 15e-9, n_modes=6, start=5, seed=1)


def expose(positions, seed):
    """Camera frames at mechanism positions (um), with 0.3 px of jitter."""
    model = sp.FocalPlaneModel(
        pupil,
        wavelength,
        32,
        sampling=2.0,
        blur=0.3,
        diversity=[response * p for p in positions],
    )
    return sp.simulate_images(model, truth, photons=2e5, read_noise=2.0, seed=seed)


sensor = sp.FocusSensor(pupil, wavelength, 32, response, sampling=2.0, basis=15, read_noise=2.0)
first = sensor.estimate(expose([0.0], seed=2)[0], [0.0])  # one image, no move yet
print(first.summary())
true_best = sensor.best_position(truth, 0.0)
print(f"true best position {true_best:+.2f} um")

if first.should_move(deadband=2.0):  # one move, to the estimate
    p1 = first.position
    images = np.concatenate([expose([0.0], seed=2), expose([p1], seed=3)])
    second = sensor.estimate(images, [0.0, p1], start=first)
    print(
        f"after one move: residual {second.move:+.2f} um, "
        f"move again: {second.should_move(deadband=2.0)}"
    )
```

Calibrating the response from a through-focus series, here with a mechanism
10 % stronger than assumed:

```python
positions = [-60.0, -30.0, 0.0, 30.0, 60.0]
model = sp.FocalPlaneModel(
    pupil,
    wavelength,
    32,
    sampling=2.0,
    blur=0.3,
    diversity=[1.1 * response * p for p in positions],
)
sweep = sp.simulate_images(model, truth, photons=2e5, read_noise=2.0, seed=4)
cal = sensor.calibrate(sweep, positions)
print(f"response scale {cal.scale:.4f} +/- {cal.sigma:.4f}")
calibrated = sensor.with_response_scale(cal.scale)
```

## References

- R. A. Gonsalves, "Phase retrieval and diversity in adaptive optics",
  Opt. Eng. 21, 829 (1982).
- R. G. Paxman, T. J. Schulz & J. R. Fienup, "Joint estimation of object and
  aberrations by using phase diversity", JOSA A 9, 1072 (1992).
- S. Meimon, T. Fusco & L. M. Mugnier, "Lift: a focal-plane wavefront sensor
  for real-time low-order sensing on faint sources", Opt. Lett. 35, 3036
  (2010).
- J. E. Krist & C. J. Burrows, "Phase-retrieval analysis of pre- and
  post-repair Hubble Space Telescope images", Appl. Opt. 34, 4951 (1995).
- M. D. Lallo et al., "Temporal optical behavior of HST: focus, coma and
  astigmatism history", Proc. SPIE 6270, 62701N (2006).
