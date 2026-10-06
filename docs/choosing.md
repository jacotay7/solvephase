# Choosing an algorithm

| You have... | Use | Guide |
|---|---|---|
| Images of a point source with known diversity (defocus, DM probes), known pupil | [`retrieve`][solvephase.retrieve] (or `FocalPlaneProblem` + `solve`) | [Focal-plane retrieval](guide/focal-plane.md) |
| Images of an **unknown extended scene** with known diversity | `phase_diversity` | [Phase diversity](guide/phase-diversity.md) |
| One astigmatic image, a few low-order modes (e.g. a tip/tilt or low-order sensor) | `lift` / `LIFT` | [AO wavefront sensing](guide/ao-wavefront-sensing.md) |
| A sequence of images while a DM corrects (closed loop, small residuals) | `FastAndFurious` | [AO wavefront sensing](guide/ao-wavefront-sensing.md) |
| A far-field diffraction pattern of an isolated object (unknown pupil/object, known support) | `cdi` with HIO/RAAR + shrinkwrap | [CDI](guide/cdi.md) |
| Generic measurements $y = \lvert A x\rvert^2$ (coded masks, random matrices) | `wirtinger` | [Generic models](guide/generic.md) |
| Near-field intensities at a few defocus distances, weak defocus | `tie` | [Transport of intensity](guide/tie.md) |

## Focal-plane decision guide

- **Default:** `sp.retrieve(images, pupil, wavelength, sampling=..., diversity=...)`.
  It is robust to aberrations up to about $0.3\lambda$ RMS with two images.
- **Known noise model:** keep `loss="poisson"`, pass `read_noise` in
  electrons, and convert images to photo-electrons (see
  [`interop.expose`](guide/interop.md)).
- **Few parameters (≤ 300 modes):** `method="lm"` (Levenberg-Marquardt) is
  the fastest and most precise.
- **Pixel-level detail** (segment edges, high-order NCPA): add
  `zonal_refinement=True`, or solve with `basis=None` and `method="lbfgs"`
  from a modal result.
- **Very large aberrations or poor diversity:** use more planes and larger
  defocus, or start from `method="gs"`.
- **Broadband light:** pass an array of wavelengths (and
  `spectral_weights`); the model sums monochromatic PSFs exactly.
- **Undersampled detector:** give the true `sampling` (< 2) and
  `oversample=2` or more for pixel integration.
- **Segmented telescopes:** `Basis.segments(pupil)` for piston/tip/tilt per
  segment. Use several wavelengths to resolve $2\pi$ piston ambiguities.
- **Turbulence-like wavefronts:** `Basis.kl(pupil, n, r0=...)` with
  `prior=True` for a maximum a posteriori estimate.
