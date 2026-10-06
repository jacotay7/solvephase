# Validation

Tests check that the code does what it says; validation checks that what it
says is physically and statistically right. `validation/validate.py` compares
solvephase with theory and with an independent code. It writes a JSON report
and a figure, and exits non-zero if any check fails. CI runs a reduced
version on every push, and the full report is versioned in
[`validation/artifacts`](https://github.com/jacotay7/solvephase/tree/main/validation/artifacts).

```bash
python validation/validate.py --output validation/artifacts      # full (~2 min)
python validation/validate.py --quick --output /tmp/validation    # CI
```

![Validation figure](https://raw.githubusercontent.com/jacotay7/solvephase/main/validation/artifacts/validation.png)

## Checks and results

| Check | Result | Threshold |
|---|---|---|
| Images vs HCIPy's Fraunhofer propagator (FFT q=2, q=3; MFT q=1.37; 3-wavelength broadband), max normalized difference | 4e-16 – 4e-15 | 1e-10 |
| Airy peak normalization vs $A\theta^2/\lambda^2$ (relative) | 2.2e-5 | 2e-3 |
| Airy first dark ring vs $1.22\,\lambda/D$ | 0.03 λ/D (pixel 1/8 λ/D) | 1/8 λ/D |
| Poisson ML modal estimator: Monte-Carlo variance / Cramér-Rao bound (200 noise draws, 12 modes) | 1.10 | 0.71 – 1.40 |
| Poisson ML estimator bias, max over modes | 0.09 σ_CRB | 0.85 σ_CRB |
| `retrieve()` capture with two images (±0.3 λ RMS defocus), success rate up to 0.3 waves RMS | 100 % (100 % to 0.4; 80 % at 0.5–0.6; 70 % at 0.8) | 100 % |
| Single vs double precision solution difference | 8e-6 nm | 0.05 nm |
| Phase diversity on an extended scene: relative wavefront error | 3.1 % | 10 % |
| LIFT variance / Cramér-Rao bound (150 draws) | 1.01 | 0.68 – 1.46 |
| Fast & Furious closed loop: residual after 20 steps / initial | 0.08 | 0.2 |
| CDI (HIO→ER, 4 starts) noise-free exact recovery rate | 100 % | 100 % |
| TIE (masked PCG) on a soft-edged aperture: phase error | 9.6e-4 rad | 0.02 rad |
| Reweighted amplitude flow, Gaussian model m = 8n: exact recovery rate | 100 % | 100 % |

## What the checks establish

- **The forward model is right.** Agreement with HCIPy, an independently
  written optics package, to machine precision (same orientation, sampling
  and normalization) covers the FFT and MFT engines and broadband
  summation. The Airy checks pin the absolute normalization and the
  sampling scale.
- **The estimator is optimal.** At high SNR the Poisson maximum-likelihood
  estimator is unbiased, and its variance equals the Cramér-Rao bound
  computed from the model's Fisher information (Paxman, Schulz & Fienup
  1992). No unbiased estimator can do better. LIFT reaches its bound too.
- **The default is robust.** `retrieve()` recovers random aberrations from
  two images over the documented range without tuning. The capture curve
  in the figure shows where it starts to fail.
- **Every family meets its documented accuracy** on reference problems with
  known truth.

Unit tests add adjoint identities for every operator, finite-difference
checks of every gradient, CPU/GPU parity, and recovery tests per algorithm
(`python -m pytest --run-slow --run-gpu`).
