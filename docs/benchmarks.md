# Benchmarks

Speed numbers are warm, steady-state medians, measured separately from setup
(building models and plans). Each artifact records the hardware and the
dependency versions; the raw JSON is versioned in
[`benchmarks/artifacts`](https://github.com/jacotay7/solvephase/tree/main/benchmarks/artifacts).

```bash
python benchmarks/run.py --output my-results.json        # full suite, CPU and GPU
python benchmarks/compare.py --output my-compare.json    # head-to-head comparison
python benchmarks/render_table.py my-results.json --compare my-compare.json
```

## Head-to-head

No maintained Python package provides a focal-plane phase-retrieval solver.
The baselines are therefore what you would write otherwise: an HCIPy forward
model driven by SciPy optimizers with finite-difference derivatives, and
textbook NumPy loops for Gerchberg-Saxton/Misell and HIO. Every method sees
the same data.

Measured on an Intel i7-10700 (8 cores) and an NVIDIA Quadro P620 (an entry-level 512-core GPU); see the artifact for versions.

**focal 128x128, 36 modes, 2 images**

| method | time | wavefront error |
|---|---|---|
| solvephase LM (cpu) | 0.46 s | 0.13 nm |
| solvephase LM (gpu) | 0.14 s | 0.13 nm |
| HCIPy + SciPy least_squares | 5.31 s | 0.19 nm |
| HCIPy + SciPy L-BFGS-B | 8.51 s | 0.12 nm |

**Misell/GS 128x128, 2 images (iterations/s)**

| method | iterations/s | speed-up |
|---|---|---|
| NumPy textbook loop | 156 | 1.0x |
| solvephase (cpu) | 268 | 1.7x |
| solvephase (gpu) | 1,493 | 9.5x |

**CDI HIO 256x256 (iterations/s)**

| method | iterations/s | speed-up |
|---|---|---|
| NumPy textbook loop | 252 | 1.0x |
| solvephase (cpu) | 794 | 3.2x |
| solvephase (cpu, 16 starts, per start) | 523 | 2.1x |
| solvephase (gpu) | 2,957 | 11.7x |
| solvephase (gpu, 16 starts, per start) | 3,492 | 13.9x |

## Suite

Hardware: Intel(R) Core(TM) i7-10700 CPU @ 2.90GHz (16 threads); Quadro P620 (CuPy 14.1.1); NumPy 2.2.6, SciPy 1.17.1; solvephase 0.1.0.

**Focal-plane LM time to solution (36 modes, 2 images)**

| size | CPU (double) | GPU (single) |
|---|---|---|
| 64 | 0.13 s | 0.09 s |
| 128 | 0.81 s | 0.25 s |
| 256 | 2.34 s | 0.89 s |

**Objective + gradient evaluation (zonal, 2 images)**

| size | wavelengths | CPU (double) | GPU (single) |
|---|---|---|---|
| 128 | 1 | 3.8 ms | 2.0 ms |
| 128 | 5 | 11.4 ms | 2.5 ms |
| 256 | 1 | 15.7 ms | 5.9 ms |
| 256 | 5 | 67.1 ms | 9.1 ms |
| 512 | 1 | 97.5 ms | 14.4 ms |

**Gerchberg-Saxton/Misell iterations per second (3 images)**

| size | CPU (double) | GPU (single) |
|---|---|---|
| 128 | 183 | 973 |
| 256 | 34 | 273 |
| 512 | 6 | 70 |

**`retrieve()` end to end (robust default)**

| size | CPU (double) | GPU (single) |
|---|---|---|
| 64 | 0.79 s | 0.60 s |
| 128 | 4.87 s | 0.88 s |

**CDI iterations per second (total over starts)**

| size | algorithm | starts | CPU (double) | GPU (single) |
|---|---|---|---|---|
| 256 | hio | 1 | 521 | 2,969 |
| 256 | hio | 16 | 553 | 3,925 |
| 256 | raar | 1 | 756 | 2,909 |
| 256 | raar | 16 | 501 | 3,331 |
| 512 | hio | 1 | 174 | 905 |
| 512 | hio | 16 | 121 | 955 |
| 512 | raar | 1 | 164 | 779 |
| 512 | raar | 16 | 108 | 816 |
| 1024 | hio | 1 | 26 | 226 |
| 1024 | raar | 1 | 25 | 195 |

**TIE solve (3 planes, non-uniform intensity)**

| size | method | CPU (double) | GPU (single) |
|---|---|---|---|
| 512 | fft | 38.0 ms | 2.7 ms |
| 512 | dct | 49.0 ms | 6.7 ms |
| 1024 | fft | 107 ms | 8.7 ms |
| 1024 | dct | 112 ms | 25.3 ms |
| 2048 | fft | 253 ms | 33.3 ms |
| 2048 | dct | 336 ms | 103 ms |

**LIFT estimate (10 modes, one image)**

| size | CPU (double) | GPU (single) |
|---|---|---|
| 32 | 10.2 ms | 37.7 ms |
| 64 | 19.6 ms | 42.9 ms |
| 128 | 70.5 ms | 45.1 ms |

**Fast & Furious step latency**

| size | CPU (double) | GPU (single) |
|---|---|---|
| 64 | 0.5 ms | 1.3 ms |
| 128 | 1.9 ms | 1.3 ms |
| 256 | 4.6 ms | 1.7 ms |

**Extended-object phase diversity solve (20 modes)**

| size | CPU (double) | GPU (single) |
|---|---|---|
| 64 | 0.09 s | 0.24 s |
| 128 | 0.26 s | 0.22 s |

**Coded-diffraction retrieval (6 masks) to 1e-6**

| size | method | CPU (double) | GPU (single) |
|---|---|---|---|
| 64 | raf | 0.17 s | 0.10 s |
| 64 | lbfgs | 0.12 s | 0.17 s |
| 128 | raf | 0.52 s | 0.11 s |
| 128 | lbfgs | 0.39 s | 0.17 s |
| 256 | raf | 2.38 s | 0.45 s |
| 256 | lbfgs | 1.64 s | 0.38 s |

**Least-squares phase unwrapping (spidered pupil)**

| size | CPU (double) | GPU (single) |
|---|---|---|
| 128 | 34.6 ms | 64.7 ms |
| 256 | 101 ms | 97.7 ms |
| 512 | 496 ms | 364 ms |

## Arm workstation: Neoverse-N1 with RTX 4060 and RTX A400

A second data point on different silicon: an 80-core Ampere Neoverse-N1
(aarch64) host with an RTX 4060 (8 GB) and an RTX A400 (4 GB). The host is
shared, so every run was pinned to 16 idle cores with 16 BLAS/FFT threads,
and CPU rows can vary with the frequency governor. Artifacts:
[`v0.1.0-neoverse-n1-rtx4060.json`](https://github.com/jacotay7/solvephase/blob/main/benchmarks/artifacts/v0.1.0-neoverse-n1-rtx4060.json),
[`v0.1.0-neoverse-n1-rtxa400.json`](https://github.com/jacotay7/solvephase/blob/main/benchmarks/artifacts/v0.1.0-neoverse-n1-rtxa400.json)
(GPU only),
[`v0.1.0-compare-neoverse-n1-rtx4060.json`](https://github.com/jacotay7/solvephase/blob/main/benchmarks/artifacts/v0.1.0-compare-neoverse-n1-rtx4060.json),
[`v0.1.0-methods-neoverse-n1-rtx4060.json`](https://github.com/jacotay7/solvephase/blob/main/benchmarks/artifacts/v0.1.0-methods-neoverse-n1-rtx4060.json).
The full test suite (`--run-gpu --run-slow`) and `validation/validate.py`
pass on this machine.

**Across machines** (lower is better for times, higher for rates)

| case | i7-10700 CPU | Neoverse-N1 CPU (16 cores) | Quadro P620 | RTX A400 | RTX 4060 |
|---|---|---|---|---|---|
| Focal LM 128², 36 modes (s) | 0.81 | 0.60 | 0.25 | 0.21 | 0.06 |
| Focal LM 256², 36 modes (s) | 2.34 | 1.87 | 0.89 | 0.40 | 0.22 |
| CDI HIO 1024², 1 start (it/s) | 26 | 34 | 226 | 309 | 1,854 |
| CDI HIO 256², 16 starts (it/s, total) | 553 | 559 | 3,925 | 5,257 | 34,722 |
| TIE 1024², FFT (ms) | 107.2 | 64.8 | 8.7 | 5.6 | 1.9 |
| Misell/GS 512² (it/s) | 6 | 8 | 70 | 95 | 266 |
| Fast & Furious 64² step (ms) | 0.5 | 0.7 | 1.3 | 1.4 | 1.3 |

- 16 Neoverse-N1 cores and the 8-core i7-10700 are within about ±30%. The
  Arm cores win on large transforms (TIE, large focal-plane problems) and
  lose on small, latency-bound solves (Fast & Furious, LIFT, small Wirtinger).
- On the GPU, small problems (≤ 128²) take the same 1–2 ms on every card:
  they are bound by kernel-launch latency, not by the GPU. Large problems
  scale with the card. CDI at 1024² is 8x faster on the RTX 4060 than on the
  P620, and batched 16-start CDI reaches about 35,000 iterations/s.
- The RTX A400 is 1.2–2.2x faster than a Quadro P620 on these cases, typically about 1.4x.

Hardware: Ampere Neoverse-N1 (aarch64), 16 of 80 cores pinned with 16 threads; NVIDIA GeForce RTX 4060 (CuPy 14.2.0); NumPy 2.5.3, SciPy 1.18.1; solvephase 0.1.0.

**Focal-plane LM time to solution (36 modes, 2 images)**

| size | CPU (double) | GPU (single) |
|---|---|---|
| 64 | 0.14 s | 0.07 s |
| 128 | 0.60 s | 0.06 s |
| 256 | 1.87 s | 0.22 s |

**Objective + gradient evaluation (zonal, 2 images)**

| size | wavelengths | CPU (double) | GPU (single) |
|---|---|---|---|
| 128 | 1 | 4.8 ms | 2.2 ms |
| 128 | 5 | 13.8 ms | 2.6 ms |
| 256 | 1 | 16.0 ms | 2.2 ms |
| 256 | 5 | 59.8 ms | 2.5 ms |
| 512 | 1 | 77.5 ms | 2.4 ms |

**Gerchberg-Saxton/Misell iterations per second (3 images)**

| size | CPU (double) | GPU (single) |
|---|---|---|
| 128 | 134 | 1,066 |
| 256 | 35 | 884 |
| 512 | 8 | 266 |

**`retrieve()` end to end (robust default)**

| size | CPU (double) | GPU (single) |
|---|---|---|
| 64 | 1.00 s | 0.57 s |
| 128 | 4.23 s | 0.61 s |

**CDI iterations per second (total over starts)**

| size | algorithm | starts | CPU (double) | GPU (single) |
|---|---|---|---|---|
| 256 | hio | 1 | 370 | 2,639 |
| 256 | hio | 16 | 559 | 34,722 |
| 256 | raar | 1 | 458 | 2,556 |
| 256 | raar | 16 | 521 | 24,050 |
| 512 | hio | 1 | 148 | 2,464 |
| 512 | hio | 16 | 120 | 3,668 |
| 512 | raar | 1 | 132 | 2,394 |
| 512 | raar | 16 | 105 | 3,046 |
| 1024 | hio | 1 | 34 | 1,854 |
| 1024 | raar | 1 | 30 | 1,361 |

**TIE solve (3 planes, non-uniform intensity)**

| size | method | CPU (double) | GPU (single) |
|---|---|---|---|
| 512 | fft | 15.5 ms | 1.9 ms |
| 512 | dct | 19.0 ms | 4.5 ms |
| 1024 | fft | 64.8 ms | 1.9 ms |
| 1024 | dct | 71.2 ms | 4.5 ms |
| 2048 | fft | 225 ms | 6.7 ms |
| 2048 | dct | 232 ms | 25.5 ms |

**LIFT estimate (10 modes, one image)**

| size | CPU (double) | GPU (single) |
|---|---|---|
| 32 | 11.0 ms | 69.8 ms |
| 64 | 27.2 ms | 64.4 ms |
| 128 | 83.5 ms | 67.4 ms |

**Fast & Furious step latency**

| size | CPU (double) | GPU (single) |
|---|---|---|
| 64 | 0.7 ms | 1.3 ms |
| 128 | 2.6 ms | 1.5 ms |
| 256 | 6.0 ms | 1.4 ms |

**Extended-object phase diversity solve (20 modes)**

| size | CPU (double) | GPU (single) |
|---|---|---|
| 64 | 0.10 s | 0.36 s |
| 128 | 0.40 s | 0.49 s |

**Coded-diffraction retrieval (6 masks) to 1e-6**

| size | method | CPU (double) | GPU (single) |
|---|---|---|---|
| 64 | raf | 0.26 s | 0.12 s |
| 64 | lbfgs | 0.20 s | 0.22 s |
| 128 | raf | 0.73 s | 0.12 s |
| 128 | lbfgs | 0.55 s | 0.20 s |
| 256 | raf | 2.79 s | 0.13 s |
| 256 | lbfgs | 1.89 s | 0.21 s |

**Least-squares phase unwrapping (spidered pupil)**

| size | CPU (double) | GPU (single) |
|---|---|---|
| 128 | 35.5 ms | 79.3 ms |
| 256 | 130 ms | 105 ms |
| 512 | 482 ms | 164 ms |

**focal 128x128, 36 modes, 2 images**

| method | time | wavefront error |
|---|---|---|
| solvephase LM (cpu) | 0.64 s | 0.13 nm |
| solvephase LM (gpu) | 0.10 s | 0.13 nm |
| HCIPy + SciPy least_squares | 2.94 s | 0.19 nm |
| HCIPy + SciPy L-BFGS-B | 3.09 s | 0.12 nm |

**Misell/GS 128x128, 2 images (iterations/s)**

| method | iterations/s | speed-up |
|---|---|---|
| NumPy textbook loop | 97 | 1.0x |
| solvephase (cpu) | 171 | 1.8x |
| solvephase (gpu) | 1,209 | 12.5x |

**CDI HIO 256x256 (iterations/s)**

| method | iterations/s | speed-up |
|---|---|---|
| NumPy textbook loop | 153 | 1.0x |
| solvephase (cpu) | 408 | 2.7x |
| solvephase (cpu, 16 starts, per start) | 523 | 3.4x |
| solvephase (gpu) | 2,634 | 17.2x |
| solvephase (gpu, 16 starts, per start) | 26,677 | 174.3x |



## Small-problem speed-ups in 0.2.0

solvephase 0.2.0 cuts the host-side overhead that bounds small GPU problems
(fused kernels, CUDA-graph replay of Fast & Furious steps, fewer scalar
products and transfers in L-BFGS and Levenberg-Marquardt) and some CPU
elementwise work. Results are bitwise identical to 0.1.0 on this machine.
Measured on the Arm workstation above: 0.1.0 and 0.2.0 run alternately,
three times each on the same 12 pinned cores (12 BLAS/FFT threads) and the
same RTX 4060, on a shared host; medians.

**RTX 4060 (single precision)**

| case | 0.1.0 | 0.2.0 | speed-up |
|---|---|---|---|
| Fast & Furious step, 64² | 1.41 ms | 0.11 ms | 12.4x |
| Fast & Furious step, 128² | 1.44 ms | 0.13 ms | 10.7x |
| Fast & Furious step, 256² | 1.55 ms | 0.27 ms | 5.8x |
| Objective + gradient, 128² (1 / 5 wavelengths) | 2.3 / 2.6 ms | 1.3 / 1.8 ms | 1.8x / 1.5x |
| Focal LM 64² / 128² / 256², 36 modes | 69 / 65 / 225 ms | 42 / 41 / 164 ms | 1.7x / 1.6x / 1.4x |
| LIFT estimate, 32² / 128² | 76 / 72 ms | 42 / 41 ms | 1.8x / 1.8x |
| Phase diversity solve, 64² / 128² | 0.40 / 0.52 s | 0.27 / 0.38 s | 1.5x / 1.4x |
| Coded diffraction, L-BFGS, 64² | 0.24 s | 0.19 s | 1.3x |
| Misell/GS, 128² (it/s) | 1,051 | 1,232 | 1.2x |
| `retrieve()`, 64² / 128² | 0.53 / 0.68 s | 0.45 / 0.57 s | 1.2x / 1.2x |

**Neoverse-N1, 12 cores (double precision)**

| case | 0.1.0 | 0.2.0 | speed-up |
|---|---|---|---|
| Misell/GS, 128² / 256² / 512² (it/s) | 128 / 33 / 7.1 | 151 / 40 / 8.8 | 1.2x |
| Objective + gradient, 256² | 18.7 ms | 16.7 ms | 1.1x |
| Focal LM 128² / 256², 36 modes | 0.69 / 2.10 s | 0.65 / 1.89 s | 1.1x |
| `retrieve()`, 64² | 1.09 s | 1.01 s | 1.1x |

Other CPU cases are unchanged within the noise of the shared host: CPU
focal-plane solves spend most of their time in the FFTs.

## Reading the numbers

- Levenberg-Marquardt needs about 5–10 iterations, so focal-plane time to
  solution is dominated by building Gauss-Newton Jacobians (one
  forward-mode propagation per mode and channel).
- GPU speed-ups grow with problem size. Small solves (≤ 64×64) are bound
  by kernel-launch latency, and the CPU is as fast; Fast & Furious, which
  replays its steps from CUDA graphs, is the exception. See
  [GPU and performance](guide/performance.md).
- Batched multi-start CDI advances all starts with one batched FFT, so the
  throughput per start rises with the batch size.
