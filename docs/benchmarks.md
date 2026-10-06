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

## Reading the numbers

- Levenberg-Marquardt needs about 5–10 iterations, so focal-plane time to
  solution is dominated by building Gauss-Newton Jacobians (one
  forward-mode propagation per mode and channel).
- GPU speed-ups grow with problem size. Small sensors (≤ 64×64) are bound
  by kernel-launch latency, and the CPU is as fast. See
  [GPU and performance](guide/performance.md).
- Batched multi-start CDI advances all starts with one batched FFT, so the
  throughput per start rises with the batch size.
