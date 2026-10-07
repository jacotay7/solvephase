# GPU and performance

## Devices

Every solver takes `device=` (`"cpu"`, `"gpu"`, `"auto"`) and `precision=`
(`"single"`, `"double"`). The defaults are double on CPU and single on GPU.

```python
import solvephase as sp

be = sp.get_backend("auto")
print(be, sp.gpu_available())
```

Arrays returned from a GPU solve stay on the GPU. Pass them straight into
the next GPU call, and use `sp.to_numpy` only when you need host data.

## When the GPU wins

A GPU iteration costs a roughly fixed launch overhead (tens of microseconds
per kernel) plus work proportional to the problem size, so the GPU pays off
for large problems:

- **Focal-plane retrieval**: from about 128×128 pupils and images, and
  always for broadband (matrix-Fourier) models, where dense batched matmuls
  run near peak.
- **CDI**: from about 256×256, and especially with batched multi-start
  (`starts=16`), where one batched FFT advances every start.
- **Small solves** (LIFT on 32×32, Levenberg-Marquardt on 64×64) are
  latency bound: they take tens of milliseconds on any GPU, and the CPU is as
  fast or faster unless the data already live on the GPU.
- **Fast & Furious** replays each step from a CUDA graph, so a 64×64 or
  128×128 step takes about 0.1 ms on the GPU, several times faster than on
  the CPU.

The [Benchmarks](../benchmarks.md) page has the measured crossover on
reference hardware; run `python benchmarks/run.py` to measure yours.

## What makes solvephase fast

- **Exact, hand-written adjoints.** One reverse propagation per channel and
  wavelength gives the full gradient, with no autodiff graph to build or
  store. Gauss-Newton Jacobians use forward-mode propagation in batches sized
  to a memory budget.
- **Second-order methods where they pay.** Levenberg-Marquardt converges in
  5–10 iterations on modal problems where first-order methods need hundreds.
- **The cheapest propagator.** Each model chooses between a zero-padded FFT
  and a matrix Fourier transform (two small matmuls, ideal for cropped or
  undersampled windows and for broadband stacks).
- **Precision where it matters.** Propagation runs in the working precision,
  but likelihoods and normal equations accumulate in float64. Single
  precision is therefore safe and about twice as fast.
- **CPU threading that degrades gracefully.** FFT thread counts scale with
  transform size, and scalar products avoid threaded BLAS level-1 calls,
  which stall when the cores are busy.
- **Few host synchronizations.** Iterative projection algorithms check
  convergence only every `check_every` iterations.
- **Few kernel launches.** On small GPU problems the time goes to launching
  kernels, not to the arithmetic. The model, its derivatives, the losses and
  the Gerchberg-Saxton and Fast & Furious updates run as fused kernels that
  compute exactly what the array expressions compute, and Fast & Furious
  steps replay a captured CUDA graph.

## Tips

- Set `AOCORE_FFT_WORKERS=1` (and `AOCORE_BLAS_THREADS=1`) when running many solves in parallel
  processes.
- Reuse models and problems in loops: building a `FocalPlaneModel` precomputes
  propagation matrices and modulations.
- Use `method="lm"` for modal problems. Switch to L-BFGS for zonal problems
  or very many modes.
- For a zonal wavefront, start from a modal solution
  (`retrieve(..., zonal_refinement=True)`).
