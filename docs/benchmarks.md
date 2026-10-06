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

<!-- COMPARE TABLE -->

## Suite

<!-- SUITE TABLE -->

## Reading the numbers

- Levenberg-Marquardt needs about 5–10 iterations, so focal-plane time to
  solution is dominated by building Gauss-Newton Jacobians (one
  forward-mode propagation per mode and channel).
- GPU speed-ups grow with problem size. Small sensors (≤ 64×64) are bound
  by kernel-launch latency, and the CPU is as fast. See
  [GPU and performance](guide/performance.md).
- Batched multi-start CDI advances all starts with one batched FFT, so the
  throughput per start rises with the batch size.
