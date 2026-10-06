# Generic measurement models (Wirtinger flow family)

Most of `solvephase` is about optics: a pupil, a propagator, a detector. This
page covers the abstract problem underneath: recover a complex vector
$x \in \mathbb{C}^n$ from $m$ phaseless linear measurements

$$
y_k = |\langle a_k, x\rangle|^2 = |(Ax)_k|^2, \qquad k = 1, \dots, m,
$$

for a known linear operator $A$ whose rows are $a_k^*$. The solution is
determined at best up to a global phase $e^{i\phi}x$. The algorithms here are
the non-convex gradient methods catalogued by PhasePack (Chandra et al.
2017): Wirtinger Flow, Truncated Wirtinger Flow, Truncated and Reweighted
Amplitude Flow, and L-BFGS, each started from a spectral-type initialization.

```python
import numpy as np

from solvephase.algorithms.wirtinger import (
    GenericResult,
    init_optimal_spectral,
    init_orthogonality_promoting,
    init_random,
    init_spectral,
    init_truncated_spectral,
    init_weighted_correlation,
    relative_error,
    wirtinger,
)
from solvephase.operators import (
    CodedDiffractionOperator,
    LinearOperator,
    MatrixOperator,
    OversampledFourierOperator,
)
```

## Quick start

```python
n, m = 64, 8 * 64
rng = np.random.default_rng(0)
x_true = rng.standard_normal(n) + 1j * rng.standard_normal(n)

A = MatrixOperator.gaussian(m, n, seed=1)        # i.i.d. complex Gaussian rows
y = np.abs(A.forward(x_true)) ** 2               # intensities |A x|^2

result = wirtinger(A, y, method="raf", seed=0)
print(result.message, result.n_iter)
print(f"relative error {relative_error(result.x, x_true):.1e}")
assert relative_error(result.x, x_true) < 1e-6
```

`wirtinger` returns a `GenericResult` with the estimate `x` (on the
operator's device), the relative amplitude residual
$\|\,|Ax| - \sqrt{y}\,\| / \|\sqrt{y}\|$ at every check (`history`, with
`times`), `n_iter`, `converged`, `message`, `elapsed`, `device`, `method` and
the `init` actually used. `result.to_numpy()` copies `x` to the host.
`relative_error(x, truth)` is the distance modulo the global phase,
$\min_\phi \|x - e^{i\phi}x_\star\| / \|x_\star\|$, with the optimal phase
$e^{i\phi} = x_\star^* x / |x_\star^* x|$.

## Measurement operators

An operator (`solvephase.operators`) is a `LinearOperator` with `forward`,
an exact `adjoint` (tested by $\langle Ax, y\rangle = \langle x, A^H y\rangle$
in single and double precision, CPU and GPU), `in_shape`/`out_shape`,
`norm_estimate()` (power iteration for $\|A\|_2$),
`frobenius_norm_squared()` and `gram_diagonal()` ($\operatorname{diag}(A^H
A)$). Leading batch axes are allowed in `forward` and `adjoint`.

| Operator | Model | `in_shape` → `out_shape` |
|---|---|---|
| `MatrixOperator(M)` | dense $y = \lvert Mx\rvert^2$; `MatrixOperator.gaussian(m, n, seed=)` draws $M_{kj} \sim \mathcal{CN}(0, 1)$ | `(n,)` → `(m,)` |
| `CodedDiffractionOperator(masks)` | $y_l = \lvert\mathcal{F}(d_l \odot x)\rvert^2$, $L$ masks, unitary FFT batched over $l$; `.random(shape, L, kind="octanary" \| "uniform", seed=)` | `(ny, nx)` → `(L, ny, nx)` |
| `OversampledFourierOperator(shape, oversampling)` | zero padding then unitary FFT | `(ny, nx)` → padded grid |

**Octanary masks** (Candès, Li & Soltanolkotabi 2015b) are $d = b_1 b_2$ with
$b_1$ uniform on $\{1, -1, i, -i\}$ and $b_2 = 1/\sqrt{2}$ with probability
4/5, $\sqrt{3}$ with probability 1/5, so $\mathbb{E}|d|^2 = 1$.

**Fourier magnitudes alone** (`OversampledFourierOperator`) are a different
problem: besides the global phase they cannot distinguish shifts and the
conjugate flip $\overline{x(-r)}$, and the Wirtinger-flow methods rarely
converge without a support constraint. Use the projection algorithms of
[Coherent diffraction imaging](cdi.md) for that model, or add masks.

Your own operator only needs `backend`, `in_shape`, `out_shape`, `forward`
and `adjoint`; the Frobenius norm then falls back to a Hutchinson estimate:

```python
class Scaled(LinearOperator):
    """3 A, without exact norms: exercises the generic fallbacks."""

    def __init__(self, inner):
        self.inner, self.backend = inner, inner.backend
        self.in_shape, self.out_shape = inner.in_shape, inner.out_shape

    def forward(self, x):
        return 3.0 * self.inner.forward(x)

    def adjoint(self, v):
        return 3.0 * self.inner.adjoint(v)


custom = wirtinger(Scaled(A), 9.0 * y, method="taf", seed=0)
assert relative_error(custom.x, x_true) < 1e-5
```

### Scaling convention

Step sizes and truncation thresholds in the literature are stated for the
unit-variance Gaussian model, where $\mathbb{E}\,y_k = \|x\|^2$ and
$\mathbb{E}\,A^HA = mI$. `wirtinger` rescales any operator to that convention,
$A \to \sqrt{s}A$ and $y \to s\,y$ with $s = mn/\|A\|_F^2$, so the published
parameters apply unchanged (for coded diffraction with a unitary FFT this
gives back the unnormalized FFT of the CDP papers). Below, $A$ and $y$ are the
rescaled quantities, $u = Az$, and $\psi = \sqrt{y}$. The norm of $x$ is
estimated as $\lambda_0^2 = \frac1m\sum_k y_k$.

## Initializations

Every spectral-type initialization takes the leading eigenvector $v$ of a
weighted covariance

$$
Y_w = \frac1m \sum_{k=1}^m w_k\, a_k a_k^* = \frac1m A^H \operatorname{diag}(w) A
$$

by the power method (100 iterations by default, stopped when the Rayleigh
quotient settles), and returns $z_0 = \lambda_0 v$. Seeds drive the random
power-method start on the host, so CPU and GPU give the same start.

| `init=` | weights $w_k$ | reference |
|---|---|---|
| `"spectral"` | $y_k$ | Netrapalli et al. 2013; Candès et al. 2015a |
| `"truncated"` | $y_k\,\mathbb{1}\{y_k \le \alpha_y^2\lambda_0^2\}$, $\alpha_y = 3$ | Chen & Candès 2017 |
| `"orthogonal"` | $\mathbb{1}\{k \in S\}$, $S$ = the $\lceil m/6\rceil$ largest $\psi_k$ | Wang, Giannakis & Eldar 2018 |
| `"weighted"` | $\psi_k^{1/2}\,\mathbb{1}\{k \in S\}$, $S$ = the $\lceil 3m/13\rceil$ largest $\psi_k$ | Wang, Giannakis, Saad & Chen 2018 |
| `"optimal"` | $\mathcal{T}(\bar y_k) = \dfrac{\bar y_k - 1}{\bar y_k + \sqrt{\delta} - 1}$, $\bar y = y/\lambda_0^2$, $\delta = m/n$ | Luo, Alghamdi & Lu 2019 |
| `"random"` | none: $z_0 = \lambda_0 g/\|g\|$, $g \sim \mathcal{CN}(0, I)$ | |

* The plain spectral matrix has $\mathbb{E}\,Y_y = \|x\|^2 I + xx^*$, but the
  heavy tail of $y$ makes it noisy; truncation removes the few largest $y_k$.
* The orthogonality-promoting and weighted-correlation starts keep only the
  measurements most aligned with $x$ (largest $\psi_k$), whose vectors $a_k$
  cluster around $x$. The papers also normalize each $a_k$ by its norm; the
  Gaussian and CDP rows have nearly equal norms, so `solvephase` omits that.
* Luo et al.'s $\mathcal{T}$ asymptotically minimizes the sampling ratio
  $\delta$ at which the estimate correlates with $x$. It is negative for
  $\bar y < 1$, so the power method runs on $Y_\mathcal{T} + cI$ with
  $c = 1.05\,\|A\|_2^2/(m(\sqrt\delta - 1))$, which is positive semidefinite;
  it needs $m > n$ and more iterations (200 by default).
* **Whitening.** When the operator reports $G = \operatorname{diag}(A^HA)$
  (all built-in ones do), the power method runs on $G^{-1/2}Y_wG^{-1/2}$ and
  maps the eigenvector back with $G^{1/2}$. For Gaussian rows $G \approx mI$
  and nothing changes. For octanary CDP masks it is essential beyond about
  $10^4$ pixels: pixels where most masks have $|d|^2 = 3$ create localized
  eigenvectors that otherwise beat the signal (cosine similarity 0.03 instead
  of 0.83 for a 256 × 256 image with 8 masks).

Measured cosine similarity $|v^*x|/\|x\|$ for the Gaussian model, $n = 64$
(mean of 8 problems):

| $m/n$ | spectral | truncated | orthogonal | weighted | optimal |
|---|---|---|---|---|---|
| 2 | 0.40 | 0.40 | 0.39 | 0.42 | 0.60 |
| 4 | 0.56 | 0.56 | 0.59 | 0.61 | 0.79 |
| 8 | 0.69 | 0.69 | 0.74 | 0.75 | 0.88 |
| 16 | 0.84 | 0.84 | 0.86 | 0.86 | 0.94 |

The initialization functions are public, and any array can be passed as
`init=` (the result then reports `init="given"`):

```python
z0 = init_optimal_spectral(A, y, seed=0)
cosine = abs(np.vdot(z0, x_true)) / (np.linalg.norm(z0) * np.linalg.norm(x_true))
print(f"optimal-preprocessing start: cosine similarity {cosine:.2f}")
from_start = wirtinger(A, y, method="taf", init=z0)
assert from_start.init == "given"
```

## Algorithms

Each iteration costs one `forward` and one `adjoint`. With $u = Az$:

**Wirtinger Flow** (`"wf"`, Candès, Li & Soltanolkotabi 2015a) is gradient
descent on the intensity loss $f(z) = \frac{1}{2m}\sum_k (|u_k|^2 - y_k)^2$:

$$
z \leftarrow z - \frac{\mu_t}{\|z_0\|^2}\,\frac1m A^H\!\left[(|u|^2 - y)\odot u\right],
\qquad \mu_t = \min\!\left(1 - e^{-t/\tau_0},\ \mu_{\max}\right),\ \tau_0 = 330.
$$

The paper uses $\mu_{\max} = 0.2$; in our tests that diverges or stalls on
every Gaussian instance with $n \ge 128$ that we tried (the reference code
transcribed directly does the same), so the default is $\mu_{\max} = 0.1$
(`step=` overrides it; 0.05 was always stable but twice as slow).

**Truncated Wirtinger Flow** (`"twf"`, Chen & Candès 2017) ascends the
Poisson log-likelihood, keeping only well-behaved measurements:

$$
z \leftarrow z + \frac{2\mu}{m} A^H\!\left[\mathbb{1}_{\mathcal{E}}\,\frac{y - |u|^2}{|u|^2}\odot u\right],
\quad
\mathcal{E} = \left\{\alpha_{lb} \le \tfrac{|u_k|}{\|z\|} \le \alpha_{ub},\ 
|y_k - |u_k|^2| \le \alpha_h\,\overline{|y - |u|^2|}\,\tfrac{|u_k|}{\|z\|}\right\}
$$

with $\mu = 0.2$, $\alpha_{lb} = 0.3$, $\alpha_{ub} = \alpha_h = 5$.

**Truncated Amplitude Flow** (`"taf"`, Wang, Giannakis & Eldar 2018) descends
the amplitude loss $\frac{1}{2m}\sum_k (|u_k| - \psi_k)^2$, dropping
measurements whose current amplitude is far below the data:

$$
z \leftarrow z - \frac{\mu}{m} A^H\!\left[\mathbb{1}\{|u| \ge \psi/(1+\gamma)\}\odot\left(u - \psi\odot\frac{u}{|u|}\right)\right],
\qquad \gamma = 0.7,\ \mu = 1.
$$

**Reweighted Amplitude Flow** (`"raf"`, Wang, Giannakis, Saad & Chen 2018)
replaces the hard truncation by smooth weights:

$$
z \leftarrow z - \frac{\mu}{m} A^H\!\left[w \odot \left(u - \psi\odot\frac{u}{|u|}\right)\right],
\qquad w_k = \frac{|u_k|}{|u_k| + \beta\psi_k},\ \beta = 5,\ \mu = 4.
$$

**L-BFGS** (`"lbfgs"`) minimizes the amplitude loss (default) or, with
`loss="intensity"`, $\frac{1}{4m}\sum_k(|u_k|^2 - y_k)^2$, with
`solvephase.optimize.lbfgs` (strong-Wolfe line search) on the real view
$(\operatorname{Re}z, \operatorname{Im}z)$, whose gradient is
$2\,\partial f/\partial\bar z$, for the amplitude loss
$\frac1m A^H[u - \psi\odot u/|u|]$. PhasePack's benchmarks found
quasi-Newton methods the fastest in practice; ours agree (below).

The TAF and RAF step sizes were tuned on the Gaussian model ($n = 64$,
$m/n$ = 3 to 8, 10 problems each) for the best success rate and speed; all
parameters can be changed through `step=` and
`options={"tau0", "alpha_lb", "alpha_ub", "alpha_h", "gamma", "beta", "memory"}`.

### Stopping and devices

Every `check_every` iterations (default 10) the relative amplitude residual
$r$ is transferred to the host. The run stops when $r$ is below `tol` (an
exact fit), when $r$ changed by less than `tol` relative to the previous check
(a stationary point, the usual stop with noisy data), when the
`callback(iteration, x, residual)` returns `True`, or at `iterations`. `tol`
defaults to $10^{-7}$ and is raised to $8\epsilon$ of the working precision
(about $10^{-6}$ in single precision). Near a minimizer with residual $r_\star$
the change of $r$ is quadratic in the distance to it, so the stagnation test
fires only once the remaining optimization error is far below the
noise-induced error; a test on the change of $x$ itself would stop slowly
contracting methods such as WF early. The checks are the only host syncs; on
the GPU the whole solve stays on the device. A non-finite residual stops the
run with a "diverged" message: reduce `step`.

## How many measurements, and which method

Complex $x \in \mathbb{C}^n$ needs $m \ge 4n - 4$ generic measurements for the
intensity map to be injective (Conca et al. 2015); the theory for these
algorithms needs $m = O(n)$ (TWF, TAF, RAF) or $O(n\log n)$ (WF). Measured
success rates on the Gaussian model (10 problems per cell, relative error
below $10^{-5}$ within 3000 iterations, default settings):

| $n$ | $m/n$ | WF | TWF | TAF | RAF | L-BFGS (amplitude) | L-BFGS (intensity) |
|---|---|---|---|---|---|---|---|
| 64 | 3 | 0 | 6 | 6 | 8 | 10 | 10 |
| 128 | 3 | 0 | 6 | 5 | 7 | 10 | 9 |
| 64, 128 | 4.5, 6, 8 | 10 | 10 | 10 | 10 | 10 | 10 |

For coded diffraction, $L = 6$ to 8 octanary masks is comfortable for every
method; TAF, RAF and L-BFGS still succeed with $L = 4$. Fourier magnitudes
alone (`OversampledFourierOperator`, 16 × 16 image, 2× oversampling) defeat
all five methods (0 of 5 problems).

Iterations to reach relative error $10^{-6}$ (double precision), and wall
time on a Quadro P620 in single precision to $10^{-5}$ (including the
initialization), with each method's default initialization:

| problem | WF | TWF | TAF | RAF | L-BFGS |
|---|---|---|---|---|---|
| Gaussian, $n = 256$, $m = 8n$: iterations | 700 | 135 | 80 | 125 | 25 |
| CDP $256 \times 256$, $L = 8$: iterations | 690 | 135 | 85 | 125 | 25 |
| CDP $256 \times 256$, $L = 8$: GPU time (s) | 1.39 | 0.52 | 0.31 | 0.41 | 0.41 |

Which to pick:

* **`"lbfgs"`** converges in the fewest iterations (20 to 40 on the problems
  above) and is the most robust at low $m/n$; it costs a few extra
  evaluations per iteration in the line search and one host sync per
  evaluation. Usually the fastest on the CPU.
* **`"taf"`** and **`"raf"`** are the fastest first-order methods (100 to 200
  iterations) and need no host syncs between checks, which suits the GPU.
* **`"twf"`** fits the Poisson likelihood: with noisy data it gives about
  the same error as the amplitude methods, slightly lower in some regimes.
* **`"wf"`** is the historical baseline: slowest, and the least stable.

With noise, all methods converge to a point whose error is proportional to
the noise level; for relative intensity noise $y_k(1 + \sigma n_k)$ the
amplitude methods reach about $0.47\sigma$ and WF (intensity loss) about
$0.97\sigma$ on the Gaussian model with $m = 8n$.

## Coded diffraction example

```python
img_rng = np.random.default_rng(1)
img = img_rng.standard_normal((32, 32)) + 1j * img_rng.standard_normal((32, 32))
cdp = CodedDiffractionOperator.random((32, 32), 6, kind="octanary", seed=3)
patterns = np.abs(cdp.forward(img)) ** 2  # shape (6, 32, 32)

for method in ("taf", "lbfgs"):
    res = wirtinger(cdp, patterns, method=method, seed=0)
    print(f"{method}: {res.n_iter} iterations, error {relative_error(res.x, img):.1e}")
    assert relative_error(res.x, img) < 1e-5
```

On a GPU build the operator on `backend="gpu"` (single precision by default)
and pass the data as a host or device array; the result stays on the GPU:

```py
cdp_gpu = CodedDiffractionOperator.random((256, 256), 8, seed=3, backend="gpu")
res = wirtinger(cdp_gpu, patterns_gpu, method="raf")
x_host = res.to_numpy().x
```

## References

- E. J. Candès, X. Li and M. Soltanolkotabi, "Phase retrieval via Wirtinger
  flow: theory and algorithms," IEEE Trans. Inf. Theory 61, 1985 (2015a).
- E. J. Candès, X. Li and M. Soltanolkotabi, "Phase retrieval from coded
  diffraction patterns," Appl. Comput. Harmon. Anal. 39, 277 (2015b).
- P. Netrapalli, P. Jain and S. Sanghavi, "Phase retrieval using alternating
  minimization," NeurIPS (2013); IEEE Trans. Signal Process. 63, 4814 (2015).
- Y. Chen and E. J. Candès, "Solving random quadratic systems of equations is
  nearly as easy as solving linear systems," Comm. Pure Appl. Math. 70, 822
  (2017).
- G. Wang, G. B. Giannakis and Y. C. Eldar, "Solving systems of random
  quadratic equations via truncated amplitude flow," IEEE Trans. Inf. Theory
  64, 773 (2018).
- G. Wang, G. B. Giannakis, Y. Saad and J. Chen, "Phase retrieval via
  reweighted amplitude flow," IEEE Trans. Signal Process. 66, 2818 (2018).
- W. Luo, W. Alghamdi and Y. M. Lu, "Optimal spectral initialization for
  signal recovery with applications to phase retrieval," IEEE Trans. Signal
  Process. 67, 2347 (2019).
- R. Chandra, Z. Zhong, J. Hontz, V. McCulloch, C. Studer and T. Goldstein,
  "PhasePack: a phase retrieval library," Asilomar Conf. Signals, Systems,
  and Computers (2017).
- A. Conca, D. Edidin, M. Hering and C. Vinzant, "An algebraic
  characterization of injectivity in phase retrieval," Appl. Comput. Harmon.
  Anal. 38, 346 (2015).
