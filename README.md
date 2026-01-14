# rkhs

Reproducing Kernel Hilbert Space primitives: kernels, MMD, random Fourier features.

The name comes from the mathematical structure underlying all kernel methods:
a **Reproducing Kernel Hilbert Space** is a function space where evaluation
at any point is a continuous linear functional (the "reproducing" property).

Dual-licensed under MIT or Apache-2.0.

```rust
use rkhs::{rbf, mmd_unbiased, mmd_permutation_test};

let x = vec![vec![0.0, 0.0], vec![0.1, 0.1], vec![0.2, 0.0]];
let y = vec![vec![5.0, 5.0], vec![5.1, 5.1], vec![5.2, 5.0]];

// Unbiased MMD estimate
let mmd = mmd_unbiased(&x, &y, |a, b| rbf(a, b, 1.0));

// Permutation test for significance
let (_, p_value) = mmd_permutation_test(&x, &y, |a, b| rbf(a, b, 1.0), 1000);
```

## Functions

| Function | Purpose |
|----------|---------|
| `rbf(x, y, sigma)` | Gaussian/RBF kernel: exp(-||x-y||^2 / 2sigma^2) |
| `polynomial(x, y, degree, bias)` | Polynomial kernel: (x . y + bias)^degree |
| `kernel_matrix(data, kernel)` | Compute n x n Gram matrix |
| `mmd_biased(x, y, kernel)` | Biased MMD estimate (O(n^2)) |
| `mmd_unbiased(x, y, kernel)` | Unbiased U-statistic (O(n^2)) |
| `mmd_permutation_test(x, y, kernel, n)` | Hypothesis test with p-value |
| `median_bandwidth(data)` | Median heuristic for kernel bandwidth |
| `nystrom_approximation(data, landmarks, kernel)` | Low-rank approximation |
| `random_fourier_features(data, n_features, sigma)` | Explicit feature map |

## Why MMD

MMD (Maximum Mean Discrepancy) is a kernel-based distance between probability
distributions. Given samples from P and Q, MMD tests whether P = Q.

Uses:
- GAN evaluation (FID-like statistics)
- Domain adaptation (minimize MMD between source/target)
- Two-sample testing (detect distribution shift)
- Model criticism (compare model vs data)

## References

- Gretton et al. (2012). "A Kernel Two-Sample Test" (JMLR)
- Muandet et al. (2017). "Kernel Mean Embedding of Distributions"
- Rahimi & Recht (2007). "Random Features for Large-Scale Kernel Machines"
