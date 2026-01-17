# rkhs

Reproducing Kernel Hilbert Space primitives for non-parametric statistical testing and kernel methods.
Implements kernel matrices, Maximum Mean Discrepancy (MMD), Random Fourier Features, and Nystrom approximation.

Dual-licensed under MIT or Apache-2.0.

[crates.io](https://crates.io/crates/rkhs) | [docs.rs](https://docs.rs/rkhs)

```rust
use rkhs::{rbf, mmd_unbiased, mmd_permutation_test};

let x = vec![vec![0.0, 0.0], vec![0.1, 0.1], vec![0.2, 0.0]];
let y = vec![vec![5.0, 5.0], vec![5.1, 5.1], vec![5.2, 5.0]];

// MMD: kernel distance between distributions
let mmd = mmd_unbiased(&x, &y, |a, b| rbf(a, b, 1.0));

// Permutation test for significance
let (_, p_value) = mmd_permutation_test(&x, &y, |a, b| rbf(a, b, 1.0), 1000);
```

## Functions

| Function | Purpose |
|----------|---------|
| `rbf` | Gaussian/RBF kernel |
| `polynomial` | Polynomial kernel |
| `kernel_matrix` | n x n Gram matrix |
| `mmd_unbiased` | Unbiased MMD U-statistic |
| `mmd_permutation_test` | Two-sample test with p-value |
| `median_bandwidth` | Bandwidth selection heuristic |
| `nystrom_approximation` | Low-rank kernel approximation |
| `random_fourier_features` | Explicit feature map for RBF |

## Why MMD

MMD (Maximum Mean Discrepancy) measures distance between distributions using
kernel mean embeddings. Given samples from P and Q, it tests whether P = Q.

- Two-sample testing (detect distribution shift)
- Domain adaptation (minimize source/target divergence)
- GAN evaluation
- Model criticism

## Why "rkhs"

Every positive-definite kernel k(x,y) uniquely defines a Reproducing Kernel
Hilbert Space (Moore-Aronszajn theorem). MMD, kernel PCA, SVM, Gaussian
processes—all operate in this space. The name reflects the unifying structure.

## Connections

- [`logp`](../logp): KL/JS for discrete distributions; MMD for continuous
- [`wass`](../wass): Wasserstein needs ground metric; MMD needs kernel
- [`lapl`](../lapl): Kernel → similarity graph → Laplacian
