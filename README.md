# rkhs

[![crates.io](https://img.shields.io/crates/v/rkhs.svg)](https://crates.io/crates/rkhs)
[![Documentation](https://docs.rs/rkhs/badge.svg)](https://docs.rs/rkhs)

Kernel methods.

`rkhs` provides positive-definite kernels, Gram matrices, MMD two-sample tests,
quantile and distribution kernels, and a kernel ridge regression memory.

See [examples/README.md](examples/README.md) for runnable examples covering
MMD, kernel thinning, Gram matrix spectra, the KRR-Hopfield associative memory
(`associative_memory`, `clam_clustering`), and Hopfield energy functions
re-exported from `hopfield`.

## Quickstart

```toml
[dependencies]
rkhs = "0.3.0"
```

```rust
use rkhs::{rbf, mmd_unbiased, mmd_permutation_test};

let x = vec![vec![0.0, 0.0], vec![0.1, 0.1], vec![0.2, 0.0]];
let y = vec![vec![5.0, 5.0], vec![5.1, 5.1], vec![5.2, 5.0]];

// MMD: kernel distance between distributions
let mmd = mmd_unbiased(&x, &y, |a, b| rbf(a, b, 1.0));

// Permutation test for significance
let (_, p_value) = mmd_permutation_test(&x, &y, |a, b| rbf(a, b, 1.0), 1000);
```

## Methods

Kernels: `rbf`, `laplacian`, `polynomial`, `linear`, `epanechnikov`,
`matern_12` / `matern_32` / `matern_52`, `triangle`, `cosine`, `quartic`,
`triweight`, `tricube`.

Gram matrices: `kernel_matrix`, `rbf_kernel_matrix_ndarray`.

MMD two-sample tests: `mmd_biased`, `mmd_unbiased`, `mmd_permutation_test`,
`median_bandwidth` (bandwidth heuristic).

Quantile kernels (`quantile_kernel`): `qmmd`, `weighted_qmmd`,
`kernel_quantile_embedding`, `quantile_function_embedding`,
`quantile_distribution_kernel`, `quantile_gram_matrix`.

Distribution kernels (`distribution_kernel`): `jensen_shannon_kernel`,
`probability_product_kernel`, `expected_likelihood_kernel`,
`fisher_kernel_categorical`.

Associative memory: `KrrMemory` (kernel ridge regression; `train` / `step` /
`retrieve`), plus `energy_lse`, `energy_lsr`, and `retrieve_memory` re-exported
from `hopfield`.

Deprecated: `clam` (use `clump::clam`), `graph_kernel` (use `graphops`).

## Why MMD

MMD (Maximum Mean Discrepancy) measures distance between distributions using
kernel mean embeddings. Given samples from P and Q, it tests whether P = Q.

- Two-sample testing (detect distribution shift)
- Domain adaptation (minimize source/target divergence)
- GAN evaluation
- Model criticism

## Why "rkhs"

Every positive-definite kernel k(x,y) uniquely defines a Reproducing Kernel
Hilbert Space (Moore-Aronszajn theorem). MMD, kernel PCA, SVM, and Gaussian
processes all operate in this space. The name reflects the unifying structure.

Dual-licensed under MIT or Apache-2.0.
