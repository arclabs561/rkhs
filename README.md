# rkhs

[![crates.io](https://img.shields.io/crates/v/rkhs.svg)](https://crates.io/crates/rkhs)
[![Documentation](https://docs.rs/rkhs/badge.svg)](https://docs.rs/rkhs)

Kernel methods.

`rkhs` provides common kernel functions, Gram matrices, MMD two-sample tests,
quantile and distribution kernels, and a kernel ridge regression memory.

See [examples/README.md](examples/README.md) for runnable examples.

## Quickstart

```toml
[dependencies]
rkhs = "0.3.1"
```

```rust
use rkhs::{mmd_permutation_test_seeded, rbf};

let x = vec![vec![0.0, 0.0], vec![0.1, 0.1], vec![0.2, 0.0]];
let y = vec![vec![5.0, 5.0], vec![5.1, 5.1], vec![5.2, 5.0]];

// Unbiased MMD estimate and a permutation-test p-value (999 shuffles, seed 0).
let (mmd, p_value) = mmd_permutation_test_seeded(&x, &y, |a, b| rbf(a, b, 1.0), 999, 0);
println!("mmd = {mmd:.3}, p = {p_value:.3}");
assert!((mmd - 1.974).abs() < 1e-3);
```

```text
mmd = 1.974, p = 0.085
```

Three points per side allow only 20 distinct splits, two of which are as
extreme as the observed one, so p stays near 0.1 however far apart the
samples are.

## Methods

Kernels: `rbf`, `laplacian`, `polynomial`, `linear`, `epanechnikov`,
`matern_12` / `matern_32` / `matern_52`, `triangle`, `cosine`, `quartic`,
`triweight`, `tricube`.

Gram matrices: `kernel_matrix`, `rbf_kernel_matrix_ndarray`.

MMD two-sample tests: `mmd_biased`, `mmd_unbiased`, `mmd_permutation_test`,
`mmd_permutation_test_seeded` (fixed RNG seed for reproducible p-values),
`median_bandwidth` (bandwidth heuristic). `tests/permutation_null.rs` checks
that the permutation test rejects about 5% of seeded same-distribution
replicates at level 0.05.

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
kernel mean embeddings. Given samples from P and Q, the permutation test asks
whether P = Q.

## License

Dual-licensed under MIT or Apache-2.0.
