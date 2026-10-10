# Changelog

## [Unreleased]

## [0.3.1] - 2026-10-09

### Added

- `kqd` module with `gaussian_ekqd_from_draws`, `GaussianEkqdConfig`,
  `KqdWeights` and `KqdError`: the empirical Gaussian expected kernel
  quantile discrepancy (e-KQD) of Naslidnyk, Chau, Briol and Muandet. Callers
  supply the landmarks and Gaussian coefficients, so results are
  reproducible.
- `mmd_permutation_test_seeded`, a seeded MMD permutation test with
  reproducible p-values. `mmd_permutation_test` now draws a seed and calls it.
- Declare `rust-version = "1.75"`.

### Changed

- Point kernels (`rbf`, `polynomial`, `linear`, `laplacian`, the Matern and
  KDE-family kernels) and the distribution kernels panic when the two inputs
  have different lengths. The distribution kernels previously checked this
  only in debug builds. `median_bandwidth` panics on ragged data.
- `KrrMemory::train` returns `None` for non-finite patterns, `sigma` that is
  not finite and positive, or `lambda` that is not finite and non-negative.
  `KrrMemory::step` and `KrrMemory::retrieve` panic on a state of the wrong
  dimension or with non-finite values.
- `QuantileWeight::TailHeavy` panics on a negative or non-finite `alpha`, and
  custom weights must be finite and non-negative with a finite positive sum.
  `TailHeavy` no longer clamps near the boundary.
- The `quantile_kernel` docs describe `kernel_quantile_embedding`, `qmmd` and
  `weighted_qmmd` as empirical lower-tail (truncated-MMD) constructions, not
  estimators of Naslidnyk et al.'s kernel quantile embeddings.

### Fixed

- `qmmd` and `weighted_qmmd` no longer skip quantile levels whose truncated
  samples have fewer than two points, so their values can differ from 0.3.0
  at low quantile levels.
- `qmmd`, `weighted_qmmd`, the quantile embeddings and the sliced-Wasserstein
  graph kernel sort with `total_cmp` and no longer panic on a NaN sample.

## [0.3.0] - 2026-06-24

### Added

- `KrrMemory`: a Kernel Ridge Regression Hopfield associative memory
  (arXiv:2504.12561). Trains dual coefficients by the closed-form solve
  `(K + lambda I) alpha = X` (Cholesky, no linear-algebra backend) and retrieves
  by iterating `sign(k_s . alpha)`. Stores far above the classical Hebbian limit
  (~0.14 N): a test verifies `P/N = 1.0` patterns are exact fixed points and a
  one-bit flip is corrected at `P/N = 0.5`.
- Matérn kernels `matern_12`, `matern_32`, `matern_52` (the half-integer
  smoothness cases ν = 1/2, 3/2, 5/2; closed-form, Euclidean distance). ν = 1/2
  is the exponential kernel (distinct from the L1 `laplacian`); higher ν is
  smoother, approaching `rbf` as ν → ∞.

## [0.2.2] - 2026-06-10

### Changed

- Associative-memory functions are now re-exported from `hopfield` instead of implemented locally; public paths unchanged.

### Added

- Modern Hopfield Networks hero example in crate docs.

