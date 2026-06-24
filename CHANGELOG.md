# Changelog

## [Unreleased]

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

