# Changelog

## [Unreleased]

### Added

- Matérn kernels `matern_12`, `matern_32`, `matern_52` (the half-integer
  smoothness cases ν = 1/2, 3/2, 5/2; closed-form, Euclidean distance). ν = 1/2
  is the exponential kernel (distinct from the L1 `laplacian`); higher ν is
  smoother, approaching `rbf` as ν → ∞.

## [0.2.2] - 2026-06-10

### Changed

- Associative-memory functions are now re-exported from `hopfield` instead of implemented locally; public paths unchanged.

### Added

- Modern Hopfield Networks hero example in crate docs.

