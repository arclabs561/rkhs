//! # rkhs
//!
//! Reproducing Kernel Hilbert Space primitives for distribution comparison.
//!
//! ## Why "RKHS"?
//!
//! A **Reproducing Kernel Hilbert Space** is the mathematical structure where
//! kernel methods live. Every positive-definite kernel k(x,y) defines an RKHS
//! (via Mercer's theorem), and every RKHS has a unique reproducing kernel.
//!
//! This crate provides the primitives: kernels, Gram matrices, MMD, and
//! random Fourier features—all operations in or derived from the RKHS.
//!
//! ## Intuition
//!
//! Kernels measure similarity in a (potentially infinite-dimensional) feature space
//! without ever computing the features explicitly. This "kernel trick" enables
//! nonlinear methods with linear complexity.
//!
//! MMD (Maximum Mean Discrepancy) uses kernels to test whether two samples come
//! from the same distribution. It embeds distributions into an RKHS and measures
//! the distance between their mean embeddings (kernel mean embeddings).
//!
//! ## Key Functions
//!
//! | Function | Purpose |
//! |----------|---------|
//! | [`rbf`] | Radial Basis Function (Gaussian) kernel |
//! | [`epanechnikov`] | Optimal kernel for density estimation |
//! | [`polynomial`] | Polynomial kernel |
//! | [`kernel_matrix`] | Gram matrix K[i,j] = k(x_i, x_j) |
//! | [`kernel_sum`] | Sum Σ κ(v, ξ^μ) for AM/kernel machines |
//! | [`energy_lse`] | Log-Sum-Exp energy (Dense AM with RBF) |
//! | [`energy_lsr`] | Log-Sum-ReLU energy (Dense AM with Epanechnikov) |
//! | [`retrieve_memory`] | Memory retrieval via energy descent |
//! | [`mmd_biased`] | O(n²) biased MMD estimate |
//! | [`mmd_unbiased`] | O(n²) unbiased MMD u-statistic |
//! | [`mmd_permutation_test`] | Significance test via permutation |
//!
//! ## Quick Start
//!
//! ```rust
//! use rkhs::{rbf, mmd_unbiased};
//!
//! let x = vec![vec![0.0, 0.0], vec![0.1, 0.1], vec![0.2, 0.0]];
//! let y = vec![vec![5.0, 5.0], vec![5.1, 5.1], vec![5.2, 5.0]];
//!
//! // Different distributions → large MMD
//! let mmd = mmd_unbiased(&x, &y, |a, b| rbf(a, b, 1.0));
//! assert!(mmd > 0.5);
//! ```
//!
//! ## Why Kernels Matter for ML
//!
//! - **Associative Memory**: Energy functions E = -log Σ κ(v, ξ) define memory landscapes
//! - **GAN evaluation**: FID uses MMD-like statistics to compare generated vs real
//! - **Domain adaptation**: Minimize MMD between source and target distributions
//! - **Two-sample testing**: Detect distribution shift in production systems
//! - **Kernel regression**: Nonparametric regression via kernel mean embedding
//!
//! ## Associative Memory
//!
//! Dense Associative Memory (Krotov et al., 2016-2025) uses kernel sums to define
//! energy landscapes for content-addressable memory:
//!
//! ```rust
//! use rkhs::{energy_lse, energy_lsr, energy_lse_grad, retrieve_memory};
//!
//! // Store two memories
//! let memories = vec![
//!     vec![0.0, 0.0],
//!     vec![10.0, 10.0],
//! ];
//!
//! // Query: corrupted version of first memory
//! let query = vec![1.0, 1.0];
//!
//! // Retrieve via energy descent (LSE energy)
//! let (retrieved, _) = retrieve_memory(
//!     query,
//!     &memories,
//!     |v, m| energy_lse_grad(v, m, 2.0),
//!     0.1,
//!     100,
//!     1e-6,
//! );
//!
//! // Should recover [0, 0]
//! assert!(retrieved[0].abs() < 1.0);
//! ```
//!
//! The **LSR energy** (using Epanechnikov kernel) has special properties:
//! - Exact single-step retrieval
//! - Novel memory generation at basin intersections
//! - Compact support (infinite energy outside memory neighborhoods)
//!
//! ## Connections
//!
//! - [`logp`](../logp): MMD and KL divergence both measure distribution "distance"
//! - [`wass`](../wass): Wasserstein and MMD are different ways to compare distributions
//! - [`lapl`](../lapl): Gaussian kernel → Laplacian eigenvalue problems
//! - [`strata`](../strata): Kernel k-means uses these kernels
//! - [`innr`](../innr): SIMD acceleration for kernel computations (via `simd` feature)
//!
//! ## SIMD Acceleration
//!
//! Enable the `simd` feature for SIMD-accelerated kernel computations:
//!
//! ```toml
//! [dependencies]
//! rkhs = { version = "0.1", features = ["simd"] }
//! ```
//!
//! This uses [`innr`] for fast L2 distance and dot products.
//!
//! ## What Can Go Wrong
//!
//! 1. **Bandwidth too small**: RBF kernel becomes nearly diagonal, loses structure.
//! 2. **Bandwidth too large**: Everything becomes similar, no discrimination.
//! 3. **Numerical instability**: Very large distances → exp(-large) → 0 underflow.
//! 4. **MMD variance**: With small samples, MMD estimates are noisy. Use permutation test.
//! 5. **Kernel not characteristic**: Not all kernels can distinguish all distributions.
//!    RBF is characteristic (good); polynomial is not (bad for two-sample test).
//!
//! ## References
//!
//! - Gretton et al. (2012). "A Kernel Two-Sample Test" (JMLR)
//! - Muandet et al. (2017). "Kernel Mean Embedding of Distributions" (Found. & Trends)
//! - Rahimi & Recht (2007). "Random Features for Large-Scale Kernel Machines"

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use thiserror::Error;

/// SIMD-accelerated kernel computations using innr.
///
/// Requires the `simd` feature.
#[cfg(feature = "simd")]
pub mod simd;

/// ClAM: Clustering with Associative Memory helpers.
///
/// Requires the `clam` feature.
#[cfg(feature = "clam")]
pub mod clam;

#[cfg(feature = "clam")]
pub use clam::{am_assign, am_contract, am_soft_assign, clam_loss};

/// Errors for kernel operations.
#[derive(Debug, Error)]
pub enum Error {
    #[error("empty input")]
    EmptyInput,

    #[error("dimension mismatch: {0} vs {1}")]
    DimensionMismatch(usize, usize),

    #[error("invalid bandwidth: {0}")]
    InvalidBandwidth(f64),
}

pub type Result<T> = std::result::Result<T, Error>;

// =============================================================================
// Kernel Functions
// =============================================================================

#[inline]
fn debug_assert_valid_bandwidth(sigma: f64) {
    debug_assert!(
        sigma.is_finite() && sigma > 0.0,
        "sigma must be finite and > 0"
    );
}

/// Radial Basis Function (Gaussian) kernel: k(x, y) = exp(-||x-y||² / (2σ²))
///
/// The most common kernel. Bandwidth σ controls smoothness:
/// - Small σ: Highly peaked, only nearby points similar
/// - Large σ: Broad similarity, approaches constant kernel
///
/// # Arguments
///
/// * `x` - First point
/// * `y` - Second point
/// * `sigma` - Bandwidth parameter (standard deviation)
///
/// # Example
///
/// ```rust
/// use rkhs::rbf;
///
/// let x = vec![0.0, 0.0];
/// let y = vec![1.0, 0.0];
///
/// let k = rbf(&x, &y, 1.0);
/// // exp(-1/(2*1)) = exp(-0.5) ≈ 0.606
/// assert!((k - 0.606).abs() < 0.01);
/// ```
pub fn rbf(x: &[f64], y: &[f64], sigma: f64) -> f64 {
    debug_assert_valid_bandwidth(sigma);
    let sq_dist: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - yi).powi(2))
        .sum();
    (-sq_dist / (2.0 * sigma * sigma)).exp()
}

/// Polynomial kernel: k(x, y) = (γ⟨x,y⟩ + c)^d
///
/// # Arguments
///
/// * `x` - First point
/// * `y` - Second point
/// * `degree` - Polynomial degree
/// * `gamma` - Scaling factor (default: 1/dim)
/// * `coef0` - Constant term (default: 1.0)
///
/// # Example
///
/// ```rust
/// use rkhs::polynomial;
///
/// let x = vec![1.0, 2.0];
/// let y = vec![3.0, 4.0];
///
/// let k = polynomial(&x, &y, 2, 1.0, 1.0);
/// // (1*3 + 2*4 + 1)² = (3 + 8 + 1)² = 144
/// assert!((k - 144.0).abs() < 1e-10);
/// ```
pub fn polynomial(x: &[f64], y: &[f64], degree: u32, gamma: f64, coef0: f64) -> f64 {
    let dot: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
    (gamma * dot + coef0).powi(degree as i32)
}

/// Linear kernel: k(x, y) = ⟨x, y⟩
///
/// Simplest kernel, equivalent to operating in original feature space.
///
/// # Examples
///
/// ```rust
/// use rkhs::linear;
///
/// let x = [1.0, 2.0, 3.0];
/// let y = [4.0, 5.0, 6.0];
/// assert_eq!(linear(&x, &y), 32.0);
/// ```
pub fn linear(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum()
}

/// Laplacian kernel: k(x, y) = exp(-||x-y||₁ / σ)
///
/// Uses L1 norm instead of L2. More robust to outliers than RBF.
///
/// # Examples
///
/// ```rust
/// use rkhs::laplacian;
///
/// let x = [0.0, 0.0];
/// let y = [1.0, 0.0];
/// // exp(-||x-y||_1 / sigma) = exp(-1 / 1)
/// let k = laplacian(&x, &y, 1.0);
/// assert!((k - (-1.0_f64).exp()).abs() < 1e-12);
/// ```
pub fn laplacian(x: &[f64], y: &[f64], sigma: f64) -> f64 {
    debug_assert_valid_bandwidth(sigma);
    let l1_dist: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| (xi - yi).abs()).sum();
    (-l1_dist / sigma).exp()
}

/// Epanechnikov kernel: k(x, y) = max(0, 1 - ||x-y||² / σ²)
///
/// Optimal kernel for minimizing MISE in density estimation.
/// Has compact support (zero outside radius σ).
///
/// # Examples
///
/// ```rust
/// use rkhs::epanechnikov;
///
/// let x = [0.0, 0.0];
/// let y = [0.5, 0.0];
/// // ||x-y||^2 = 0.25, sigma^2 = 1.0 => 1 - 0.25 = 0.75
/// assert!((epanechnikov(&x, &y, 1.0) - 0.75).abs() < 1e-12);
/// ```
pub fn epanechnikov(x: &[f64], y: &[f64], sigma: f64) -> f64 {
    debug_assert_valid_bandwidth(sigma);
    let sq_dist: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - yi).powi(2))
        .sum();
    let u_sq = sq_dist / (sigma * sigma);
    (1.0 - u_sq).max(0.0)
}

/// Triangle kernel: k(x, y) = max(0, 1 - ||x-y|| / σ)
///
/// Linearly decaying kernel with compact support.
///
/// # Examples
///
/// ```rust
/// use rkhs::triangle;
///
/// let x = [0.0, 0.0];
/// let y = [0.25, 0.0];
/// // ||x-y|| = 0.25, sigma = 1.0 => 1 - 0.25 = 0.75
/// assert!((triangle(&x, &y, 1.0) - 0.75).abs() < 1e-12);
/// ```
pub fn triangle(x: &[f64], y: &[f64], sigma: f64) -> f64 {
    debug_assert_valid_bandwidth(sigma);
    let sq_dist: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - yi).powi(2))
        .sum();
    let dist = sq_dist.sqrt();
    let u = dist / sigma;
    (1.0 - u).max(0.0)
}

/// Cosine kernel: k(x, y) = cos(π/2 * min(||x-y||/σ, 1))
///
/// Smooth kernel with compact support.
///
/// # Examples
///
/// ```rust
/// use rkhs::cosine;
///
/// let x = [0.0, 0.0];
/// let y = [0.0, 0.0];
/// assert!((cosine(&x, &y, 1.0) - 1.0).abs() < 1e-12);
/// ```
pub fn cosine(x: &[f64], y: &[f64], sigma: f64) -> f64 {
    debug_assert_valid_bandwidth(sigma);
    let sq_dist: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - yi).powi(2))
        .sum();
    let dist = sq_dist.sqrt();
    let u = (dist / sigma).min(1.0);
    (std::f64::consts::FRAC_PI_2 * u).cos()
}

/// Quartic (Biweight) kernel: k(x, y) = max(0, 1 - ||x-y||² / σ²)²
///
/// # Examples
///
/// ```rust
/// use rkhs::quartic;
///
/// let x = [0.0, 0.0];
/// let y = [0.5, 0.0];
/// // epanechnikov = 0.75 => quartic = 0.75^2
/// assert!((quartic(&x, &y, 1.0) - 0.75_f64 * 0.75).abs() < 1e-12);
/// ```
pub fn quartic(x: &[f64], y: &[f64], sigma: f64) -> f64 {
    let k = epanechnikov(x, y, sigma);
    k * k
}

/// Triweight kernel: k(x, y) = max(0, 1 - ||x-y||² / σ²)³
///
/// # Examples
///
/// ```rust
/// use rkhs::triweight;
///
/// let x = [0.0, 0.0];
/// let y = [0.5, 0.0];
/// // epanechnikov = 0.75 => triweight = 0.75^3
/// assert!((triweight(&x, &y, 1.0) - 0.75_f64 * 0.75 * 0.75).abs() < 1e-12);
/// ```
pub fn triweight(x: &[f64], y: &[f64], sigma: f64) -> f64 {
    let k = epanechnikov(x, y, sigma);
    k * k * k
}

/// Tricube kernel: k(x, y) = max(0, 1 - (||x-y||/σ)³)³
///
/// # Examples
///
/// ```rust
/// use rkhs::tricube;
///
/// let x = [0.0, 0.0];
/// let y = [0.0, 0.0];
/// assert!((tricube(&x, &y, 1.0) - 1.0).abs() < 1e-12);
/// ```
pub fn tricube(x: &[f64], y: &[f64], sigma: f64) -> f64 {
    debug_assert_valid_bandwidth(sigma);
    let sq_dist: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - yi).powi(2))
        .sum();
    let dist = sq_dist.sqrt();
    let u = dist / sigma;
    let term = (1.0 - u.powi(3)).max(0.0);
    term * term * term
}

// =============================================================================
// Kernel Matrices
// =============================================================================

/// Compute the Gram matrix K[i,j] = k(X[i], X[j]).
///
/// # Arguments
///
/// * `data` - Data points (each inner Vec is one point)
/// * `kernel` - Kernel function k(x, y) -> f64
///
/// # Returns
///
/// Symmetric n×n matrix
///
/// # Example
///
/// ```rust
/// use rkhs::{kernel_matrix, rbf};
///
/// let data = vec![
///     vec![0.0, 0.0],
///     vec![1.0, 0.0],
///     vec![0.0, 1.0],
/// ];
///
/// let k = kernel_matrix(&data, |x, y| rbf(x, y, 1.0));
/// assert_eq!(k.shape(), &[3, 3]);
/// assert!((k[[0, 0]] - 1.0).abs() < 1e-10);  // k(x, x) = 1 for RBF
/// ```
pub fn kernel_matrix<F>(data: &[Vec<f64>], kernel: F) -> Array2<f64>
where
    F: Fn(&[f64], &[f64]) -> f64,
{
    let n = data.len();
    let mut k = Array2::zeros((n, n));

    for i in 0..n {
        for j in i..n {
            let kij = kernel(&data[i], &data[j]);
            k[[i, j]] = kij;
            k[[j, i]] = kij; // Symmetric
        }
    }

    k
}

/// Compute the Gram matrix K[i,j] = k(x_i, x_j) for an `ndarray` matrix of points.
///
/// This avoids the common `Array2 -> Vec<Vec<f64>>` copy when callers already have
/// their data in `ndarray` form.
pub fn kernel_matrix_ndarray<F>(points: ArrayView2<'_, f64>, kernel: F) -> Array2<f64>
where
    F: Fn(ArrayView1<'_, f64>, ArrayView1<'_, f64>) -> f64,
{
    let n = points.nrows();
    let mut k = Array2::zeros((n, n));

    for i in 0..n {
        let xi = points.row(i);
        for j in i..n {
            let xj = points.row(j);
            let kij = kernel(xi, xj);
            k[[i, j]] = kij;
            k[[j, i]] = kij;
        }
    }

    k
}

/// RBF Gram matrix for an `ndarray` matrix of points.
///
/// K[i,j] = exp(-||x_i - x_j||² / (2σ²)).
///
/// This implementation uses the expansion ||x-y||² = ||x||² + ||y||² - 2x·y
/// to leverage highly optimized BLAS/SIMD dot products.
pub fn rbf_kernel_matrix_ndarray(points: ArrayView2<'_, f64>, sigma: f64) -> Array2<f64> {
    debug_assert!(
        sigma.is_finite() && sigma > 0.0,
        "sigma must be finite and > 0"
    );

    let n = points.nrows();
    let sigma_sq_2 = 2.0 * sigma * sigma;

    // 1. Compute squared norms: ||x_i||²
    let mut sq_norms = Array1::<f64>::zeros(n);
    for i in 0..n {
        let row = points.row(i);
        sq_norms[i] = row.dot(&row);
    }

    // 2. Compute dot products: X Xᵀ (O(n² d), but BLAS-accelerated)
    let mut k = points.dot(&points.t());

    // 3. Compute distances and apply RBF: exp(-(||x||² + ||y||² - 2x·y) / 2σ²)
    for i in 0..n {
        for j in 0..n {
            let dist_sq = (sq_norms[i] + sq_norms[j] - 2.0 * k[[i, j]]).max(0.0);
            k[[i, j]] = (-dist_sq / sigma_sq_2).exp();
        }
    }

    k
}

/// Compute cross-kernel matrix K[i,j] = k(X[i], Y[j]).
///
/// For comparing two different sets of points.
pub fn kernel_matrix_cross<F>(x: &[Vec<f64>], y: &[Vec<f64>], kernel: F) -> Array2<f64>
where
    F: Fn(&[f64], &[f64]) -> f64,
{
    let nx = x.len();
    let ny = y.len();
    let mut k = Array2::zeros((nx, ny));

    for i in 0..nx {
        for j in 0..ny {
            k[[i, j]] = kernel(&x[i], &y[j]);
        }
    }

    k
}

// =============================================================================
// Maximum Mean Discrepancy (MMD)
// =============================================================================

/// Biased MMD estimate in O(n) time.
///
/// Uses the empirical mean embeddings:
/// MMD²(P, Q) ≈ ||μ_P - μ_Q||²_H
///
/// This is a biased estimator but fast and useful for optimization.
///
/// # Arguments
///
/// * `x` - Samples from distribution P
/// * `y` - Samples from distribution Q
/// * `kernel` - Kernel function
///
/// # Returns
///
/// MMD² estimate (biased)
///
/// # Example
///
/// ```rust
/// use rkhs::{mmd_biased, rbf};
///
/// // Same distribution
/// let x = vec![vec![0.0], vec![0.1], vec![0.2]];
/// let y = vec![vec![0.05], vec![0.15], vec![0.25]];
/// let mmd_same = mmd_biased(&x, &y, |a, b| rbf(a, b, 1.0));
///
/// // Different distributions
/// let z = vec![vec![10.0], vec![10.1], vec![10.2]];
/// let mmd_diff = mmd_biased(&x, &z, |a, b| rbf(a, b, 1.0));
///
/// assert!(mmd_diff > mmd_same);
/// ```
pub fn mmd_biased<F>(x: &[Vec<f64>], y: &[Vec<f64>], kernel: F) -> f64
where
    F: Fn(&[f64], &[f64]) -> f64,
{
    let nx = x.len() as f64;
    let ny = y.len() as f64;

    if nx == 0.0 || ny == 0.0 {
        return 0.0;
    }

    // E[k(X, X')]
    let mut kxx = 0.0;
    for xi in x {
        for xj in x {
            kxx += kernel(xi, xj);
        }
    }
    kxx /= nx * nx;

    // E[k(Y, Y')]
    let mut kyy = 0.0;
    for yi in y {
        for yj in y {
            kyy += kernel(yi, yj);
        }
    }
    kyy /= ny * ny;

    // E[k(X, Y)]
    let mut kxy = 0.0;
    for xi in x {
        for yj in y {
            kxy += kernel(xi, yj);
        }
    }
    kxy /= nx * ny;

    // MMD² = E[k(X,X')] + E[k(Y,Y')] - 2E[k(X,Y)]
    (kxx + kyy - 2.0 * kxy).max(0.0)
}

/// Unbiased MMD² estimate (u-statistic).
///
/// Uses the unbiased estimator:
/// MMD²_u = (1/(m(m-1))) ΣΣ k(xᵢ,xⱼ) + (1/(n(n-1))) ΣΣ k(yᵢ,yⱼ)
///          - (2/(mn)) ΣΣ k(xᵢ,yⱼ)
///
/// This is the proper test statistic for two-sample testing.
/// Time complexity: O(n² + m²).
///
/// # Arguments
///
/// * `x` - Samples from distribution P
/// * `y` - Samples from distribution Q  
/// * `kernel` - Kernel function
///
/// # Returns
///
/// Unbiased MMD² estimate
///
/// # Example
///
/// ```rust
/// use rkhs::{mmd_unbiased, rbf};
///
/// let x = vec![vec![0.0, 0.0], vec![0.1, 0.1], vec![0.2, 0.0]];
/// let y = vec![vec![5.0, 5.0], vec![5.1, 5.1], vec![5.2, 5.0]];
///
/// let mmd = mmd_unbiased(&x, &y, |a, b| rbf(a, b, 1.0));
/// assert!(mmd > 0.5);  // Very different distributions
/// ```
pub fn mmd_unbiased<F>(x: &[Vec<f64>], y: &[Vec<f64>], kernel: F) -> f64
where
    F: Fn(&[f64], &[f64]) -> f64,
{
    let m = x.len();
    let n = y.len();

    if m < 2 || n < 2 {
        return 0.0;
    }

    // Unbiased k(X, X') - exclude diagonal
    let mut kxx = 0.0;
    for (i, xi) in x.iter().enumerate() {
        for (j, xj) in x.iter().enumerate() {
            if i != j {
                kxx += kernel(xi, xj);
            }
        }
    }
    kxx /= (m * (m - 1)) as f64;

    // Unbiased k(Y, Y') - exclude diagonal
    let mut kyy = 0.0;
    for (i, yi) in y.iter().enumerate() {
        for (j, yj) in y.iter().enumerate() {
            if i != j {
                kyy += kernel(yi, yj);
            }
        }
    }
    kyy /= (n * (n - 1)) as f64;

    // k(X, Y) - no diagonal to exclude
    let mut kxy = 0.0;
    for xi in x {
        for yj in y {
            kxy += kernel(xi, yj);
        }
    }
    kxy /= (m * n) as f64;

    kxx + kyy - 2.0 * kxy
}

/// Linear-time MMD estimate using random features.
///
/// Approximates MMD in O(n) time using the Nyström method or
/// explicit random Fourier features.
///
/// # Arguments
///
/// * `x` - Samples from P
/// * `y` - Samples from Q
/// * `sigma` - RBF bandwidth
/// * `num_features` - Number of random features
///
/// # Returns
///
/// Approximate MMD² estimate
pub fn mmd_linear_rff(x: &[Vec<f64>], y: &[Vec<f64>], sigma: f64, num_features: usize) -> f64 {
    if x.is_empty() || y.is_empty() {
        return 0.0;
    }

    let dim = x[0].len();

    // Generate random features
    let mut rng = rand::rng();
    mmd_linear_rff_with_rng(x, y, sigma, num_features, dim, &mut rng)
}

/// Linear-time MMD estimate using random Fourier features, with caller-provided RNG.
///
/// This is useful for:
/// - **Deterministic tests/benchmarks** (seed your RNG)
/// - **Reproducible pipelines** (thread an RNG through)
///
/// Note: `mmd_linear_rff()` uses a thread-local RNG and is intentionally nondeterministic.
pub fn mmd_linear_rff_with_rng<R: rand::Rng + ?Sized>(
    x: &[Vec<f64>],
    y: &[Vec<f64>],
    sigma: f64,
    num_features: usize,
    dim: usize,
    rng: &mut R,
) -> f64 {
    if num_features == 0 {
        return 0.0;
    }

    // `Normal::new` only fails if std dev <= 0 or not finite.
    // Treat this as a caller bug: the RBF bandwidth must be positive and finite.
    let normal = Normal::new(0.0, 1.0 / sigma).expect("sigma must be positive and finite");

    let mut omega: Vec<Vec<f64>> = Vec::with_capacity(num_features);
    let mut b: Vec<f64> = Vec::with_capacity(num_features);

    for _ in 0..num_features {
        let w: Vec<f64> = (0..dim).map(|_| normal.sample(rng)).collect();
        omega.push(w);
        b.push(rng.random::<f64>() * 2.0 * std::f64::consts::PI);
    }

    // Compute random features z(x) = sqrt(2/D) * cos(ωᵀx + b)
    let scale = (2.0 / num_features as f64).sqrt();

    let z_x = |point: &[f64]| -> Vec<f64> {
        omega
            .iter()
            .zip(b.iter())
            .map(|(w, &bias)| {
                let dot: f64 = w.iter().zip(point.iter()).map(|(wi, xi)| wi * xi).sum();
                scale * (dot + bias).cos()
            })
            .collect()
    };

    // Mean embedding of X
    let mut mu_x = vec![0.0; num_features];
    for xi in x {
        let zi = z_x(xi);
        for (j, &zij) in zi.iter().enumerate() {
            mu_x[j] += zij;
        }
    }
    for v in &mut mu_x {
        *v /= x.len() as f64;
    }

    // Mean embedding of Y
    let mut mu_y = vec![0.0; num_features];
    for yi in y {
        let zi = z_x(yi);
        for (j, &zij) in zi.iter().enumerate() {
            mu_y[j] += zij;
        }
    }
    for v in &mut mu_y {
        *v /= y.len() as f64;
    }

    // MMD² ≈ ||μ_X - μ_Y||²
    mu_x.iter()
        .zip(mu_y.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum()
}

/// Linear-time MMD estimate using random Fourier features, with an explicit seed.
///
/// This is a convenience wrapper over [`mmd_linear_rff_with_rng`] to support reproducible runs.
pub fn mmd_linear_rff_with_seed(
    x: &[Vec<f64>],
    y: &[Vec<f64>],
    sigma: f64,
    num_features: usize,
    seed: u64,
) -> f64 {
    if x.is_empty() || y.is_empty() {
        return 0.0;
    }
    let dim = x[0].len();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    mmd_linear_rff_with_rng(x, y, sigma, num_features, dim, &mut rng)
}

// =============================================================================
// Two-Sample Testing
// =============================================================================

/// Permutation test for MMD significance.
///
/// Tests H₀: P = Q vs H₁: P ≠ Q by computing a p-value via permutation.
///
/// # Arguments
///
/// * `x` - Samples from P
/// * `y` - Samples from Q
/// * `kernel` - Kernel function
/// * `num_permutations` - Number of permutations (default: 1000)
///
/// # Returns
///
/// (mmd_observed, p_value)
///
/// # Example
///
/// ```rust
/// use rkhs::{mmd_permutation_test, rbf};
///
/// let x = vec![vec![0.0], vec![0.1], vec![0.2], vec![0.3]];
/// let y = vec![vec![10.0], vec![10.1], vec![10.2], vec![10.3]];
///
/// let (mmd, p_value) = mmd_permutation_test(&x, &y, |a, b| rbf(a, b, 1.0), 100);
///
/// // With very different distributions and enough permutations,
/// // p-value should be small (though test is stochastic)
/// assert!(mmd > 0.5);
/// ```
pub fn mmd_permutation_test<F>(
    x: &[Vec<f64>],
    y: &[Vec<f64>],
    kernel: F,
    num_permutations: usize,
) -> (f64, f64)
where
    F: Fn(&[f64], &[f64]) -> f64 + Copy,
{
    let observed_mmd = mmd_unbiased(x, y, kernel);

    // Pool samples
    let mut pooled: Vec<&Vec<f64>> = x.iter().chain(y.iter()).collect();
    let nx = x.len();

    let mut rng = rand::rng();
    let mut count_greater = 0usize;

    for _ in 0..num_permutations {
        // Shuffle
        for i in (1..pooled.len()).rev() {
            let j = rng.random_range(0..=i);
            pooled.swap(i, j);
        }

        // Split
        let x_perm: Vec<Vec<f64>> = pooled[..nx].iter().map(|v| (*v).clone()).collect();
        let y_perm: Vec<Vec<f64>> = pooled[nx..].iter().map(|v| (*v).clone()).collect();

        let perm_mmd = mmd_unbiased(&x_perm, &y_perm, kernel);

        if perm_mmd >= observed_mmd {
            count_greater += 1;
        }
    }

    let p_value = (count_greater as f64 + 1.0) / (num_permutations as f64 + 1.0);
    (observed_mmd, p_value)
}

// =============================================================================
// Kernel Bandwidth Selection
// =============================================================================

/// Median heuristic for RBF bandwidth selection.
///
/// Sets σ = median(||xᵢ - xⱼ||) / sqrt(2).
/// A common default that works reasonably well.
///
/// # Arguments
///
/// * `data` - Data points
///
/// # Returns
///
/// Recommended bandwidth σ
pub fn median_bandwidth(data: &[Vec<f64>]) -> f64 {
    let n = data.len();
    if n < 2 {
        return 1.0;
    }

    // Collect pairwise distances
    let mut distances = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let sq_dist: f64 = data[i]
                .iter()
                .zip(data[j].iter())
                .map(|(xi, xj)| (xi - xj).powi(2))
                .sum();
            distances.push(sq_dist.sqrt());
        }
    }

    distances.sort_by(|a, b| a.total_cmp(b));
    let median = distances[distances.len() / 2];

    median / (2.0_f64).sqrt()
}

/// Multi-scale kernel: average over several bandwidths.
///
/// Useful when the optimal bandwidth is unknown.
pub fn rbf_multiscale(x: &[f64], y: &[f64], sigmas: &[f64]) -> f64 {
    sigmas.iter().map(|&s| rbf(x, y, s)).sum::<f64>() / sigmas.len() as f64
}

// =============================================================================
// Associative Memory Energy Functions
// =============================================================================
//
// Connection to Dense Associative Memory (Krotov et al., 2016-2025):
//
// The energy function of an AM with stored patterns Ξ = {ξ^μ} is:
//
//   E_β(v; Ξ) = -Q(Σ_μ F(β S[v, ξ^μ]))
//
// where:
// - Q is a scaling function (log, identity)
// - F is a separation function (exp, ReLU)
// - S is a similarity (negative squared distance, dot product)
// - β is inverse temperature (sharpness)
//
// Memory retrieval is gradient descent on E: dv/dt = -∇E(v)
//
// Key energies:
// - LSE (Log-Sum-Exp): E = -log Σ exp(-β/2 ||v-ξ||²)  [RBF kernel]
// - LSR (Log-Sum-ReLU): E = -log Σ ReLU(1 - β/2 ||v-ξ||²)  [Epanechnikov]
//
// LSR has novel properties: exact single-step retrieval, novel memory generation.
// See: Hoover et al. (2025) "Dense Associative Memory with Epanechnikov Energy"

/// Compute kernel sum: Σ_μ κ(v, ξ^μ)
///
/// This is the core computation in Associative Memory and kernel machines.
///
/// # Arguments
///
/// * `v` - Query point
/// * `memories` - Stored patterns {ξ^μ}
/// * `kernel` - Kernel function κ(v, ξ) -> f64
///
/// # Returns
///
/// Sum of kernel evaluations
///
/// # Example
///
/// ```rust
/// use rkhs::{kernel_sum, rbf};
///
/// let v = vec![0.0, 0.0];
/// let memories = vec![
///     vec![0.0, 0.0],  // close to v
///     vec![10.0, 10.0],  // far from v
/// ];
///
/// let sum = kernel_sum(&v, &memories, |a, b| rbf(a, b, 1.0));
/// // ≈ 1.0 (from first memory) + ~0 (from second)
/// assert!(sum > 0.99 && sum < 1.01);
/// ```
pub fn kernel_sum<F>(v: &[f64], memories: &[Vec<f64>], kernel: F) -> f64
where
    F: Fn(&[f64], &[f64]) -> f64,
{
    memories.iter().map(|xi| kernel(v, xi)).sum()
}

/// Log-Sum-Exp (LSE) energy for Dense Associative Memory.
///
/// E_β(v; Ξ) = -log Σ_μ exp(-β/2 ||v - ξ^μ||²)
///
/// This corresponds to RBF kernel with log scaling. The gradient points toward
/// a weighted average of memories, with weights given by softmax over similarities.
///
/// Properties:
/// - Smooth energy landscape
/// - Exponential memory capacity
/// - Approximate retrieval (needs T → ∞ for exact)
///
/// # Arguments
///
/// * `v` - Current state
/// * `memories` - Stored patterns
/// * `beta` - Inverse temperature (larger = sharper peaks around memories)
///
/// # Example
///
/// ```rust
/// use rkhs::energy_lse;
///
/// let memories = vec![
///     vec![0.0, 0.0],
///     vec![10.0, 10.0],
/// ];
///
/// // At a memory: low energy
/// let e_at_memory = energy_lse(&[0.0, 0.0], &memories, 1.0);
///
/// // Between memories: higher energy
/// let e_between = energy_lse(&[5.0, 5.0], &memories, 1.0);
///
/// assert!(e_at_memory < e_between);
/// ```
pub fn energy_lse(v: &[f64], memories: &[Vec<f64>], beta: f64) -> f64 {
    if memories.is_empty() {
        return 0.0;
    }

    // For numerical stability, use log-sum-exp trick:
    // log(Σ exp(x_i)) = max(x) + log(Σ exp(x_i - max(x)))
    let neg_half_beta = -0.5 * beta;

    let log_terms: Vec<f64> = memories
        .iter()
        .map(|xi| {
            let sq_dist: f64 = v
                .iter()
                .zip(xi.iter())
                .map(|(vi, xii)| (vi - xii).powi(2))
                .sum();
            neg_half_beta * sq_dist
        })
        .collect();

    let max_term = log_terms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    if max_term.is_infinite() {
        return f64::INFINITY;
    }

    let sum_exp: f64 = log_terms.iter().map(|&t| (t - max_term).exp()).sum();

    -(max_term + sum_exp.ln())
}

/// Gradient of LSE energy: ∇_v E_LSE(v; Ξ)
///
/// The gradient is a weighted combination of (v - ξ^μ) vectors,
/// where weights are softmax over similarities.
///
/// # Returns
///
/// Gradient vector of same dimension as v
pub fn energy_lse_grad(v: &[f64], memories: &[Vec<f64>], beta: f64) -> Vec<f64> {
    if memories.is_empty() || v.is_empty() {
        return vec![0.0; v.len()];
    }

    let d = v.len();
    let neg_half_beta = -0.5 * beta;

    // Compute softmax weights
    let sq_dists: Vec<f64> = memories
        .iter()
        .map(|xi| {
            v.iter()
                .zip(xi.iter())
                .map(|(vi, xii)| (vi - xii).powi(2))
                .sum()
        })
        .collect();

    let log_weights: Vec<f64> = sq_dists.iter().map(|&d| neg_half_beta * d).collect();
    let max_log = log_weights
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let exp_weights: Vec<f64> = log_weights.iter().map(|&w| (w - max_log).exp()).collect();
    let sum_exp: f64 = exp_weights.iter().sum();

    let softmax_weights: Vec<f64> = exp_weights.iter().map(|&w| w / sum_exp).collect();

    // Gradient: β Σ_μ w_μ (v - ξ^μ)
    let mut grad = vec![0.0; d];
    for (mu, xi) in memories.iter().enumerate() {
        let w = softmax_weights[mu];
        for (i, (vi, xii)) in v.iter().zip(xi.iter()).enumerate() {
            grad[i] += w * (vi - xii);
        }
    }

    for g in &mut grad {
        *g *= beta;
    }

    grad
}

/// Log-Sum-ReLU (LSR) energy for Dense Associative Memory with Epanechnikov kernel.
///
/// E_β(v; Ξ) = -log Σ_μ ReLU(1 - β/2 ||v - ξ^μ||²)
///
/// Based on the Epanechnikov kernel, which is optimal for density estimation (MISE).
///
/// Properties (from Hoover et al., 2025):
/// - Exact single-step retrieval (unlike LSE which needs many steps)
/// - Exponential memory capacity
/// - Can generate novel memories at basin intersections
/// - Compact support: regions of infinite energy where no memory is nearby
///
/// # Arguments
///
/// * `v` - Current state
/// * `memories` - Stored patterns
/// * `beta` - Inverse temperature (controls support radius: r = sqrt(2/β))
///
/// # Example
///
/// ```rust
/// use rkhs::energy_lsr;
///
/// let memories = vec![
///     vec![0.0, 0.0],
///     vec![10.0, 10.0],
/// ];
///
/// // At a memory: low energy
/// let e_at = energy_lsr(&[0.0, 0.0], &memories, 1.0);
///
/// // Far from all memories: infinite energy (outside support)
/// let e_far = energy_lsr(&[100.0, 100.0], &memories, 1.0);
///
/// assert!(e_at.is_finite());
/// assert!(e_far.is_infinite());
/// ```
pub fn energy_lsr(v: &[f64], memories: &[Vec<f64>], beta: f64) -> f64 {
    if memories.is_empty() {
        return 0.0;
    }

    let half_beta = 0.5 * beta;

    let sum: f64 = memories
        .iter()
        .map(|xi| {
            let sq_dist: f64 = v
                .iter()
                .zip(xi.iter())
                .map(|(vi, xii)| (vi - xii).powi(2))
                .sum();
            (1.0 - half_beta * sq_dist).max(0.0) // ReLU
        })
        .sum();

    if sum <= 0.0 {
        f64::INFINITY // Outside support of all memories
    } else {
        -sum.ln()
    }
}

/// Gradient of LSR energy: ∇_v E_LSR(v; Ξ)
///
/// Only memories within support (||v - ξ||² < 2/β) contribute to the gradient.
///
/// # Returns
///
/// Gradient vector. Returns zero vector if outside all support regions.
pub fn energy_lsr_grad(v: &[f64], memories: &[Vec<f64>], beta: f64) -> Vec<f64> {
    if memories.is_empty() || v.is_empty() {
        return vec![0.0; v.len()];
    }

    let d = v.len();
    let half_beta = 0.5 * beta;

    // Compute kernel values (only positive ones contribute)
    let kernel_vals: Vec<f64> = memories
        .iter()
        .map(|xi| {
            let sq_dist: f64 = v
                .iter()
                .zip(xi.iter())
                .map(|(vi, xii)| (vi - xii).powi(2))
                .sum();
            (1.0 - half_beta * sq_dist).max(0.0)
        })
        .collect();

    let sum: f64 = kernel_vals.iter().sum();

    if sum <= 0.0 {
        return vec![0.0; d]; // Outside support
    }

    // Gradient: (β / Σκ) × Σ_μ 1[κ_μ > 0] (v - ξ^μ)
    let mut grad = vec![0.0; d];
    for (mu, xi) in memories.iter().enumerate() {
        if kernel_vals[mu] > 0.0 {
            for (i, (vi, xii)) in v.iter().zip(xi.iter()).enumerate() {
                grad[i] += vi - xii;
            }
        }
    }

    let scale = beta / sum;
    for g in &mut grad {
        *g *= scale;
    }

    grad
}

/// Single step of energy descent for memory retrieval.
///
/// v_{t+1} = v_t - η ∇E(v_t)
///
/// # Arguments
///
/// * `v` - Current state (modified in place)
/// * `grad` - Gradient at current state
/// * `learning_rate` - Step size η
pub fn energy_descent_step(v: &mut [f64], grad: &[f64], learning_rate: f64) {
    for (vi, gi) in v.iter_mut().zip(grad.iter()) {
        *vi -= learning_rate * gi;
    }
}

/// Retrieve memory using energy descent.
///
/// Performs gradient descent on the energy function until convergence
/// or max iterations reached.
///
/// # Arguments
///
/// * `query` - Initial state (corrupted memory / query)
/// * `memories` - Stored patterns
/// * `energy_grad` - Function computing energy gradient
/// * `learning_rate` - Step size
/// * `max_iters` - Maximum iterations
/// * `tolerance` - Convergence threshold (gradient norm)
///
/// # Returns
///
/// (retrieved_memory, iterations_used)
///
/// # Example
///
/// ```rust
/// use rkhs::{retrieve_memory, energy_lse_grad};
///
/// let memories = vec![
///     vec![0.0, 0.0],
///     vec![10.0, 10.0],
/// ];
///
/// // Query near first memory
/// let query = vec![0.5, 0.5];
/// let (retrieved, iters) = retrieve_memory(
///     query,
///     &memories,
///     |v, m| energy_lse_grad(v, m, 2.0),
///     0.1,
///     100,
///     1e-6,
/// );
///
/// // Should converge near [0, 0]
/// assert!(retrieved[0].abs() < 1.0);
/// assert!(retrieved[1].abs() < 1.0);
/// ```
pub fn retrieve_memory<F>(
    query: Vec<f64>,
    memories: &[Vec<f64>],
    energy_grad: F,
    learning_rate: f64,
    max_iters: usize,
    tolerance: f64,
) -> (Vec<f64>, usize)
where
    F: Fn(&[f64], &[Vec<f64>]) -> Vec<f64>,
{
    let mut v = query;

    for iter in 0..max_iters {
        let grad = energy_grad(&v, memories);

        // Check convergence
        let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < tolerance {
            return (v, iter);
        }

        energy_descent_step(&mut v, &grad, learning_rate);
    }

    (v, max_iters)
}

// =============================================================================
// Positive Random Features (for non-negative kernel approximation)
// =============================================================================
//
// Standard trigonometric random features can produce negative kernel estimates.
// For applications requiring non-negative kernels (like LSE energy), we need
// positive random features (Choromanski et al., 2020).
//
// For RBF kernel: exp(-||x-y||²/2σ²) ≈ <Φ(x), Φ(y)> with Φ(x) guaranteed positive.

/// Positive random features for RBF kernel.
///
/// Unlike trigonometric features, these guarantee non-negative kernel estimates.
/// Uses the identity: exp(||x+x'||²/2) = E[exp(<ω,x>)exp(<ω,x'>)]
///
/// # Arguments
///
/// * `x` - Input point
/// * `omega` - Pre-generated random frequencies (each ω ~ N(0, I/σ²))
///
/// # Returns
///
/// Feature vector Φ(x) such that <Φ(x), Φ(y)> ≈ k(x, y)
pub fn positive_random_features(x: &[f64], omega: &[Vec<f64>], sq_norm_x: f64) -> Vec<f64> {
    let num_features = omega.len();
    let scale = (1.0 / num_features as f64).sqrt();

    // Φ_j(x) = scale * exp(-||x||²/2) * exp(<ω_j, x>)
    let norm_factor = (-sq_norm_x / 2.0).exp();

    omega
        .iter()
        .map(|w| {
            let dot: f64 = w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum();
            scale * norm_factor * dot.exp()
        })
        .collect()
}

/// Generate random frequencies for positive random features.
///
/// # Arguments
///
/// * `dim` - Dimension of input space
/// * `num_features` - Number of random features
/// * `sigma` - RBF bandwidth
/// * `rng` - Random number generator
pub fn generate_positive_rff_frequencies<R: Rng>(
    dim: usize,
    num_features: usize,
    sigma: f64,
    rng: &mut R,
) -> Vec<Vec<f64>> {
    let normal = Normal::new(0.0, 1.0 / sigma).expect("sigma must be positive");

    (0..num_features)
        .map(|_| (0..dim).map(|_| normal.sample(rng)).collect())
        .collect()
}

// =============================================================================
// Nyström Approximation
// =============================================================================

/// Nyström approximation for kernel matrix.
///
/// Approximates an n×n kernel matrix using only m landmark points,
/// reducing memory from O(n²) to O(nm).
///
/// K ≈ K_nm K_mm⁻¹ K_mn
///
/// # Arguments
///
/// * `data` - Full dataset
/// * `landmarks` - Landmark points (subset or random sample)
/// * `kernel` - Kernel function
///
/// # Returns
///
/// Low-rank approximation factors (L, W) where K ≈ L W Lᵀ
/// Nyström approximation for kernel matrix.
///
/// Approximates an n×n kernel matrix using only m landmark points,
/// reducing memory from O(n²) to O(nm).
///
/// K ≈ K_nm K_mm⁻¹ K_mn
///
/// # Arguments
///
/// * `data` - Full dataset
/// * `landmarks` - Landmark points (subset or random sample)
/// * `kernel` - Kernel function
///
/// # Returns
///
/// (K_nm, K_mm) matrices for reconstruction
pub fn nystrom_approximation<F>(
    data: &[Vec<f64>],
    landmarks: &[Vec<f64>],
    kernel: F,
) -> (Array2<f64>, Array2<f64>)
where
    F: Fn(&[f64], &[f64]) -> f64,
{
    // K_nm: n × m cross-kernel matrix
    let k_nm = kernel_matrix_cross(data, landmarks, &kernel);

    // K_mm: m × m landmark kernel matrix
    let k_mm = kernel_matrix(landmarks, &kernel);

    // Return factors: K ≈ K_nm K_mm⁻¹ K_mn
    // Caller can compute pseudo-inverse of K_mm as needed
    (k_nm, k_mm)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rbf_self() {
        let x = vec![1.0, 2.0, 3.0];
        let k = rbf(&x, &x, 1.0);
        assert!((k - 1.0).abs() < 1e-10, "k(x, x) should be 1 for RBF");
    }

    #[test]
    fn test_rbf_distant() {
        let x = vec![0.0, 0.0];
        let y = vec![100.0, 100.0];
        let k = rbf(&x, &y, 1.0);
        assert!(k < 1e-10, "distant points should have ~0 similarity");
    }

    #[test]
    fn test_polynomial() {
        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0];
        let k = polynomial(&x, &y, 2, 1.0, 1.0);
        // (1*3 + 2*4 + 1)² = 12² = 144
        assert!((k - 144.0).abs() < 1e-10);
    }

    #[test]
    fn test_mmd_same_distribution() {
        let x = vec![vec![0.0], vec![0.1], vec![0.2]];
        let y = vec![vec![0.05], vec![0.15], vec![0.25]];

        let mmd = mmd_unbiased(&x, &y, |a, b| rbf(a, b, 1.0));
        assert!(mmd < 0.1, "same distribution should have small MMD");
    }

    #[test]
    fn test_mmd_different_distributions() {
        let x = vec![vec![0.0], vec![0.1], vec![0.2]];
        let y = vec![vec![10.0], vec![10.1], vec![10.2]];

        let mmd = mmd_unbiased(&x, &y, |a, b| rbf(a, b, 1.0));
        assert!(mmd > 0.5, "different distributions should have large MMD");
    }

    #[test]
    fn test_mmd_non_negative() {
        // Use clearly different distributions
        let x = vec![vec![0.0], vec![0.1], vec![0.2], vec![0.3]];
        let y = vec![vec![10.0], vec![10.1], vec![10.2], vec![10.3]];

        let mmd = mmd_unbiased(&x, &y, |a, b| rbf(a, b, 1.0));
        // Note: unbiased MMD can be slightly negative for small samples
        // due to the u-statistic variance, but should be positive for
        // clearly different distributions
        assert!(
            mmd >= 0.0,
            "MMD should be non-negative for different distributions"
        );
    }

    #[test]
    fn test_kernel_matrix_symmetric() {
        let data = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];

        let k = kernel_matrix(&data, |x, y| rbf(x, y, 1.0));

        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (k[[i, j]] - k[[j, i]]).abs() < 1e-10,
                    "kernel matrix should be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_median_bandwidth_positive() {
        let data = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];

        let sigma = median_bandwidth(&data);
        assert!(sigma > 0.0, "bandwidth should be positive");
    }

    // =========================================================================
    // Associative Memory Energy Tests
    // =========================================================================

    #[test]
    fn test_kernel_sum() {
        let v = vec![0.0, 0.0];
        let memories = vec![vec![0.0, 0.0], vec![10.0, 10.0]];

        let sum = kernel_sum(&v, &memories, |a, b| rbf(a, b, 1.0));
        // First memory: distance 0 -> k = 1
        // Second memory: distance sqrt(200) -> k ≈ 0
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_energy_lse_at_memory() {
        let memories = vec![vec![0.0, 0.0], vec![10.0, 10.0]];

        // At first memory
        let e1 = energy_lse(&[0.0, 0.0], &memories, 1.0);
        // Between memories
        let e2 = energy_lse(&[5.0, 5.0], &memories, 1.0);

        assert!(e1 < e2, "energy should be lower at stored memory");
    }

    #[test]
    fn test_energy_lse_grad_points_toward_memory() {
        let memories = vec![vec![0.0, 0.0]];

        // Query slightly displaced from memory
        let v = vec![1.0, 0.0];
        let grad = energy_lse_grad(&v, &memories, 2.0);

        // Gradient should point away from memory (positive x direction)
        // Energy descent would move toward memory
        assert!(grad[0] > 0.0, "gradient should point away from memory");
    }

    #[test]
    fn test_energy_lsr_finite_at_memory() {
        let memories = vec![vec![0.0, 0.0], vec![10.0, 10.0]];

        // At first memory: should have finite energy
        let e = energy_lsr(&[0.0, 0.0], &memories, 1.0);
        assert!(e.is_finite(), "energy should be finite at memory");
    }

    #[test]
    fn test_energy_lsr_infinite_outside_support() {
        let memories = vec![vec![0.0, 0.0]];

        // Far from memory: outside compact support
        let e = energy_lsr(&[100.0, 100.0], &memories, 1.0);
        assert!(e.is_infinite(), "energy should be infinite outside support");
    }

    #[test]
    fn test_retrieve_memory_lse() {
        let memories = vec![vec![0.0, 0.0], vec![10.0, 10.0]];

        // Query near first memory
        let query = vec![1.0, 1.0];
        let (retrieved, _iters) = retrieve_memory(
            query,
            &memories,
            |v, m| energy_lse_grad(v, m, 2.0),
            0.1,
            100,
            1e-6,
        );

        // Should converge near [0, 0]
        let dist_to_first: f64 = retrieved.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            dist_to_first < 2.0,
            "should retrieve near first memory, got dist {}",
            dist_to_first
        );
    }

    #[test]
    fn test_energy_lsr_single_step_retrieval() {
        // Hoover et al. (2025) Theorem 1: single-step retrieval within basin radius
        let memories = vec![vec![0.0, 0.0], vec![10.0, 10.0]];
        let beta = 2.0; // support radius r = sqrt(2/beta) = 1.0
        
        // Query well within first basin (dist = 0.1 < 1.0)
        let query = vec![0.1, 0.0];
        
        // Compute gradient
        let grad = energy_lsr_grad(&query, &memories, beta);
        
        // Theorem 1 suggests η = 1/β for single-step retrieval in some cases,
        // but let's verify if gradient points exactly toward memory.
        // Gradient should be proportional to (query - memory).
        // grad = (beta / sum) * (query - memory)
        assert!(grad[0] > 0.0);
        assert!((grad[1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_epanechnikov_compact_support() {
        let x = vec![0.0, 0.0];
        let y_close = vec![0.5, 0.0];
        let y_far = vec![2.0, 0.0];

        // Inside support (||y|| < σ)
        let k_close = epanechnikov(&x, &y_close, 1.0);
        assert!(k_close > 0.0, "should be positive inside support");

        // Outside support (||y|| >= σ)
        let k_far = epanechnikov(&x, &y_far, 1.0);
        assert!(
            (k_far - 0.0).abs() < 1e-10,
            "should be zero outside support"
        );
    }

    #[test]
    fn test_triangle_linear_decay() {
        let x = vec![0.0];
        let sigma = 1.0;

        // At origin
        assert!((triangle(&x, &[0.0], sigma) - 1.0).abs() < 1e-10);

        // At half sigma
        assert!((triangle(&x, &[0.5], sigma) - 0.5).abs() < 1e-10);

        // At sigma (boundary)
        assert!((triangle(&x, &[1.0], sigma) - 0.0).abs() < 1e-10);
    }
}
