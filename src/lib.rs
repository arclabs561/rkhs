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
//! This crate provides the primitives: kernels, Gram matrices, MMD, and kernel
//! quantile embeddings. Dense Associative Memory (AM) functions are re-exported
//! from the [`hopfield`] crate.
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
//! | [`kernel_matrix`] | Gram matrix K\[i,j\] = k(x_i, x_j) |
//! | [`kernel_sum`] | Sum Σ κ(v, ξ^μ) for AM/kernel machines (from `hopfield`) |
//! | [`energy_lse`] | Log-Sum-Exp energy (Dense AM with RBF) (from `hopfield`) |
//! | [`energy_lsr`] | Log-Sum-ReLU energy (Dense AM with Epanechnikov) (from `hopfield`) |
//! | [`retrieve_memory`] | Memory retrieval via energy descent (from `hopfield`) |
//! | [`mmd_biased`] | O(n²) biased MMD estimate |
//! | [`mmd_unbiased`] | O(n²) unbiased MMD u-statistic |
//! | [`mmd_permutation_test`] | Significance test via permutation |
//! | [`kernel_quantile_embedding`] | Kernel embedding at a quantile level |
//! | [`qmmd`] | Quantile MMD (tail-sensitive distribution comparison) |
//! | [`weighted_qmmd`] | QMMD with configurable quantile-level weighting |
//! | [`quantile_function_embedding`] | Kernel-smoothed quantile function at specified levels |
//! | [`quantile_distribution_kernel`] | Kernel between distributions via quantile embeddings |
//! | [`quantile_gram_matrix`] | Gram matrix restricted to a quantile level |
//!
//! ## Modern Hopfield Networks in 10 Lines
//!
//! Dense Associative Memory (AM) functions are provided by the [`hopfield`] crate
//! and re-exported here for convenience:
//!
//! ```rust
//! use rkhs::{energy_lse_grad, retrieve_memory};
//!
//! // Store three patterns (colours in RGB-ish space)
//! let memories = vec![
//!     vec![1.0, 0.0, 0.0],  // red
//!     vec![0.0, 1.0, 0.0],  // green
//!     vec![0.0, 0.0, 1.0],  // blue
//! ];
//!
//! // Noisy query: mostly red but corrupted
//! let query = vec![0.9, 0.2, 0.1];
//!
//! // Retrieve via energy descent
//! let (retrieved, iters) = retrieve_memory(
//!     query,
//!     &memories,
//!     |v, m| energy_lse_grad(v, m, 10.0),  // beta=10 → sharp attractor
//!     0.1,   // learning rate
//!     200,   // max iterations
//!     1e-7,  // convergence tolerance
//! );
//!
//! // Nearest pattern is red: [1,0,0]
//! assert!(retrieved[0] > 0.9, "should converge to red");
//! assert!(retrieved[1] < 0.1, "green component suppressed");
//! assert!(retrieved[2] < 0.1, "blue component suppressed");
//! println!("Converged in {iters} iterations: {retrieved:?}");
//! ```
//!
//! ## Quick Start (MMD)
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
//! - Associative Memory: Energy functions E = -log Σ κ(v, ξ) define memory landscapes
//! - GAN evaluation: FID uses MMD-like statistics to compare generated vs real
//! - Domain adaptation: Minimize MMD between source and target distributions
//! - Two-sample testing: Detect distribution shift in production systems
//! - Kernel regression: Nonparametric regression via kernel mean embedding
//!
//! ## Connections
//!
//! - [`logp`](../logp): MMD and KL divergence both measure distribution "distance"
//! - [`wass`](../wass): Wasserstein and MMD are different ways to compare distributions
//! - [`lapl`](../lapl): Gaussian kernel → Laplacian eigenvalue problems
//!
//! ## What Can Go Wrong
//!
//! 1. Bandwidth too small: RBF kernel becomes nearly diagonal, loses structure.
//! 2. Bandwidth too large: Everything becomes similar, no discrimination.
//! 3. Numerical instability: Very large distances → exp(-large) → 0 underflow.
//! 4. MMD variance: With small samples, MMD estimates are noisy. Use permutation test.
//! 5. Kernel not characteristic: Not all kernels can distinguish all distributions.
//!    RBF is characteristic (good); polynomial is not (bad for two-sample test).
//!
//! ## References
//!
//! - Gretton et al. (2012). "A Kernel Two-Sample Test" (JMLR)
//! - Muandet et al. (2017). "Kernel Mean Embedding of Distributions" (Found. & Trends)
//! - Naslidnyk et al. (2025). "Kernel Quantile Embeddings"

use ndarray::{Array1, Array2, ArrayView2};
use rand::Rng;

/// CLAM: Clustering with Associative Memory helpers.
///
/// **Deprecated**: use `clump::clam` (with feature `rkhs`) instead.
#[deprecated(
    since = "0.2.0",
    note = "use `clump::clam` with feature `rkhs` instead"
)]
pub mod clam;
/// Kernels on probability distributions.
pub mod distribution_kernel;
/// Kernels on labeled graphs.
///
/// **Deprecated**: use `graphops::graph_kernel` instead.
pub mod graph_kernel;
/// Kernel quantile embeddings for tail-sensitive distribution comparison.
pub mod quantile_kernel;

#[allow(deprecated)]
pub use clam::{am_assign, am_contract, am_soft_assign, clam_loss};
pub use distribution_kernel::{
    expected_likelihood_kernel, fisher_kernel_categorical, jensen_shannon_kernel,
    probability_product_kernel,
};
#[allow(deprecated)]
pub use graph_kernel::{
    random_walk_kernel, sliced_wasserstein_graph_kernel, structural_node_features,
    wl_subtree_kernel,
};
pub use hopfield::{
    energy_lse, energy_lse_grad, energy_lsr, energy_lsr_grad, kernel_sum, retrieve_memory,
};
pub use quantile_kernel::{
    kernel_quantile_embedding, qmmd, quantile_distribution_kernel, quantile_function_embedding,
    quantile_gram_matrix, weighted_qmmd, QuantileWeight,
};

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

/// Compute the Gram matrix K\[i,j\] = k(X\[i\], X\[j\]).
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

/// RBF Gram matrix for an `ndarray` matrix of points.
///
/// K\[i,j\] = exp(-||x_i - x_j||² / (2σ²)).
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

// =============================================================================
// Maximum Mean Discrepancy (MMD)
// =============================================================================

/// Biased MMD estimate in O(n^2) time.
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
