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
//! | [`polynomial`] | Polynomial kernel |
//! | [`kernel_matrix`] | Gram matrix K[i,j] = k(x_i, x_j) |
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
//! ## Why MMD Matters for ML
//!
//! - **GAN evaluation**: FID uses MMD-like statistics to compare generated vs real
//! - **Domain adaptation**: Minimize MMD between source and target distributions
//! - **Two-sample testing**: Detect distribution shift in production systems
//! - **Kernel regression**: Nonparametric regression via kernel mean embedding
//!
//! ## Connections
//!
//! - [`surp`](../surp): MMD and KL divergence both measure distribution "distance"
//! - [`wass`](../wass): Wasserstein and MMD are different ways to compare distributions
//! - [`lapl`](../lapl): Gaussian kernel → Laplacian eigenvalue problems
//! - [`stratify`](../stratify): Kernel k-means uses these kernels
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

use ndarray::Array2;
use rand::Rng;
use rand_distr::{Normal, Distribution};
use thiserror::Error;

/// SIMD-accelerated kernel computations using innr.
///
/// Requires the `simd` feature.
#[cfg(feature = "simd")]
pub mod simd;

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
    let sq_dist: f64 = x.iter()
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
pub fn linear(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum()
}

/// Laplacian kernel: k(x, y) = exp(-||x-y||₁ / σ)
///
/// Uses L1 norm instead of L2. More robust to outliers than RBF.
pub fn laplacian(x: &[f64], y: &[f64], sigma: f64) -> f64 {
    let l1_dist: f64 = x.iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - yi).abs())
        .sum();
    (-l1_dist / sigma).exp()
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
            k[[j, i]] = kij;  // Symmetric
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
    for i in 0..x.len() {
        for j in 0..x.len() {
            kxx += kernel(&x[i], &x[j]);
        }
    }
    kxx /= nx * nx;
    
    // E[k(Y, Y')]
    let mut kyy = 0.0;
    for i in 0..y.len() {
        for j in 0..y.len() {
            kyy += kernel(&y[i], &y[j]);
        }
    }
    kyy /= ny * ny;
    
    // E[k(X, Y)]
    let mut kxy = 0.0;
    for i in 0..x.len() {
        for j in 0..y.len() {
            kxy += kernel(&x[i], &y[j]);
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
    for i in 0..m {
        for j in 0..m {
            if i != j {
                kxx += kernel(&x[i], &x[j]);
            }
        }
    }
    kxx /= (m * (m - 1)) as f64;
    
    // Unbiased k(Y, Y') - exclude diagonal
    let mut kyy = 0.0;
    for i in 0..n {
        for j in 0..n {
            if i != j {
                kyy += kernel(&y[i], &y[j]);
            }
        }
    }
    kyy /= (n * (n - 1)) as f64;
    
    // k(X, Y) - no diagonal to exclude
    let mut kxy = 0.0;
    for i in 0..m {
        for j in 0..n {
            kxy += kernel(&x[i], &y[j]);
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
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 1.0 / sigma).unwrap();
    
    let mut omega: Vec<Vec<f64>> = Vec::with_capacity(num_features);
    let mut b: Vec<f64> = Vec::with_capacity(num_features);
    
    for _ in 0..num_features {
        let w: Vec<f64> = (0..dim).map(|_| normal.sample(&mut rng)).collect();
        omega.push(w);
        b.push(rng.gen::<f64>() * 2.0 * std::f64::consts::PI);
    }
    
    // Compute random features z(x) = sqrt(2/D) * cos(ωᵀx + b)
    let scale = (2.0 / num_features as f64).sqrt();
    
    let z_x = |point: &[f64]| -> Vec<f64> {
        omega.iter()
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
    for j in 0..num_features {
        mu_x[j] /= x.len() as f64;
    }
    
    // Mean embedding of Y
    let mut mu_y = vec![0.0; num_features];
    for yi in y {
        let zi = z_x(yi);
        for (j, &zij) in zi.iter().enumerate() {
            mu_y[j] += zij;
        }
    }
    for j in 0..num_features {
        mu_y[j] /= y.len() as f64;
    }
    
    // MMD² ≈ ||μ_X - μ_Y||²
    mu_x.iter()
        .zip(mu_y.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum()
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
    num_permutations: usize
) -> (f64, f64)
where
    F: Fn(&[f64], &[f64]) -> f64 + Copy,
{
    let observed_mmd = mmd_unbiased(x, y, kernel);
    
    // Pool samples
    let mut pooled: Vec<&Vec<f64>> = x.iter().chain(y.iter()).collect();
    let nx = x.len();
    
    let mut rng = rand::thread_rng();
    let mut count_greater = 0usize;
    
    for _ in 0..num_permutations {
        // Shuffle
        for i in (1..pooled.len()).rev() {
            let j = rng.gen_range(0..=i);
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
            let sq_dist: f64 = data[i].iter()
                .zip(data[j].iter())
                .map(|(xi, xj)| (xi - xj).powi(2))
                .sum();
            distances.push(sq_dist.sqrt());
        }
    }
    
    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = distances[distances.len() / 2];
    
    median / (2.0_f64).sqrt()
}

/// Multi-scale kernel: average over several bandwidths.
///
/// Useful when the optimal bandwidth is unknown.
pub fn rbf_multiscale(x: &[f64], y: &[f64], sigmas: &[f64]) -> f64 {
    sigmas.iter()
        .map(|&s| rbf(x, y, s))
        .sum::<f64>() / sigmas.len() as f64
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
        assert!(mmd >= 0.0, "MMD should be non-negative for different distributions");
    }

    #[test]
    fn test_kernel_matrix_symmetric() {
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        
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
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        
        let sigma = median_bandwidth(&data);
        assert!(sigma > 0.0, "bandwidth should be positive");
    }
}
