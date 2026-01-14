//! SIMD-accelerated kernel computations.
//!
//! Uses [`innr`] for fast vector operations when computing kernel matrices.
//!
//! # Performance
//!
//! For large matrices (n > 100), SIMD can provide 2-4x speedup depending on hardware.
//! The main gains come from:
//!
//! - Fast squared L2 distance for RBF kernel
//! - Fast dot products for linear/polynomial kernels
//! - Batch processing of multiple kernel values
//!
//! # Example
//!
//! ```rust,ignore
//! use rkhs::simd::{rbf_simd, kernel_matrix_rbf_simd};
//!
//! let x = vec![1.0_f32, 0.0, 0.0];
//! let y = vec![0.5, 0.5, 0.0];
//!
//! // SIMD-accelerated RBF kernel
//! let k = rbf_simd(&x, &y, 1.0);
//! ```

use ndarray::Array2;

/// SIMD-accelerated RBF kernel for f32 vectors.
///
/// Uses `innr::l2_distance_squared` for fast computation.
pub fn rbf_simd(x: &[f32], y: &[f32], sigma: f32) -> f32 {
    let sq_dist = innr::l2_distance_squared(x, y);
    (-sq_dist / (2.0 * sigma * sigma)).exp()
}

/// SIMD-accelerated linear kernel for f32 vectors.
pub fn linear_simd(x: &[f32], y: &[f32]) -> f32 {
    innr::dot(x, y)
}

/// SIMD-accelerated polynomial kernel for f32 vectors.
pub fn polynomial_simd(x: &[f32], y: &[f32], degree: i32, gamma: f32, coef0: f32) -> f32 {
    let dot = innr::dot(x, y);
    (gamma * dot + coef0).powi(degree)
}

/// SIMD-accelerated cosine kernel for f32 vectors.
///
/// Note: cosine similarity is already a kernel (with implicit normalization).
pub fn cosine_simd(x: &[f32], y: &[f32]) -> f32 {
    innr::cosine(x, y)
}

/// Compute RBF kernel matrix using SIMD.
///
/// More efficient than generic `kernel_matrix` for large datasets.
///
/// # Arguments
///
/// * `data` - n points, each a Vec<f32>
/// * `sigma` - RBF bandwidth
///
/// # Returns
///
/// n × n symmetric kernel matrix
pub fn kernel_matrix_rbf_simd(data: &[Vec<f32>], sigma: f32) -> Array2<f32> {
    let n = data.len();
    let mut k = Array2::zeros((n, n));
    
    let sigma_sq_2 = 2.0 * sigma * sigma;
    
    for i in 0..n {
        k[[i, i]] = 1.0;  // k(x, x) = 1 for RBF
        
        for j in (i + 1)..n {
            let sq_dist = innr::l2_distance_squared(&data[i], &data[j]);
            let kij = (-sq_dist / sigma_sq_2).exp();
            k[[i, j]] = kij;
            k[[j, i]] = kij;
        }
    }
    
    k
}

/// Compute linear kernel matrix using SIMD.
pub fn kernel_matrix_linear_simd(data: &[Vec<f32>]) -> Array2<f32> {
    let n = data.len();
    let mut k = Array2::zeros((n, n));
    
    for i in 0..n {
        for j in i..n {
            let kij = innr::dot(&data[i], &data[j]);
            k[[i, j]] = kij;
            k[[j, i]] = kij;
        }
    }
    
    k
}

/// Batch MMD computation with SIMD-accelerated kernels.
///
/// Computes unbiased MMD² between samples X and Y using RBF kernel.
pub fn mmd_unbiased_simd(x: &[Vec<f32>], y: &[Vec<f32>], sigma: f32) -> f32 {
    let m = x.len();
    let n = y.len();
    
    if m < 2 || n < 2 {
        return 0.0;
    }
    
    let sigma_sq_2 = 2.0 * sigma * sigma;
    
    // Unbiased k(X, X') - exclude diagonal
    let mut kxx = 0.0f32;
    for i in 0..m {
        for j in 0..m {
            if i != j {
                let sq_dist = innr::l2_distance_squared(&x[i], &x[j]);
                kxx += (-sq_dist / sigma_sq_2).exp();
            }
        }
    }
    kxx /= (m * (m - 1)) as f32;
    
    // Unbiased k(Y, Y') - exclude diagonal
    let mut kyy = 0.0f32;
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let sq_dist = innr::l2_distance_squared(&y[i], &y[j]);
                kyy += (-sq_dist / sigma_sq_2).exp();
            }
        }
    }
    kyy /= (n * (n - 1)) as f32;
    
    // k(X, Y)
    let mut kxy = 0.0f32;
    for i in 0..m {
        for j in 0..n {
            let sq_dist = innr::l2_distance_squared(&x[i], &y[j]);
            kxy += (-sq_dist / sigma_sq_2).exp();
        }
    }
    kxy /= (m * n) as f32;
    
    kxx + kyy - 2.0 * kxy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rbf_simd_self() {
        let x = vec![1.0f32, 2.0, 3.0];
        let k = rbf_simd(&x, &x, 1.0);
        assert!((k - 1.0).abs() < 1e-6, "k(x, x) should be 1 for RBF");
    }

    #[test]
    fn test_rbf_simd_distant() {
        let x = vec![0.0f32; 64];
        let mut y = vec![100.0f32; 64];
        y[0] = 100.0;
        let k = rbf_simd(&x, &y, 1.0);
        assert!(k < 1e-6, "distant points should have ~0 similarity");
    }

    #[test]
    fn test_linear_simd() {
        let x = vec![1.0f32, 2.0, 3.0];
        let y = vec![4.0f32, 5.0, 6.0];
        let k = linear_simd(&x, &y);
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert!((k - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_kernel_matrix_rbf_simd_symmetric() {
        let data = vec![
            vec![0.0f32, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        
        let k = kernel_matrix_rbf_simd(&data, 1.0);
        
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (k[[i, j]] - k[[j, i]]).abs() < 1e-6,
                    "kernel matrix should be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_mmd_simd_same_distribution() {
        let x = vec![vec![0.0f32], vec![0.1], vec![0.2]];
        let y = vec![vec![0.05f32], vec![0.15], vec![0.25]];
        
        let mmd = mmd_unbiased_simd(&x, &y, 1.0);
        assert!(mmd < 0.1, "same distribution should have small MMD: {}", mmd);
    }

    #[test]
    fn test_mmd_simd_different_distributions() {
        let x = vec![vec![0.0f32], vec![0.1], vec![0.2]];
        let y = vec![vec![10.0f32], vec![10.1], vec![10.2]];
        
        let mmd = mmd_unbiased_simd(&x, &y, 1.0);
        assert!(mmd > 0.5, "different distributions should have large MMD: {}", mmd);
    }
}
