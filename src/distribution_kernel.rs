//! Distribution kernels: similarity measures between probability distributions.
//!
//! These kernels operate on discrete probability distributions (histograms/PMFs)
//! rather than individual points. They connect RKHS methods to information-theoretic
//! quantities like Bhattacharyya coefficient, Jensen-Shannon divergence, and
//! Fisher information.
//!
//! All functions expect valid probability distributions (non-negative, sum to 1)
//! with matching lengths. Behavior on invalid inputs is unspecified.

/// Probability Product Kernel: `k(p, q) = sum_i p_i^rho * q_i^rho`
///
/// For discrete distributions (histograms), this integrates the pointwise product
/// of powered densities.
///
/// Special cases:
/// - `rho = 1`: inner product of distributions
/// - `rho = 0.5`: Bhattacharyya coefficient (Hellinger affinity)
///
/// # Arguments
///
/// * `p` - First distribution (PMF)
/// * `q` - Second distribution (PMF)
/// * `rho` - Power parameter (must be > 0)
///
/// # Example
///
/// ```rust
/// use rkhs::probability_product_kernel;
///
/// let p = [0.5, 0.3, 0.2];
/// let q = [0.5, 0.3, 0.2];
///
/// // Identical distributions with rho=0.5 => Bhattacharyya coeff = 1.0
/// let k = probability_product_kernel(&p, &q, 0.5);
/// assert!((k - 1.0).abs() < 1e-10);
/// ```
pub fn probability_product_kernel(p: &[f64], q: &[f64], rho: f64) -> f64 {
    debug_assert!(p.len() == q.len(), "distributions must have same length");
    debug_assert!(rho > 0.0, "rho must be positive");

    p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| pi.powf(rho) * qi.powf(rho))
        .sum()
}

/// Expected Likelihood Kernel: `k(p, q) = sum_i p_i * q_i`
///
/// The inner product of two distributions. Equivalent to
/// [`probability_product_kernel`] with `rho = 1`.
///
/// # Example
///
/// ```rust
/// use rkhs::expected_likelihood_kernel;
///
/// let p = [0.5, 0.5];
/// let q = [0.5, 0.5];
/// let k = expected_likelihood_kernel(&p, &q);
/// assert!((k - 0.5).abs() < 1e-10);
/// ```
pub fn expected_likelihood_kernel(p: &[f64], q: &[f64]) -> f64 {
    debug_assert!(p.len() == q.len(), "distributions must have same length");

    p.iter().zip(q.iter()).map(|(&pi, &qi)| pi * qi).sum()
}

/// Exponentiated Jensen-Shannon kernel: `k(p, q) = exp(-lambda * JSD(p, q))`
///
/// JSD is symmetric, bounded in `[0, ln(2)]`, and `sqrt(JSD)` is a metric.
/// The exponentiated form is a valid positive-definite kernel.
///
/// Uses natural logarithm internally. Zero-probability bins are skipped
/// (0 * log(0) = 0 by convention).
///
/// # Arguments
///
/// * `p` - First distribution (PMF)
/// * `q` - Second distribution (PMF)
/// * `lambda` - Scale parameter (must be > 0)
///
/// # Example
///
/// ```rust
/// use rkhs::jensen_shannon_kernel;
///
/// let p = [0.5, 0.5];
/// let q = [0.5, 0.5];
///
/// // Identical distributions => JSD = 0 => kernel = 1
/// let k = jensen_shannon_kernel(&p, &q, 1.0);
/// assert!((k - 1.0).abs() < 1e-10);
/// ```
pub fn jensen_shannon_kernel(p: &[f64], q: &[f64], lambda: f64) -> f64 {
    debug_assert!(p.len() == q.len(), "distributions must have same length");
    debug_assert!(lambda > 0.0, "lambda must be positive");

    // JSD(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m) where m = (p + q) / 2
    let mut jsd = 0.0;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        let mi = 0.5 * (pi + qi);
        if mi > 0.0 {
            if pi > 0.0 {
                jsd += 0.5 * pi * (pi / mi).ln();
            }
            if qi > 0.0 {
                jsd += 0.5 * qi * (qi / mi).ln();
            }
        }
    }

    (-lambda * jsd).exp()
}

/// Fisher kernel for categorical distributions: `k(p, q) = sum_i sqrt(p_i * q_i)`
///
/// This is the Bhattacharyya coefficient (equivalently, the Hellinger affinity),
/// which arises from the Fisher information metric on the simplex.
/// Equal to [`probability_product_kernel`] with `rho = 0.5`.
///
/// Returns 1.0 for identical distributions and 0.0 for distributions with
/// disjoint support.
///
/// # Example
///
/// ```rust
/// use rkhs::fisher_kernel_categorical;
///
/// let p = [0.25, 0.25, 0.25, 0.25];
/// let q = [0.25, 0.25, 0.25, 0.25];
/// let k = fisher_kernel_categorical(&p, &q);
/// assert!((k - 1.0).abs() < 1e-10);
/// ```
pub fn fisher_kernel_categorical(p: &[f64], q: &[f64]) -> f64 {
    debug_assert!(p.len() == q.len(), "distributions must have same length");

    p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| (pi * qi).sqrt())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Probability Product Kernel
    // =========================================================================

    #[test]
    fn ppk_rho_half_is_bhattacharyya() {
        let p = [0.2, 0.3, 0.5];
        let q = [0.1, 0.6, 0.3];

        let ppk = probability_product_kernel(&p, &q, 0.5);
        let bc: f64 = p
            .iter()
            .zip(q.iter())
            .map(|(&pi, &qi)| (pi * qi).sqrt())
            .sum();

        assert!(
            (ppk - bc).abs() < 1e-12,
            "PPK(rho=0.5) should equal Bhattacharyya coefficient"
        );
    }

    #[test]
    fn ppk_identical_rho_half() {
        let p = [0.25, 0.25, 0.25, 0.25];
        let k = probability_product_kernel(&p, &p, 0.5);
        assert!((k - 1.0).abs() < 1e-10);
    }

    #[test]
    fn ppk_non_negative() {
        let p = [0.1, 0.2, 0.7];
        let q = [0.3, 0.3, 0.4];
        for rho in [0.25, 0.5, 1.0, 2.0] {
            let k = probability_product_kernel(&p, &q, rho);
            assert!(k >= 0.0, "PPK must be non-negative, got {k} for rho={rho}");
        }
    }

    #[test]
    fn ppk_symmetry() {
        let p = [0.1, 0.4, 0.5];
        let q = [0.3, 0.3, 0.4];
        let k_pq = probability_product_kernel(&p, &q, 0.7);
        let k_qp = probability_product_kernel(&q, &p, 0.7);
        assert!((k_pq - k_qp).abs() < 1e-12);
    }

    // =========================================================================
    // Expected Likelihood Kernel
    // =========================================================================

    #[test]
    fn elk_symmetric() {
        let p = [0.2, 0.5, 0.3];
        let q = [0.4, 0.1, 0.5];
        let k_pq = expected_likelihood_kernel(&p, &q);
        let k_qp = expected_likelihood_kernel(&q, &p);
        assert!((k_pq - k_qp).abs() < 1e-12);
    }

    #[test]
    fn elk_equals_ppk_rho1() {
        let p = [0.3, 0.3, 0.4];
        let q = [0.1, 0.2, 0.7];
        let elk = expected_likelihood_kernel(&p, &q);
        let ppk = probability_product_kernel(&p, &q, 1.0);
        assert!((elk - ppk).abs() < 1e-12);
    }

    #[test]
    fn elk_uniform() {
        // For uniform distributions of size n: ELK = sum(1/n * 1/n) = 1/n
        let p = [0.25, 0.25, 0.25, 0.25];
        let k = expected_likelihood_kernel(&p, &p);
        assert!((k - 0.25).abs() < 1e-12);
    }

    // =========================================================================
    // Jensen-Shannon Kernel
    // =========================================================================

    #[test]
    fn jsk_identical_distributions() {
        let p = [0.3, 0.5, 0.2];
        let k = jensen_shannon_kernel(&p, &p, 1.0);
        assert!(
            (k - 1.0).abs() < 1e-10,
            "JSK of identical distributions should be 1"
        );
    }

    #[test]
    fn jsk_disjoint_distributions() {
        let p = [1.0, 0.0];
        let q = [0.0, 1.0];
        let k = jensen_shannon_kernel(&p, &q, 1.0);
        // JSD of disjoint distributions = ln(2), so k = exp(-ln(2)) = 0.5
        assert!(
            (k - 0.5).abs() < 1e-10,
            "JSK of disjoint distributions with lambda=1 should be 0.5, got {k}"
        );
    }

    #[test]
    fn jsk_psd_gram_3x3() {
        // Verify positive semi-definiteness via 3x3 Gram matrix eigenvalues.
        let dists = [[0.5, 0.3, 0.2], [0.1, 0.6, 0.3], [0.33, 0.34, 0.33]];
        let n = dists.len();
        let mut gram = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                gram[i][j] = jensen_shannon_kernel(&dists[i], &dists[j], 1.0);
            }
        }
        // For a 3x3 PSD matrix, all principal minors must be non-negative.
        // det(1x1) = gram[0][0] >= 0
        assert!(gram[0][0] >= 0.0);
        // det(2x2) >= 0
        let det2 = gram[0][0] * gram[1][1] - gram[0][1] * gram[1][0];
        assert!(
            det2 >= -1e-10,
            "2x2 minor should be non-negative, got {det2}"
        );
        // det(3x3) >= 0
        let det3 = gram[0][0] * (gram[1][1] * gram[2][2] - gram[1][2] * gram[2][1])
            - gram[0][1] * (gram[1][0] * gram[2][2] - gram[1][2] * gram[2][0])
            + gram[0][2] * (gram[1][0] * gram[2][1] - gram[1][1] * gram[2][0]);
        assert!(
            det3 >= -1e-10,
            "3x3 determinant should be non-negative, got {det3}"
        );
    }

    // =========================================================================
    // Fisher Kernel (Categorical)
    // =========================================================================

    #[test]
    fn fisher_identical_is_one() {
        let p = [0.1, 0.2, 0.3, 0.4];
        let k = fisher_kernel_categorical(&p, &p);
        assert!((k - 1.0).abs() < 1e-10);
    }

    #[test]
    fn fisher_disjoint_is_zero() {
        let p = [1.0, 0.0, 0.0];
        let q = [0.0, 0.5, 0.5];
        let k = fisher_kernel_categorical(&p, &q);
        assert!((k - 0.0).abs() < 1e-12);
    }

    #[test]
    fn fisher_equals_ppk_half() {
        let p = [0.2, 0.3, 0.5];
        let q = [0.4, 0.1, 0.5];
        let fk = fisher_kernel_categorical(&p, &q);
        let ppk = probability_product_kernel(&p, &q, 0.5);
        assert!((fk - ppk).abs() < 1e-12);
    }

    #[test]
    fn fisher_symmetric() {
        let p = [0.6, 0.3, 0.1];
        let q = [0.2, 0.2, 0.6];
        let k_pq = fisher_kernel_categorical(&p, &q);
        let k_qp = fisher_kernel_categorical(&q, &p);
        assert!((k_pq - k_qp).abs() < 1e-12);
    }
}
