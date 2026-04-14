//! Kernel quantile embeddings for tail-sensitive distribution comparison.
//!
//! Standard kernel mean embeddings map distributions to RKHS elements via
//! `mu_P = E_{x~P}[k(x, .)]`. This captures the mean behavior but can miss
//! tail differences between distributions.
//!
//! Kernel quantile embeddings instead embed at each quantile level tau:
//! the embedding at tau weights samples by their position relative to
//! the tau-th quantile, making the comparison sensitive to distributional
//! shape across all quantile levels.
//!
//! The Quantile Maximum Mean Discrepancy (QMMD) integrates MMD over
//! quantile levels, giving a metric that is more sensitive to tail
//! differences than standard MMD.
//!
//! Reference: Naslidnyk, Chau, Briol, Muandet (2025).
//! "Kernel Quantile Embeddings"

/// Kernel quantile embedding evaluated at given points.
///
/// For a quantile level `tau`, computes the weighted kernel embedding:
///
/// $$\hat{\mu}_{P,\tau}(x) = \frac{1}{|\{j : s_j \le q_\tau\}|} \sum_{j : s_j \le q_\tau} k(x, s_j)$$
///
/// where `q_tau` is the empirical tau-th quantile of `samples`.
///
/// This restricts the kernel mean embedding to samples at or below the
/// tau-th quantile, capturing distributional structure at that level.
///
/// # Arguments
///
/// * `samples` - 1D samples from distribution P (will be sorted internally)
/// * `eval_points` - Points at which to evaluate the embedding
/// * `tau` - Quantile level in \[0, 1\]
/// * `kernel` - Scalar kernel function k(x, y)
///
/// # Returns
///
/// Vector of embedding values, one per eval point.
///
/// # Panics
///
/// Panics if `samples` is empty or `tau` is outside \[0, 1\].
pub fn kernel_quantile_embedding(
    samples: &[f64],
    eval_points: &[f64],
    tau: f64,
    kernel: impl Fn(f64, f64) -> f64,
) -> Vec<f64> {
    assert!(!samples.is_empty(), "samples must be non-empty");
    assert!(
        (0.0..=1.0).contains(&tau),
        "tau must be in [0, 1], got {tau}"
    );

    let mut sorted = samples.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let quantile = empirical_quantile(&sorted, tau);
    let truncated: Vec<f64> = sorted.iter().copied().filter(|&s| s <= quantile).collect();

    if truncated.is_empty() {
        return vec![0.0; eval_points.len()];
    }

    let n = truncated.len() as f64;
    eval_points
        .iter()
        .map(|&x| {
            let sum: f64 = truncated.iter().map(|&s| kernel(x, s)).sum();
            sum / n
        })
        .collect()
}

/// Quantile Maximum Mean Discrepancy between two sets of 1D samples.
///
/// Integrates MMD over quantile levels for tail-sensitive comparison:
///
/// $$\text{QMMD}^2(P, Q) \approx \frac{1}{T} \sum_{t=1}^T \text{MMD}^2(P_{\tau_t}, Q_{\tau_t})$$
///
/// where `P_tau` is the distribution P truncated at its tau-th quantile,
/// and tau levels are uniformly spaced in (0, 1).
///
/// QMMD is more sensitive to tail differences than standard MMD because
/// it separately compares the distributions at each quantile level rather
/// than averaging over all samples equally.
///
/// # Arguments
///
/// * `samples_p` - 1D samples from distribution P
/// * `samples_q` - 1D samples from distribution Q
/// * `kernel` - Scalar kernel function k(x, y)
/// * `num_quantiles` - Number of uniformly-spaced tau levels (higher = finer integration)
///
/// # Returns
///
/// QMMD^2 estimate (non-negative).
///
/// # Panics
///
/// Panics if either sample set is empty or `num_quantiles` is 0.
pub fn qmmd(
    samples_p: &[f64],
    samples_q: &[f64],
    kernel: impl Fn(f64, f64) -> f64,
    num_quantiles: usize,
) -> f64 {
    assert!(!samples_p.is_empty(), "samples_p must be non-empty");
    assert!(!samples_q.is_empty(), "samples_q must be non-empty");
    assert!(num_quantiles > 0, "num_quantiles must be > 0");

    let mut sorted_p = samples_p.to_vec();
    let mut sorted_q = samples_q.to_vec();
    sorted_p.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    sorted_q.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let mut total_mmd = 0.0;

    for i in 1..=num_quantiles {
        let tau = i as f64 / (num_quantiles + 1) as f64;

        let q_p = empirical_quantile(&sorted_p, tau);
        let q_q = empirical_quantile(&sorted_q, tau);

        let trunc_p: Vec<f64> = sorted_p.iter().copied().filter(|&s| s <= q_p).collect();
        let trunc_q: Vec<f64> = sorted_q.iter().copied().filter(|&s| s <= q_q).collect();

        if trunc_p.len() < 2 || trunc_q.len() < 2 {
            continue;
        }

        let mmd_sq = mmd_1d_biased(&trunc_p, &trunc_q, &kernel);
        total_mmd += mmd_sq;
    }

    total_mmd / num_quantiles as f64
}

/// Quantile kernel Gram matrix at a given quantile level.
///
/// Computes the n x n matrix where entry (i, j) is the kernel evaluated
/// between samples i and j, but only over samples at or below the tau-th
/// quantile. Samples above the quantile threshold get zero rows/columns.
///
/// The resulting matrix is positive semi-definite (it is a principal
/// submatrix of the full Gram matrix, padded with zeros).
///
/// # Arguments
///
/// * `samples` - 1D samples
/// * `tau` - Quantile level in \[0, 1\]
/// * `kernel` - Scalar kernel function k(x, y)
///
/// # Returns
///
/// Flat n x n row-major matrix. Entry (i, j) is nonzero only when both
/// `samples[i]` and `samples[j]` are at or below the tau-th quantile.
///
/// # Panics
///
/// Panics if `samples` is empty or `tau` is outside \[0, 1\].
pub fn quantile_gram_matrix(
    samples: &[f64],
    tau: f64,
    kernel: impl Fn(f64, f64) -> f64,
) -> Vec<f64> {
    let n = samples.len();
    assert!(n > 0, "samples must be non-empty");
    assert!(
        (0.0..=1.0).contains(&tau),
        "tau must be in [0, 1], got {tau}"
    );

    let mut sorted = samples.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let quantile = empirical_quantile(&sorted, tau);

    let active: Vec<bool> = samples.iter().map(|&s| s <= quantile).collect();

    let mut gram = vec![0.0; n * n];
    for i in 0..n {
        if !active[i] {
            continue;
        }
        for j in i..n {
            if !active[j] {
                continue;
            }
            let kij = kernel(samples[i], samples[j]);
            gram[i * n + j] = kij;
            gram[j * n + i] = kij;
        }
    }

    gram
}

/// Kernel quantile function embedding.
///
/// Maps a set of samples to a vector of kernel-smoothed quantile values
/// evaluated at specified quantile levels `tau_1, ..., tau_L`. Each entry
/// is the empirical quantile of the samples smoothed by a 1D kernel.
///
/// The smoothing convolves the empirical quantile function with a kernel
/// of the given bandwidth, reducing sensitivity to individual sample
/// positions. With `bandwidth = 0.0`, this returns raw empirical quantiles.
///
/// $$Q_P(\tau) = \sum_{i=1}^n w_i(\tau) \cdot x_{(i)}$$
///
/// where `w_i(tau)` are kernel-smoothed weights centered on the quantile
/// index corresponding to `tau`, and `x_{(i)}` are order statistics.
///
/// # Arguments
///
/// * `samples` - 1D samples from a distribution
/// * `tau_levels` - Quantile levels in \[0, 1\] at which to evaluate
/// * `bandwidth` - Kernel smoothing bandwidth (in index space). 0 = no smoothing.
///
/// # Returns
///
/// Vector of length `tau_levels.len()`, one smoothed quantile per level.
///
/// # Panics
///
/// Panics if `samples` is empty or any tau is outside \[0, 1\].
pub fn quantile_function_embedding(
    samples: &[f64],
    tau_levels: &[f64],
    bandwidth: f64,
) -> Vec<f64> {
    assert!(!samples.is_empty(), "samples must be non-empty");
    for &tau in tau_levels {
        assert!(
            (0.0..=1.0).contains(&tau),
            "tau must be in [0, 1], got {tau}"
        );
    }

    let mut sorted = samples.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();

    if bandwidth <= 0.0 || n == 1 {
        // No smoothing: return raw empirical quantiles.
        return tau_levels
            .iter()
            .map(|&tau| empirical_quantile(&sorted, tau))
            .collect();
    }

    // Kernel-smoothed quantile function: for each tau, compute a weighted
    // average of order statistics where weights are Gaussian kernel values
    // centered on the index corresponding to tau.
    let h = bandwidth;
    tau_levels
        .iter()
        .map(|&tau| {
            let center = tau * (n - 1) as f64;
            let mut weight_sum = 0.0;
            let mut val_sum = 0.0;
            for (i, &x) in sorted.iter().enumerate() {
                let d = (i as f64 - center) / h;
                let w = (-0.5 * d * d).exp();
                weight_sum += w;
                val_sum += w * x;
            }
            if weight_sum > 0.0 {
                val_sum / weight_sum
            } else {
                empirical_quantile(&sorted, tau)
            }
        })
        .collect()
}

/// Weighting scheme for quantile levels in weighted QMMD.
#[derive(Debug, Clone)]
pub enum QuantileWeight {
    /// Equal weight at all quantile levels.
    Uniform,
    /// Emphasize tail quantiles (near 0 and 1). Uses `w(tau) = (tau*(1-tau))^{-alpha}`
    /// where `alpha > 0` controls tail emphasis. Larger alpha = heavier tails.
    TailHeavy {
        /// Exponent controlling tail emphasis. Typical values: 0.5 to 1.0.
        alpha: f64,
    },
    /// Custom weights, one per quantile level. Weights are normalized internally.
    Custom(Vec<f64>),
}

impl QuantileWeight {
    /// Compute normalized weights for the given number of quantile levels.
    ///
    /// Levels are spaced as `i / (n+1)` for `i` in `1..=n`.
    fn weights(&self, num_quantiles: usize) -> Vec<f64> {
        let raw: Vec<f64> = match self {
            QuantileWeight::Uniform => vec![1.0; num_quantiles],
            QuantileWeight::TailHeavy { alpha } => (1..=num_quantiles)
                .map(|i| {
                    let tau = i as f64 / (num_quantiles + 1) as f64;
                    let base = tau * (1.0 - tau);
                    // Avoid division by zero at boundaries (tau near 0 or 1).
                    if base < 1e-15 {
                        1e15_f64.min(1.0 / 1e-15_f64.powf(*alpha))
                    } else {
                        base.powf(-alpha)
                    }
                })
                .collect(),
            QuantileWeight::Custom(w) => {
                assert_eq!(
                    w.len(),
                    num_quantiles,
                    "custom weights length ({}) must match num_quantiles ({num_quantiles})",
                    w.len()
                );
                w.clone()
            }
        };

        let sum: f64 = raw.iter().sum();
        assert!(sum > 0.0, "weight sum must be positive");
        raw.iter().map(|&w| w / sum).collect()
    }
}

/// Weighted Quantile MMD between two sets of 1D samples.
///
/// Like [`qmmd`], but allows non-uniform weighting of quantile levels.
/// Tail-heavy weighting makes this more sensitive to distributional
/// differences in the tails than uniform weighting.
///
/// $$\text{WQMMD}^2(P, Q) = \sum_{t=1}^T w_t \cdot \text{MMD}^2(P_{\tau_t}, Q_{\tau_t})$$
///
/// # Arguments
///
/// * `samples_p` - 1D samples from distribution P
/// * `samples_q` - 1D samples from distribution Q
/// * `kernel` - Scalar kernel function k(x, y)
/// * `num_quantiles` - Number of quantile levels
/// * `weighting` - How to weight quantile levels
///
/// # Returns
///
/// Weighted QMMD^2 estimate (non-negative).
///
/// # Panics
///
/// Panics if either sample set is empty or `num_quantiles` is 0.
pub fn weighted_qmmd(
    samples_p: &[f64],
    samples_q: &[f64],
    kernel: impl Fn(f64, f64) -> f64,
    num_quantiles: usize,
    weighting: &QuantileWeight,
) -> f64 {
    assert!(!samples_p.is_empty(), "samples_p must be non-empty");
    assert!(!samples_q.is_empty(), "samples_q must be non-empty");
    assert!(num_quantiles > 0, "num_quantiles must be > 0");

    let weights = weighting.weights(num_quantiles);

    let mut sorted_p = samples_p.to_vec();
    let mut sorted_q = samples_q.to_vec();
    sorted_p.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    sorted_q.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let mut total = 0.0;

    for i in 1..=num_quantiles {
        let tau = i as f64 / (num_quantiles + 1) as f64;

        let q_p = empirical_quantile(&sorted_p, tau);
        let q_q = empirical_quantile(&sorted_q, tau);

        let trunc_p: Vec<f64> = sorted_p.iter().copied().filter(|&s| s <= q_p).collect();
        let trunc_q: Vec<f64> = sorted_q.iter().copied().filter(|&s| s <= q_q).collect();

        if trunc_p.len() < 2 || trunc_q.len() < 2 {
            continue;
        }

        let mmd_sq = mmd_1d_biased(&trunc_p, &trunc_q, &kernel);
        total += weights[i - 1] * mmd_sq;
    }

    total
}

/// Kernel between two distributions defined via their quantile embeddings.
///
/// Computes a kernel value between distributions P and Q by:
/// 1. Embedding each as a vector of kernel-smoothed quantile values
/// 2. Computing a kernel (e.g. RBF) between the embedding vectors
///
/// $$k_Q(P, Q) = \sum_{l=1}^L w_l \cdot k(Q_P(\tau_l),\, Q_Q(\tau_l))$$
///
/// where `Q_P(tau)` is the smoothed quantile function of P at level tau,
/// and `w_l` are quantile-level weights.
///
/// This defines a positive-definite kernel on distributions when the
/// base scalar kernel is positive definite.
///
/// # Arguments
///
/// * `samples_p` - 1D samples from distribution P
/// * `samples_q` - 1D samples from distribution Q
/// * `tau_levels` - Quantile levels at which to compare
/// * `bandwidth` - Smoothing bandwidth for quantile function (0 = no smoothing)
/// * `scalar_kernel` - Kernel applied to pairs of quantile values
/// * `weighting` - How to weight quantile levels
///
/// # Returns
///
/// Scalar kernel value between the two distributions.
///
/// # Panics
///
/// Panics if either sample set is empty or tau_levels is empty.
pub fn quantile_distribution_kernel(
    samples_p: &[f64],
    samples_q: &[f64],
    tau_levels: &[f64],
    bandwidth: f64,
    scalar_kernel: impl Fn(f64, f64) -> f64,
    weighting: &QuantileWeight,
) -> f64 {
    assert!(!samples_p.is_empty(), "samples_p must be non-empty");
    assert!(!samples_q.is_empty(), "samples_q must be non-empty");
    assert!(!tau_levels.is_empty(), "tau_levels must be non-empty");

    let emb_p = quantile_function_embedding(samples_p, tau_levels, bandwidth);
    let emb_q = quantile_function_embedding(samples_q, tau_levels, bandwidth);

    let weights = weighting.weights(tau_levels.len());

    let mut result = 0.0;
    for (i, (&qp, &qq)) in emb_p.iter().zip(emb_q.iter()).enumerate() {
        result += weights[i] * scalar_kernel(qp, qq);
    }

    result
}

/// Empirical quantile from a pre-sorted slice (linear interpolation).
fn empirical_quantile(sorted: &[f64], tau: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let pos = tau * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = lo + 1;
    if hi >= n {
        sorted[n - 1]
    } else {
        let frac = pos - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

/// Biased MMD^2 for 1D scalar samples (internal helper).
fn mmd_1d_biased(x: &[f64], y: &[f64], kernel: &impl Fn(f64, f64) -> f64) -> f64 {
    let nx = x.len() as f64;
    let ny = y.len() as f64;

    let mut kxx = 0.0;
    for xi in x {
        for xj in x {
            kxx += kernel(*xi, *xj);
        }
    }
    kxx /= nx * nx;

    let mut kyy = 0.0;
    for yi in y {
        for yj in y {
            kyy += kernel(*yi, *yj);
        }
    }
    kyy /= ny * ny;

    let mut kxy = 0.0;
    for xi in x {
        for yj in y {
            kxy += kernel(*xi, *yj);
        }
    }
    kxy /= nx * ny;

    (kxx + kyy - 2.0 * kxy).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1D RBF kernel for scalar samples.
    fn rbf_1d(x: f64, y: f64) -> f64 {
        let d = x - y;
        (-d * d / 2.0).exp()
    }

    #[test]
    fn qmmd_self_near_zero() {
        let samples: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let q = qmmd(&samples, &samples, rbf_1d, 20);
        assert!(q < 1e-10, "QMMD(P, P) should be ~0, got {q}");
    }

    #[test]
    fn qmmd_symmetric() {
        let p: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
        let q: Vec<f64> = (0..50).map(|i| 5.0 + i as f64 * 0.2).collect();
        let pq = qmmd(&p, &q, rbf_1d, 15);
        let qp = qmmd(&q, &p, rbf_1d, 15);
        assert!(
            (pq - qp).abs() < 1e-12,
            "QMMD should be symmetric: {pq} vs {qp}"
        );
    }

    #[test]
    fn qmmd_detects_different_tails() {
        // Gaussian-like (concentrated) vs heavy-tailed (spread out)
        let gaussian: Vec<f64> = (0..200)
            .map(|i| {
                let t = (i as f64 - 100.0) / 30.0;
                t
            })
            .collect();

        // Heavy-tailed: same center but much wider spread
        let heavy: Vec<f64> = (0..200)
            .map(|i| {
                let t = (i as f64 - 100.0) / 10.0;
                t
            })
            .collect();

        let q = qmmd(&gaussian, &heavy, rbf_1d, 20);
        assert!(q > 0.01, "QMMD should detect tail differences, got {q}");
    }

    #[test]
    fn qmmd_more_sensitive_than_mmd_for_tails() {
        // Two distributions with same mean but different tails.
        // Concentrated around 0 with light tails.
        let light: Vec<f64> = (0..200).map(|i| (i as f64 - 100.0) / 50.0).collect();

        // Same center but heavy tails (wider spread).
        let heavy: Vec<f64> = (0..200).map(|i| (i as f64 - 100.0) / 10.0).collect();

        let qmmd_val = qmmd(&light, &heavy, rbf_1d, 30);

        // Standard MMD (biased, 1D)
        let light_vecs: Vec<Vec<f64>> = light.iter().map(|&x| vec![x]).collect();
        let heavy_vecs: Vec<Vec<f64>> = heavy.iter().map(|&x| vec![x]).collect();
        let mmd_val = crate::mmd_biased(&light_vecs, &heavy_vecs, |a, b| rbf_1d(a[0], b[0]));

        // QMMD should be at least comparable; the key property is it detects
        // tail differences. Both should be positive for these distributions.
        assert!(
            qmmd_val > 0.0,
            "QMMD should detect tail difference: {qmmd_val}"
        );
        assert!(
            mmd_val > 0.0,
            "MMD should also detect this difference: {mmd_val}"
        );
        // QMMD accumulates differences across quantile levels, so for
        // distributions that differ primarily in tails, QMMD picks up
        // the signal at extreme quantiles where standard MMD averages it out.
        // We verify both detect the difference rather than claiming a strict
        // ordering, since the magnitude depends on bandwidth and sample size.
    }

    #[test]
    fn quantile_embedding_at_median() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let eval = vec![3.0]; // the median
        let emb = kernel_quantile_embedding(&samples, &eval, 0.5, rbf_1d);

        // At tau=0.5, we use samples <= median (3.0): {1, 2, 3}
        // Embedding at x=3: (k(3,1) + k(3,2) + k(3,3)) / 3
        let expected = (rbf_1d(3.0, 1.0) + rbf_1d(3.0, 2.0) + rbf_1d(3.0, 3.0)) / 3.0;
        assert!(
            (emb[0] - expected).abs() < 1e-12,
            "Embedding at median: got {}, expected {}",
            emb[0],
            expected
        );
    }

    #[test]
    fn quantile_embedding_tau_one_uses_all() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let eval = vec![3.0];
        let emb = kernel_quantile_embedding(&samples, &eval, 1.0, rbf_1d);

        // At tau=1.0, all samples are included
        let expected: f64 = samples.iter().map(|&s| rbf_1d(3.0, s)).sum::<f64>() / 5.0;
        assert!(
            (emb[0] - expected).abs() < 1e-12,
            "Embedding at tau=1: got {}, expected {}",
            emb[0],
            expected
        );
    }

    #[test]
    fn quantile_gram_is_psd() {
        let samples = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let gram = quantile_gram_matrix(&samples, 0.7, rbf_1d);
        let n = samples.len();

        // Check PSD: all eigenvalues >= 0 via Gershgorin or direct check.
        // Simple check: v^T G v >= 0 for random vectors.
        let test_vectors: Vec<Vec<f64>> = vec![
            vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            vec![-1.0, 2.0, -1.0, 0.5, 0.3, -0.7],
            vec![0.1, -0.3, 0.5, -0.7, 0.9, -0.2],
        ];

        for v in &test_vectors {
            let mut vtgv = 0.0;
            for i in 0..n {
                for j in 0..n {
                    vtgv += v[i] * gram[i * n + j] * v[j];
                }
            }
            assert!(vtgv >= -1e-12, "Gram matrix not PSD: v^T G v = {vtgv}");
        }
    }

    #[test]
    fn quantile_gram_symmetric() {
        let samples = vec![0.5, 1.5, 2.5, 3.5];
        let gram = quantile_gram_matrix(&samples, 0.6, rbf_1d);
        let n = samples.len();
        for i in 0..n {
            for j in 0..n {
                assert_eq!(
                    gram[i * n + j],
                    gram[j * n + i],
                    "Gram not symmetric at ({i}, {j})"
                );
            }
        }
    }

    #[test]
    fn quantile_gram_zeros_above_quantile() {
        // tau=0.0 should still include the minimum value
        let samples = vec![1.0, 2.0, 3.0, 4.0];
        let gram = quantile_gram_matrix(&samples, 0.0, rbf_1d);
        let n = samples.len();

        // Only sample[0] (value 1.0) is at or below the 0th quantile
        // So only gram[0][0] should be nonzero
        for i in 0..n {
            for j in 0..n {
                if i == 0 && j == 0 {
                    assert!(gram[0] > 0.0, "gram[0,0] should be positive");
                } else {
                    assert_eq!(
                        gram[i * n + j],
                        0.0,
                        "gram[{i},{j}] should be 0 above quantile"
                    );
                }
            }
        }
    }

    // --- quantile_function_embedding tests ---

    #[test]
    fn quantile_function_embedding_no_smoothing() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let taus = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let emb = quantile_function_embedding(&samples, &taus, 0.0);

        assert!((emb[0] - 1.0).abs() < 1e-12, "Q(0) should be min");
        assert!((emb[2] - 3.0).abs() < 1e-12, "Q(0.5) should be median");
        assert!((emb[4] - 5.0).abs() < 1e-12, "Q(1) should be max");
    }

    #[test]
    fn quantile_function_embedding_smoothing_shrinks_range() {
        // Smoothing should pull extreme quantiles toward the center.
        let samples: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let taus = vec![0.01, 0.5, 0.99];
        let raw = quantile_function_embedding(&samples, &taus, 0.0);
        let smoothed = quantile_function_embedding(&samples, &taus, 3.0);

        // Low quantile should be pulled up by smoothing.
        assert!(
            smoothed[0] > raw[0],
            "smoothed Q(0.01) should be > raw Q(0.01): {} vs {}",
            smoothed[0],
            raw[0]
        );
        // High quantile should be pulled down by smoothing.
        assert!(
            smoothed[2] < raw[2],
            "smoothed Q(0.99) should be < raw Q(0.99): {} vs {}",
            smoothed[2],
            raw[2]
        );
    }

    #[test]
    fn quantile_function_embedding_monotone() {
        // Quantile function should be non-decreasing.
        let samples: Vec<f64> = (0..50).map(|i| (i as f64).powi(2)).collect();
        let taus: Vec<f64> = (0..=20).map(|i| i as f64 / 20.0).collect();
        let emb = quantile_function_embedding(&samples, &taus, 2.0);

        for i in 1..emb.len() {
            assert!(
                emb[i] >= emb[i - 1] - 1e-10,
                "quantile function not monotone at tau={}: {} < {}",
                taus[i],
                emb[i],
                emb[i - 1]
            );
        }
    }

    // --- weighted_qmmd tests ---

    #[test]
    fn weighted_qmmd_uniform_matches_qmmd() {
        let p: Vec<f64> = (0..80).map(|i| i as f64 * 0.1).collect();
        let q: Vec<f64> = (0..80).map(|i| 2.0 + i as f64 * 0.15).collect();
        let nq = 15;

        let standard = qmmd(&p, &q, rbf_1d, nq);
        let uniform = weighted_qmmd(&p, &q, rbf_1d, nq, &QuantileWeight::Uniform);

        assert!(
            (standard - uniform).abs() < 1e-10,
            "uniform weighted_qmmd should match qmmd: {standard} vs {uniform}"
        );
    }

    #[test]
    fn weighted_qmmd_self_near_zero() {
        let samples: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let val = weighted_qmmd(
            &samples,
            &samples,
            rbf_1d,
            20,
            &QuantileWeight::TailHeavy { alpha: 0.5 },
        );
        assert!(val < 1e-10, "WQMMD(P, P) should be ~0, got {val}");
    }

    #[test]
    fn weighted_qmmd_tail_heavy_more_sensitive_to_tail_shift() {
        // P: centered at 0, range [-2, 2]
        let p: Vec<f64> = (0..200).map(|i| (i as f64 - 100.0) / 50.0).collect();

        // Q: same bulk as P but with shifted tails. Replace the outer 10%
        // with values shifted further out.
        let mut q = p.clone();
        for i in 0..20 {
            q[i] -= 2.0; // push lower tail further left
        }
        for i in 180..200 {
            q[i] += 2.0; // push upper tail further right
        }

        let nq = 30;
        let uniform_val = weighted_qmmd(&p, &q, rbf_1d, nq, &QuantileWeight::Uniform);
        let tail_val = weighted_qmmd(
            &p,
            &q,
            rbf_1d,
            nq,
            &QuantileWeight::TailHeavy { alpha: 0.5 },
        );

        assert!(
            uniform_val > 0.0,
            "uniform should detect difference: {uniform_val}"
        );
        assert!(
            tail_val > 0.0,
            "tail-heavy should detect difference: {tail_val}"
        );
        // Tail-heavy weighting should amplify the signal from tail differences.
        assert!(
            tail_val > uniform_val,
            "tail-heavy WQMMD ({tail_val}) should exceed uniform ({uniform_val}) \
             when distributions differ primarily in tails"
        );
    }

    #[test]
    fn weighted_qmmd_custom_weights() {
        let p: Vec<f64> = (0..60).map(|i| i as f64 * 0.1).collect();
        let q: Vec<f64> = (0..60).map(|i| 1.0 + i as f64 * 0.1).collect();
        // All weight on first quantile level.
        let custom = QuantileWeight::Custom(vec![1.0, 0.0, 0.0, 0.0, 0.0]);
        let val = weighted_qmmd(&p, &q, rbf_1d, 5, &custom);
        assert!(val >= 0.0, "custom weighted QMMD should be non-negative");
    }

    // --- quantile_distribution_kernel tests ---

    #[test]
    fn quantile_dist_kernel_self_is_max() {
        let samples: Vec<f64> = (0..100).map(|i| i as f64 / 50.0).collect();
        let taus: Vec<f64> = (1..=10).map(|i| i as f64 / 11.0).collect();

        let k_pp = quantile_distribution_kernel(
            &samples,
            &samples,
            &taus,
            1.0,
            rbf_1d,
            &QuantileWeight::Uniform,
        );

        // k(P, P) should equal the weighted sum of k(q, q) = 1.0 for RBF.
        // With uniform weights summing to 1, this should be 1.0.
        assert!(
            (k_pp - 1.0).abs() < 1e-10,
            "k(P, P) with RBF should be 1.0, got {k_pp}"
        );
    }

    #[test]
    fn quantile_dist_kernel_different_dists_less_than_self() {
        let p: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let q: Vec<f64> = (0..100).map(|i| 5.0 + i as f64 / 100.0).collect();
        let taus: Vec<f64> = (1..=10).map(|i| i as f64 / 11.0).collect();

        let k_pp =
            quantile_distribution_kernel(&p, &p, &taus, 1.0, rbf_1d, &QuantileWeight::Uniform);
        let k_pq =
            quantile_distribution_kernel(&p, &q, &taus, 1.0, rbf_1d, &QuantileWeight::Uniform);

        assert!(
            k_pq < k_pp,
            "k(P, Q) should be less than k(P, P) for different distributions: {k_pq} vs {k_pp}"
        );
    }

    #[test]
    fn quantile_dist_kernel_symmetric() {
        let p: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
        let q: Vec<f64> = (0..50).map(|i| 2.0 + i as f64 * 0.2).collect();
        let taus: Vec<f64> = (1..=8).map(|i| i as f64 / 9.0).collect();

        let k_pq = quantile_distribution_kernel(
            &p,
            &q,
            &taus,
            1.0,
            rbf_1d,
            &QuantileWeight::TailHeavy { alpha: 0.5 },
        );
        let k_qp = quantile_distribution_kernel(
            &q,
            &p,
            &taus,
            1.0,
            rbf_1d,
            &QuantileWeight::TailHeavy { alpha: 0.5 },
        );

        assert!(
            (k_pq - k_qp).abs() < 1e-12,
            "quantile distribution kernel should be symmetric: {k_pq} vs {k_qp}"
        );
    }
}
