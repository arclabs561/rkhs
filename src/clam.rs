//! ClAM: Clustering with Associative Memory helpers.
//!
//! Based on Saha et al. (2023), "End-to-End Differentiable Clustering with AM".

use crate::{energy_lse_grad, retrieve_memory};

/// Hard assignment of points to cluster centroids using AM energy.
pub fn am_assign(data: &[Vec<f64>], centroids: &[Vec<f64>], beta: f64) -> Vec<usize> {
    data.iter()
        .map(|point| {
            let mut min_dist_sq = f64::MAX;
            let mut best_label = 0;
            for (i, centroid) in centroids.iter().enumerate() {
                let dist_sq: f64 = point
                    .iter()
                    .zip(centroid.iter())
                    .map(|(p, c)| (p - c).powi(2))
                    .sum();
                if dist_sq < min_dist_sq {
                    min_dist_sq = dist_sq;
                    best_label = i;
                }
            }
            best_label
        })
        .collect()
}

/// Soft assignment (probabilities) of points to cluster centroids.
///
/// P(i) = exp(-beta/2 ||x - c_i||^2) / Σ_j exp(-beta/2 ||x - c_j||^2)
pub fn am_soft_assign(data: &[Vec<f64>], centroids: &[Vec<f64>], beta: f64) -> Vec<Vec<f64>> {
    data.iter()
        .map(|point| {
            let neg_half_beta = -0.5 * beta;
            let log_weights: Vec<f64> = centroids
                .iter()
                .map(|c| {
                    let sq_dist: f64 = point
                        .iter()
                        .zip(c.iter())
                        .map(|(pi, ci)| (pi - ci).powi(2))
                        .sum();
                    neg_half_beta * sq_dist
                })
                .collect();

            let max_log = log_weights
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let exp_weights: Vec<f64> = log_weights.iter().map(|&w| (w - max_log).exp()).collect();
            let sum_exp: f64 = exp_weights.iter().sum();

            exp_weights.iter().map(|&w| w / sum_exp).collect()
        })
        .collect()
}

/// Contract a point toward centroids using AM energy descent.
pub fn am_contract(
    point: &[f64],
    centroids: &[Vec<f64>],
    beta: f64,
    steps: usize,
    lr: f64,
) -> Vec<f64> {
    let (contracted, _) = retrieve_memory(
        point.to_vec(),
        centroids,
        |v, m| energy_lse_grad(v, m, beta),
        lr,
        steps,
        1e-10,
    );
    contracted
}

/// Differentiable clustering loss (ClAM loss).
///
/// L = Σ ||x_i - contract(x_i, centroids)||^2
pub fn clam_loss(
    data: &[Vec<f64>],
    centroids: &[Vec<f64>],
    beta: f64,
    steps: usize,
    lr: f64,
) -> f64 {
    data.iter()
        .map(|point| {
            let contracted = am_contract(point, centroids, beta, steps, lr);
            point
                .iter()
                .zip(contracted.iter())
                .map(|(p, c)| (p - c).powi(2))
                .sum::<f64>()
        })
        .sum()
}
