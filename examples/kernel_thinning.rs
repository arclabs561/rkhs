//! Kernel thinning demo: select a coreset from random 2D points using rkhs + kuji.
//!
//! Generates 200 random 2D points, computes the RBF Gram matrix with rkhs,
//! then uses kuji's kernel_thin to select 20 representative points.
//! Compares the thinned subset's MMD against a random subset of the same size.

use kuji::{kernel_thin, mmd_sq_from_gram};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rkhs::{kernel_matrix, rbf};

fn main() {
    let mut rng = SmallRng::seed_from_u64(42);
    let n = 200;
    let k = 20;

    // Generate random 2D points from two clusters
    let points: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let (cx, cy) = if i < n / 2 { (0.0, 0.0) } else { (3.0, 3.0) };
            vec![
                cx + rng.random_range(-1.0..1.0),
                cy + rng.random_range(-1.0..1.0),
            ]
        })
        .collect();

    // Compute RBF Gram matrix
    let sigma = 1.0;
    let gram_nd = kernel_matrix(&points, |a, b| rbf(a, b, sigma));

    // Flatten to row-major for kuji
    let gram: Vec<f64> = gram_nd.iter().copied().collect();

    // Kernel thinning
    let thinned = kernel_thin(&gram, n, k);
    let mmd_thin = mmd_sq_from_gram(&gram, n, &thinned);

    // Random subset for comparison (5 trials, take the best)
    let mut best_random_mmd = f64::INFINITY;
    for trial in 0..5 {
        let mut indices: Vec<usize> = (0..n).collect();
        // Fisher-Yates shuffle of first k elements
        let mut rng_trial = SmallRng::seed_from_u64(100 + trial);
        for i in 0..k {
            let j = rng_trial.random_range(i..n);
            indices.swap(i, j);
        }
        let random_subset: Vec<usize> = indices[..k].to_vec();
        let mmd_rand = mmd_sq_from_gram(&gram, n, &random_subset);
        if mmd_rand < best_random_mmd {
            best_random_mmd = mmd_rand;
        }
    }

    println!("n = {n}, k = {k}, sigma = {sigma}");
    println!("Thinned MMD^2:         {mmd_thin:.6}");
    println!("Best random MMD^2:     {best_random_mmd:.6}");
    println!("Ratio (thin/random):   {:.3}", mmd_thin / best_random_mmd);

    if mmd_thin < best_random_mmd {
        println!("Kernel thinning produced a better coreset than random selection.");
    } else {
        println!("Random selection happened to match or beat thinning (unusual).");
    }

    // Show which points were selected
    println!("\nSelected indices: {thinned:?}");

    // Count how many from each cluster
    let from_cluster_0 = thinned.iter().filter(|&&i| i < n / 2).count();
    let from_cluster_1 = k - from_cluster_0;
    println!("From cluster 0: {from_cluster_0}, from cluster 1: {from_cluster_1}");
}
