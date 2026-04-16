#![allow(deprecated)]
//! ClAM (associative memory clustering) vs k-means.
//!
//! Compares rkhs ClAM-style clustering (energy-based contraction toward
//! centroids via dense associative memory) against clump's standard k-means.
//! On well-separated clusters they agree; on overlapping clusters ClAM's
//! soft assignment reveals uncertainty that hard k-means hides.
//!
//! Run: cargo run --example clam_vs_kmeans

use clump::Kmeans;
use rkhs::clam::{am_assign, am_soft_assign, clam_loss};

fn main() {
    // 3 clusters in 2D, moderate overlap
    let data: Vec<Vec<f64>> = vec![
        // Cluster 0 (around (-3, 0))
        vec![-3.0, 0.2],
        vec![-2.8, -0.1],
        vec![-3.2, 0.3],
        vec![-2.9, -0.3],
        // Cluster 1 (around (0, 3))
        vec![0.1, 3.0],
        vec![-0.2, 2.8],
        vec![0.3, 3.2],
        vec![0.0, 2.7],
        // Cluster 2 (around (3, 0))
        vec![3.0, -0.1],
        vec![2.8, 0.2],
        vec![3.1, -0.3],
        vec![3.2, 0.1],
        // Ambiguous points (between clusters)
        vec![-1.0, 1.5], // between 0 and 1
        vec![1.5, 1.5],  // between 1 and 2
    ];

    // --- k-means via clump ---
    let data_f32: Vec<Vec<f32>> = data
        .iter()
        .map(|v| v.iter().map(|&x| x as f32).collect())
        .collect();
    let fit = Kmeans::new(3)
        .with_seed(42)
        .fit(&data_f32)
        .expect("k-means converged");

    println!("=== k-means (clump) ===");
    println!("Iterations: {}", fit.iters);
    for (i, c) in fit.centroids.iter().enumerate() {
        println!("  centroid {i}: ({:.2}, {:.2})", c[0], c[1]);
    }
    println!("  labels: {:?}", fit.labels);

    // --- ClAM via rkhs ---
    // Use k-means centroids (upcast to f64) as ClAM centroids
    let centroids: Vec<Vec<f64>> = fit
        .centroids
        .iter()
        .map(|c| c.iter().map(|&x| x as f64).collect())
        .collect();

    // Hard assignment (should match k-means)
    let clam_labels = am_assign(&data, &centroids);
    println!("\n=== ClAM hard assignment (rkhs) ===");
    println!("  labels: {:?}", clam_labels);
    let agree = fit
        .labels
        .iter()
        .zip(clam_labels.iter())
        .filter(|(a, b)| a == b)
        .count();
    println!("  agreement with k-means: {agree}/{}", data.len());

    // Soft assignment reveals uncertainty
    println!("\n=== ClAM soft assignment (beta sweep) ===");
    for &beta in &[0.5, 1.0, 5.0, 20.0] {
        let soft = am_soft_assign(&data, &centroids, beta);
        println!("  beta = {beta:.1}:");
        // Show only the ambiguous points (indices 12, 13)
        for &idx in &[12, 13] {
            let probs = &soft[idx];
            let max_p = probs.iter().cloned().fold(0.0f64, f64::max);
            let entropy: f64 = -probs
                .iter()
                .filter(|&&p| p > 0.0)
                .map(|&p| p * p.ln())
                .sum::<f64>();
            println!(
                "    point {idx} ({:.1},{:.1}): P=[{:.3}, {:.3}, {:.3}]  max={:.3}  H={:.3}",
                data[idx][0], data[idx][1], probs[0], probs[1], probs[2], max_p, entropy
            );
        }
    }

    // ClAM loss: measures how well centroids explain the data
    let loss = clam_loss(&data, &centroids, 5.0, 20, 0.1);
    println!("\n=== ClAM loss ===");
    println!("  loss(beta=5, steps=20, lr=0.1) = {loss:.4}");
    println!("  (lower = data contracts more tightly toward centroids)");
}
