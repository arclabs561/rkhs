//! ClAM: Clustering with Associative Memory
//!
//! Demonstrates AM-based clustering from Section 5.3 of the tutorial.
//!
//! Key insight: AM energy descent contracts points toward cluster centers.
//! This makes clustering end-to-end differentiable (unlike k-means).
//!
//! Run: cargo run --example clam_clustering
// This example depends on experimental "ClAM" helpers (hard/soft assignment,
// contraction, and a differentiable clustering loss). Those helpers are not part
// of the default `rkhs` API surface.
//
// To avoid breaking `cargo test` / `cargo build` for the crate, the full example
// is behind a feature gate.

#[cfg(not(feature = "clam"))]
fn main() {
    eprintln!(
        "rkhs example `clam_clustering` is disabled by default.\n\
Enable it with: `cargo run -p rkhs --example clam_clustering --features clam`"
    );
}

#[cfg(feature = "clam")]
fn main() {
    use rkhs::{am_assign, am_contract, am_soft_assign, clam_loss, energy_lse};

    println!("=== ClAM: Clustering with Associative Memory ===\n");

    // Generate synthetic clustered data
    let cluster_1: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0],
        vec![0.5, 0.2],
        vec![0.2, 0.5],
        vec![0.3, 0.3],
    ];

    let cluster_2: Vec<Vec<f64>> = vec![
        vec![5.0, 5.0],
        vec![5.3, 4.8],
        vec![4.8, 5.2],
        vec![5.1, 5.1],
    ];

    let cluster_3: Vec<Vec<f64>> = vec![
        vec![0.0, 5.0],
        vec![0.2, 4.8],
        vec![0.3, 5.3],
        vec![0.1, 5.1],
    ];

    let data: Vec<Vec<f64>> = cluster_1
        .iter()
        .chain(cluster_2.iter())
        .chain(cluster_3.iter())
        .cloned()
        .collect();

    println!("Data: {} points in 3 clusters\n", data.len());

    // Initialize centroids (in practice, use k-means++ or random)
    let centroids = vec![
        vec![0.1, 0.1], // Near cluster 1
        vec![4.9, 5.1], // Near cluster 2
        vec![0.1, 4.9], // Near cluster 3
    ];

    println!("Initial centroids:");
    for (i, c) in centroids.iter().enumerate() {
        println!("  c[{}] = ({:.2}, {:.2})", i, c[0], c[1]);
    }
    println!();

    // =========================================================================
    // Demo 1: Hard assignment (like k-means)
    // =========================================================================
    println!("--- Hard Assignment ---\n");

    let labels = am_assign(&data, &centroids, 2.0);

    println!("Point assignments:");
    for (i, label) in labels.iter().enumerate() {
        let point = &data[i];
        let expected = if i < 4 {
            0
        } else if i < 8 {
            1
        } else {
            2
        };
        let correct = if *label == expected { "✓" } else { "✗" };
        println!(
            "  ({:.1}, {:.1}) → cluster {} {}",
            point[0], point[1], label, correct
        );
    }
    println!();

    // =========================================================================
    // Demo 2: Soft assignment (probabilities)
    // =========================================================================
    println!("--- Soft Assignment ---\n");

    let probs = am_soft_assign(&data, &centroids, 1.0);

    println!("First 3 points (cluster 1) probabilities:");
    for i in 0..3 {
        println!(
            "  ({:.1}, {:.1}) → P = [{:.3}, {:.3}, {:.3}]",
            data[i][0], data[i][1], probs[i][0], probs[i][1], probs[i][2]
        );
    }
    println!();

    // =========================================================================
    // Demo 3: Contraction (AM energy descent)
    // =========================================================================
    println!("--- Contraction (Energy Descent) ---\n");

    let beta = 2.0;
    let steps = 20;
    let lr = 0.1;

    println!("Contracting points toward centroids:");
    for i in [0, 4, 8].iter() {
        let point = &data[*i];
        let contracted = am_contract(point, &centroids, beta, steps, lr);

        let dist_before: f64 = centroids
            .iter()
            .map(|c| {
                point
                    .iter()
                    .zip(c.iter())
                    .map(|(p, c)| (p - c).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .fold(f64::MAX, f64::min);

        let dist_after: f64 = centroids
            .iter()
            .map(|c| {
                contracted
                    .iter()
                    .zip(c.iter())
                    .map(|(p, c)| (p - c).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .fold(f64::MAX, f64::min);

        println!(
            "  ({:.1}, {:.1}) → ({:.3}, {:.3})  [dist to nearest: {:.3} → {:.3}]",
            point[0], point[1], contracted[0], contracted[1], dist_before, dist_after
        );
    }
    println!();

    // =========================================================================
    // Demo 4: ClAM loss (for gradient-based optimization)
    // =========================================================================
    println!("--- ClAM Loss ---\n");

    let loss = clam_loss(&data, &centroids, beta, steps, lr);
    println!("ClAM loss with current centroids: {:.4}", loss);

    // Compare with different centroids
    let bad_centroids = vec![
        vec![2.5, 2.5], // Bad: between clusters
        vec![2.5, 2.6],
        vec![2.6, 2.5],
    ];

    let bad_loss = clam_loss(&data, &bad_centroids, beta, steps, lr);
    println!("ClAM loss with bad centroids:     {:.4}", bad_loss);
    println!();

    assert!(loss < bad_loss, "Good centroids should have lower loss");

    // =========================================================================
    // Demo 5: Energy landscape visualization
    // =========================================================================
    println!("--- Energy Landscape (1D slice) ---\n");

    println!("Energy along y=2.5 (passes through cluster regions):");
    println!("   x   | E_LSE");
    println!("-------|--------");

    for x in (0..=50).map(|i| i as f64 * 0.1) {
        let e = energy_lse(&[x, 2.5], &centroids, 1.0);
        let bar_len = ((50.0 - e).max(0.0) / 5.0) as usize;
        let bar = "█".repeat(bar_len.min(20));
        println!("{:5.1} | {:6.2} {}", x, e, bar);
    }
    println!();

    // =========================================================================
    // Summary
    // =========================================================================
    println!("=== Summary ===\n");
    println!("ClAM provides end-to-end differentiable clustering by:");
    println!("1. Treating cluster centroids as 'memories'");
    println!("2. Using AM energy descent for soft/hard assignment");
    println!("3. Minimizing contraction distance as loss");
    println!();
    println!("Advantages over k-means:");
    println!("- Fully differentiable (can backprop through clustering)");
    println!("- Natural soft assignments via energy-based probabilities");
    println!("- Can incorporate into neural network training");
    println!();
    println!("See: Saha et al. (2023) 'End-to-End Differentiable Clustering with AM'");
}
