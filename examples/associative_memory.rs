//! Associative Memory example using kernel-based energy functions.
//!
//! Demonstrates:
//! - Memory storage and retrieval
//! - LSE vs LSR energy comparison
//! - Error correction (recovering corrupted patterns)
//! - Novel memory generation with LSR
//!
//! Run: cargo run --example associative_memory

use rkhs::{
    energy_lse, energy_lse_grad, energy_lsr, energy_lsr_grad, epanechnikov, rbf, retrieve_memory,
};

fn main() {
    println!("=== Associative Memory with Kernel Energy Functions ===\n");

    // Store some 2D patterns (memories)
    let memories = vec![
        vec![0.0, 0.0],  // Memory 0: origin
        vec![10.0, 0.0], // Memory 1: right
        vec![5.0, 8.66], // Memory 2: top (equilateral triangle)
    ];

    println!("Stored memories:");
    for (i, m) in memories.iter().enumerate() {
        println!("  ξ[{}] = ({:.2}, {:.2})", i, m[0], m[1]);
    }
    println!();

    // =========================================================================
    // Demo 1: Energy landscape
    // =========================================================================
    println!("--- Energy Landscape Comparison ---\n");

    let test_points = vec![
        (vec![0.0, 0.0], "at memory 0"),
        (vec![10.0, 0.0], "at memory 1"),
        (vec![5.0, 2.89], "centroid"),
        (vec![5.0, 0.0], "between 0 and 1"),
        (vec![20.0, 20.0], "far from all"),
    ];

    let beta = 0.5;

    println!("Point               | E_LSE    | E_LSR    | Notes");
    println!("--------------------|----------|----------|------");
    for (point, desc) in &test_points {
        let e_lse = energy_lse(point, &memories, beta);
        let e_lsr = energy_lsr(point, &memories, beta);
        println!(
            "({:5.1}, {:5.1}) {:12} | {:8.3} | {:8} |",
            point[0],
            point[1],
            desc,
            e_lse,
            if e_lsr.is_finite() {
                format!("{:.3}", e_lsr)
            } else {
                "∞".to_string()
            }
        );
    }
    println!();

    // =========================================================================
    // Demo 2: Memory retrieval (error correction)
    // =========================================================================
    println!("--- Memory Retrieval (Error Correction) ---\n");

    // Corrupt memory 0 with noise
    let corrupted = vec![1.5, 1.2];
    println!(
        "Corrupted query: ({:.2}, {:.2})",
        corrupted[0], corrupted[1]
    );
    println!("(Should recover memory 0 at origin)\n");

    // Retrieve with LSE energy
    let (retrieved_lse, iters_lse) = retrieve_memory(
        corrupted.clone(),
        &memories,
        |v, m| energy_lse_grad(v, m, 2.0),
        0.1,
        200,
        1e-6,
    );

    // Retrieve with LSR energy
    let (retrieved_lsr, iters_lsr) = retrieve_memory(
        corrupted.clone(),
        &memories,
        |v, m| energy_lsr_grad(v, m, 0.1), // Lower beta for larger support
        0.1,
        200,
        1e-6,
    );

    println!(
        "LSE retrieval: ({:.4}, {:.4}) in {} iters",
        retrieved_lse[0], retrieved_lse[1], iters_lse
    );
    println!(
        "LSR retrieval: ({:.4}, {:.4}) in {} iters",
        retrieved_lsr[0], retrieved_lsr[1], iters_lsr
    );

    // Distance to nearest memory
    let dist_lse = (retrieved_lse[0].powi(2) + retrieved_lse[1].powi(2)).sqrt();
    let dist_lsr = (retrieved_lsr[0].powi(2) + retrieved_lsr[1].powi(2)).sqrt();
    println!(
        "Distance to memory 0: LSE={:.4}, LSR={:.4}",
        dist_lse, dist_lsr
    );
    println!();

    // =========================================================================
    // Demo 3: Kernel comparison at different distances
    // =========================================================================
    println!("--- Kernel Response vs Distance ---\n");

    let origin = vec![0.0, 0.0];
    let sigma = 2.0;

    println!("Distance | RBF(σ={})  | Epanechnikov(σ={})", sigma, sigma);
    println!("---------|------------|--------------------");
    for dist in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] {
        let point = vec![dist, 0.0];
        let k_rbf = rbf(&origin, &point, sigma);
        let k_epan = epanechnikov(&origin, &point, sigma);
        println!("{:8.1} | {:10.4} | {:10.4}", dist, k_rbf, k_epan);
    }
    println!();
    println!("Note: Epanechnikov has compact support (zero beyond σ)");
    println!("      RBF decays but never reaches zero");
    println!();

    // =========================================================================
    // Demo 4: Novel memory emergence with LSR
    // =========================================================================
    println!("--- Novel Memory Generation (LSR special property) ---\n");

    // With two close memories, LSR can create a novel minimum between them
    let close_memories = vec![vec![-1.0, 0.0], vec![1.0, 0.0]];

    // Check energy at the midpoint
    let midpoint = vec![0.0, 0.0];
    let beta_lsr = 0.3; // Low beta = large support = overlapping basins

    let e_at_m0 = energy_lsr(&close_memories[0], &close_memories, beta_lsr);
    let e_at_mid = energy_lsr(&midpoint, &close_memories, beta_lsr);
    let e_at_m1 = energy_lsr(&close_memories[1], &close_memories, beta_lsr);

    println!("Two close memories: ξ[0]=(-1,0), ξ[1]=(1,0)");
    println!("With β={} (large support, overlapping basins):", beta_lsr);
    println!();
    println!("  E_LSR at ξ[0]:      {:.4}", e_at_m0);
    println!("  E_LSR at midpoint:  {:.4}", e_at_mid);
    println!("  E_LSR at ξ[1]:      {:.4}", e_at_m1);
    println!();

    if e_at_mid < e_at_m0 && e_at_mid < e_at_m1 {
        println!("Novel memory emerged at midpoint (lower energy than stored memories)!");
    } else {
        println!("Stored memories have lower energy than midpoint.");
    }

    println!();
    println!("This novel memory generation is key to LSR's generative capabilities.");
    println!("See: Hoover et al. (2025) 'Dense Associative Memory with Epanechnikov Energy'");
}
