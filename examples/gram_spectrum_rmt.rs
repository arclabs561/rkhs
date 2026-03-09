//! Spectral analysis of kernel gram matrices using RMT.
//!
//! Computes an RBF gram matrix via rkhs, extracts eigenvalues via a simple
//! Jacobi solver, then uses rmt to classify the spectrum: how many eigenvalues
//! are signal (above the Marchenko-Pastur noise floor) vs noise?
//!
//! Run: cargo run --example gram_spectrum_rmt

use ndarray::Array2;
use rkhs::{kernel_matrix, rbf};
use rmt::{effective_dimension, empirical_spectral_density, mean_spacing_ratio};

fn main() {
    println!("=== Gram Matrix Spectral Analysis via RMT ===\n");

    // Dataset 1: points with genuine 2D structure embedded in 5D
    let structured = vec![
        vec![1.0, 0.0, 0.0, 0.0, 0.0],
        vec![1.1, 0.1, 0.0, 0.0, 0.0],
        vec![0.9, -0.1, 0.0, 0.0, 0.0],
        vec![-1.0, 0.0, 0.0, 0.0, 0.0],
        vec![-0.9, 0.1, 0.0, 0.0, 0.0],
        vec![-1.1, -0.1, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0, 0.0],
        vec![0.1, 1.1, 0.0, 0.0, 0.0],
        vec![-0.1, 0.9, 0.0, 0.0, 0.0],
        vec![0.0, -1.0, 0.0, 0.0, 0.0],
        vec![0.1, -1.1, 0.0, 0.0, 0.0],
        vec![-0.1, -0.9, 0.0, 0.0, 0.0],
    ];

    // Dataset 2: pure noise in 5D (LCG pseudo-random)
    let noise = {
        let mut data = Vec::new();
        let mut state = 42u64;
        for _ in 0..12 {
            let mut point = Vec::new();
            for _ in 0..5 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                point.push((state >> 33) as f64 / (1u64 << 31) as f64 - 0.5);
            }
            data.push(point);
        }
        data
    };

    for (name, data) in [
        ("Structured (2D in 5D)", &structured),
        ("Pure noise (5D)", &noise),
    ] {
        let n = data.len();
        let d = data[0].len();
        let gram = kernel_matrix(data, |a, b| rbf(a, b, 1.0));
        let eigs = eigenvalues_symmetric(&gram);

        let eff_dim = effective_dimension(&eigs, n, d);
        let msr = mean_spacing_ratio(&eigs);
        let regime = if msr > 0.48 {
            "GOE (correlated)"
        } else if msr < 0.42 {
            "Poisson (uncorrelated)"
        } else {
            "intermediate"
        };

        println!("--- {name} ---");
        println!("  Gram matrix: {n}x{n}, bandwidth=1.0");
        println!("  Eigenvalue range: [{:.4}, {:.4}]", eigs[0], eigs[n - 1]);
        println!("  Signal dimensions (above MP): {eff_dim}");
        println!("  Mean spacing ratio: {msr:.4} -> {regime}");

        let (centers, densities) = empirical_spectral_density(&eigs, 8);
        println!("  Spectral density:");
        for (c, d) in centers.iter().zip(densities.iter()) {
            let bar = "#".repeat((d * 15.0) as usize);
            println!("    {c:7.3} | {d:5.3} {bar}");
        }
        println!();
    }
}

/// Jacobi eigenvalue algorithm for small symmetric matrices.
fn eigenvalues_symmetric(m: &Array2<f64>) -> Vec<f64> {
    let n = m.nrows();
    let mut a = m.clone();
    for _ in 0..100 * n * n {
        let mut max_val = 0.0f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                if a[[i, j]].abs() > max_val {
                    max_val = a[[i, j]].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-12 {
            break;
        }
        let theta = if (a[[p, p]] - a[[q, q]]).abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * a[[p, q]] / (a[[p, p]] - a[[q, q]])).atan()
        };
        let c = theta.cos();
        let s = theta.sin();
        let mut new_a = a.clone();
        for i in 0..n {
            if i != p && i != q {
                new_a[[i, p]] = c * a[[i, p]] + s * a[[i, q]];
                new_a[[p, i]] = new_a[[i, p]];
                new_a[[i, q]] = -s * a[[i, p]] + c * a[[i, q]];
                new_a[[q, i]] = new_a[[i, q]];
            }
        }
        new_a[[p, p]] = c * c * a[[p, p]] + 2.0 * s * c * a[[p, q]] + s * s * a[[q, q]];
        new_a[[q, q]] = s * s * a[[p, p]] - 2.0 * s * c * a[[p, q]] + c * c * a[[q, q]];
        new_a[[p, q]] = 0.0;
        new_a[[q, p]] = 0.0;
        a = new_a;
    }
    let mut eigs: Vec<f64> = (0..n).map(|i| a[[i, i]]).collect();
    eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eigs
}
