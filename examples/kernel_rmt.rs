//! Kernel Gram matrix spectrum and MMD power loss via RMT.
//!
//! Builds RBF Gram matrices at varying p/n ratios and compares
//! their eigenvalue distributions against the Marchenko-Pastur
//! prediction. Demonstrates that as p/n -> 1, the Gram spectrum
//! concentrates and MMD loses statistical power.
//!
//! The existing `gram_spectrum_rmt` example shows signal-vs-noise
//! separation. This example focuses on the p/n ratio regime and
//! its consequences for two-sample testing.
//!
//! Run: cargo run --example kernel_rmt

use ndarray::Array2;
use rkhs::{kernel_matrix, mmd_unbiased, rbf};
use rmt::{
    empirical_spectral_density, marchenko_pastur_density, marchenko_pastur_support,
    mean_spacing_ratio,
};

fn main() {
    println!("=== Kernel Gram Matrix Spectrum and MMD Power via RMT ===\n");

    // ---------------------------------------------------------------
    // Part 1: Gram eigenvalue distribution vs Marchenko-Pastur
    // ---------------------------------------------------------------
    println!("--- Part 1: Gram spectrum vs MP at different p/n ratios ---\n");

    let seed = 123u64;
    let bandwidth = 1.0;

    // For random Gaussian data, the centered Gram matrix K - 11^T K / n
    // has eigenvalues that (after rescaling) follow the Marchenko-Pastur
    // distribution with gamma = p/n.
    for (n, p) in [(60, 10), (60, 30), (60, 50)] {
        let gamma = p as f64 / n as f64;
        let data = lcg_gaussian_data(n, p, seed);

        let gram = kernel_matrix(&data, |a, b| rbf(a, b, bandwidth));
        let eigs = eigenvalues_symmetric(&gram);
        let msr = mean_spacing_ratio(&eigs);

        // Rescale eigenvalues to match MP normalization
        let mean_eig: f64 = eigs.iter().sum::<f64>() / n as f64;
        let rescaled: Vec<f64> = eigs.iter().map(|&e| e / mean_eig).collect();

        let (mp_lo, mp_hi) = marchenko_pastur_support(gamma, 1.0);

        println!("  p/n = {p}/{n} = {gamma:.2}");
        println!("  Eigenvalue range: [{:.4}, {:.4}]", eigs[0], eigs[n - 1]);
        println!("  Mean spacing ratio: {msr:.4}");
        println!("  MP support (rescaled): [{mp_lo:.4}, {mp_hi:.4}]");

        let bins = 10;
        let (centers, densities) = empirical_spectral_density(&rescaled, bins);
        println!("  {:>8} | {:>8} {:>8}", "lambda", "empirical", "MP");
        for (c, emp_d) in centers.iter().zip(densities.iter()) {
            let mp_d = marchenko_pastur_density(*c, gamma, 1.0);
            let bar = "#".repeat((emp_d * 10.0).min(30.0) as usize);
            println!("  {c:8.4} | {emp_d:8.4} {mp_d:8.4} {bar}");
        }
        println!();
    }

    // ---------------------------------------------------------------
    // Part 2: MMD power loss as p/n -> 1
    // ---------------------------------------------------------------
    println!("--- Part 2: MMD power loss as p/n increases ---\n");

    // Two distributions: X ~ N(0, I_p), Y ~ N(delta, I_p)
    // As p grows relative to n, MMD loses power to detect the shift.
    let n_per_sample = 40;
    let delta = 0.3; // mean shift per dimension
    let n_trials = 10;

    println!(
        "  {:>6} {:>6} {:>10} {:>12}",
        "p", "p/n", "mean MMD", "detections"
    );
    println!("  {:-<6} {:-<6} {:-<10} {:-<12}", "", "", "", "");

    for p in [5, 10, 20, 30, 38] {
        let ratio = p as f64 / n_per_sample as f64;
        let mut mmd_sum = 0.0;
        let mut detections = 0;

        for trial in 0..n_trials {
            let trial_seed = seed + trial;
            let x = lcg_gaussian_data(n_per_sample, p, trial_seed);
            let y = lcg_shifted_data(n_per_sample, p, delta, trial_seed + 1000);

            let mmd = mmd_unbiased(&x, &y, |a, b| rbf(a, b, bandwidth));
            mmd_sum += mmd;
            if mmd > 0.01 {
                detections += 1;
            }
        }

        let mean_mmd = mmd_sum / n_trials as f64;
        println!("  {p:6} {ratio:6.2} {mean_mmd:10.4} {detections:12}/{n_trials}");
    }

    println!();
    println!("  As p/n approaches 1, the Gram matrix spectrum concentrates");
    println!("  (eigenvalues crowd into the MP bulk), leaving less spectral");
    println!("  room for the signal. MMD, which depends on kernel eigenvalue");
    println!("  spread, loses power to distinguish the distributions.");

    // ---------------------------------------------------------------
    // Part 3: Spectral concentration visualization
    // ---------------------------------------------------------------
    println!("\n--- Part 3: Spectral concentration at p/n extremes ---\n");

    let n = 50;
    for p in [5, 45] {
        let gamma = p as f64 / n as f64;
        let data = lcg_gaussian_data(n, p, seed);
        let gram = kernel_matrix(&data, |a, b| rbf(a, b, bandwidth));
        let eigs = eigenvalues_symmetric(&gram);

        let min_eig = eigs[0];
        let max_eig = eigs[n - 1];
        let spread = max_eig - min_eig;
        let (mp_lo, mp_hi) = marchenko_pastur_support(gamma, 1.0);

        println!("  p/n = {p}/{n} = {gamma:.2}");
        println!("  Eigenvalue spread: {spread:.4} (range [{min_eig:.4}, {max_eig:.4}])");
        println!("  MP bulk width: {:.4}", mp_hi - mp_lo);
        println!();
    }
    println!("  Wider spread = more spectral room for signal detection.");
    println!("  At p/n ~ 1, spread collapses and MP bulk dominates.");
}

/// Generate n vectors of dimension p from pseudo-random standard Gaussians.
///
/// Uses an LCG + Box-Muller transform for reproducibility without
/// pulling in extra dependencies.
fn lcg_gaussian_data(n: usize, p: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut state = seed;
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        let mut point = Vec::with_capacity(p);
        let mut i = 0;
        while i < p {
            // Generate pairs via Box-Muller
            let u1 = lcg_uniform(&mut state);
            let u2 = lcg_uniform(&mut state);
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            point.push(r * theta.cos());
            i += 1;
            if i < p {
                point.push(r * theta.sin());
                i += 1;
            }
        }
        data.push(point);
    }
    data
}

/// Generate n vectors of dimension p from N(delta, 1) per coordinate.
fn lcg_shifted_data(n: usize, p: usize, delta: f64, seed: u64) -> Vec<Vec<f64>> {
    let mut data = lcg_gaussian_data(n, p, seed);
    for point in &mut data {
        for x in point.iter_mut() {
            *x += delta;
        }
    }
    data
}

/// LCG producing a uniform in (0, 1).
fn lcg_uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    // Use upper bits for better quality
    let val = (*state >> 11) as f64 / (1u64 << 53) as f64;
    // Clamp away from 0 for Box-Muller safety
    val.max(1e-15)
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
    eigs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    eigs
}
