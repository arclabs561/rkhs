//! Calibration of the MMD permutation test under the null hypothesis.
//!
//! When both samples come from the same distribution, a valid permutation
//! p-value is (super-)uniform, so a 5% test rejects about 5% of the time.
//! With 400 seeded replicates the rejection count has standard deviation
//! about 4.4 around 20, so the band [6, 36] is roughly +-3.5 sd.

use rand::{Rng, SeedableRng};
use rkhs::{mmd_permutation_test_seeded, rbf};

fn gaussian_sample(rng: &mut rand::rngs::StdRng, n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|_| {
            // Box-Muller; u1 in (0, 1] avoids ln(0).
            let u1: f64 = 1.0 - rng.random::<f64>();
            let u2: f64 = rng.random::<f64>();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            vec![z]
        })
        .collect()
}

#[test]
fn permutation_test_rejects_about_five_percent_under_the_null() {
    let mut data_rng = rand::rngs::StdRng::seed_from_u64(2012);
    let kernel = |a: &[f64], b: &[f64]| rbf(a, b, 1.0);
    let replicates = 400;
    let mut rejections = 0;
    for r in 0..replicates {
        let x = gaussian_sample(&mut data_rng, 12);
        let y = gaussian_sample(&mut data_rng, 12);
        let (_, p) = mmd_permutation_test_seeded(&x, &y, kernel, 99, r as u64);
        if p <= 0.05 {
            rejections += 1;
        }
    }
    assert!(
        (6..=36).contains(&rejections),
        "{rejections} of {replicates} null replicates rejected at 5%"
    );
}

#[test]
fn permutation_test_detects_a_mean_shift() {
    let mut data_rng = rand::rngs::StdRng::seed_from_u64(7);
    let x = gaussian_sample(&mut data_rng, 20);
    let y: Vec<Vec<f64>> = gaussian_sample(&mut data_rng, 20)
        .into_iter()
        .map(|v| vec![v[0] + 3.0])
        .collect();
    let (_, p) = mmd_permutation_test_seeded(&x, &y, |a, b| rbf(a, b, 1.0), 199, 1);
    assert!(p <= 0.01, "p = {p} for a 3-sd mean shift");
}

#[test]
fn seeded_permutation_test_is_reproducible() {
    let mut data_rng = rand::rngs::StdRng::seed_from_u64(3);
    let x = gaussian_sample(&mut data_rng, 10);
    let y = gaussian_sample(&mut data_rng, 10);
    let k = |a: &[f64], b: &[f64]| rbf(a, b, 1.0);
    assert_eq!(
        mmd_permutation_test_seeded(&x, &y, k, 49, 11),
        mmd_permutation_test_seeded(&x, &y, k, 49, 11)
    );
}
