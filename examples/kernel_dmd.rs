//! Kernel DMD proof sketch over RBF features.
//!
//! Run: cargo run --example kernel_dmd
//!
//! This is a `koopdmd` proof sketch, not public API. It lifts scalar dynamics
//! into RBF features, fits a linear one-step operator by ridge regression, and
//! compares feature-space prediction error against an identity baseline.

use rkhs::rbf;

const SIGMA: f64 = 0.35;
const RIDGE: f64 = 1e-6;

fn dynamics(x: f64) -> f64 {
    0.72 * x + 0.22 * (2.4 * x).sin()
}

fn feature(x: f64, centers: &[f64]) -> Vec<f64> {
    centers.iter().map(|&c| rbf(&[x], &[c], SIGMA)).collect()
}

fn l2(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn fit_operator(x_features: &[Vec<f64>], y_features: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = x_features.len();
    let d = x_features[0].len();
    let mut normal = vec![vec![0.0; d]; d];
    let mut rhs = vec![vec![0.0; d]; d];

    for sample in 0..n {
        for i in 0..d {
            for j in 0..d {
                normal[i][j] += x_features[sample][i] * x_features[sample][j];
                rhs[i][j] += x_features[sample][i] * y_features[sample][j];
            }
        }
    }

    for (i, row) in normal.iter_mut().enumerate() {
        row[i] += RIDGE;
    }

    solve(normal, rhs)
}

fn apply_operator(x: &[f64], operator: &[Vec<f64>]) -> Vec<f64> {
    let d = x.len();
    let mut out = vec![0.0; d];
    for i in 0..d {
        for (j, out_j) in out.iter_mut().enumerate() {
            *out_j += x[i] * operator[i][j];
        }
    }
    out
}

fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = a.len();
    for pivot in 0..n {
        let best = (pivot..n)
            .max_by(|&i, &j| a[i][pivot].abs().total_cmp(&a[j][pivot].abs()))
            .unwrap();
        a.swap(pivot, best);
        b.swap(pivot, best);

        let denom = a[pivot][pivot];
        assert!(denom.abs() > 1e-12, "singular ridge system");
        for value in a[pivot].iter_mut().skip(pivot) {
            *value /= denom;
        }
        for value in &mut b[pivot] {
            *value /= denom;
        }

        let pivot_a = a[pivot].clone();
        let pivot_b = b[pivot].clone();
        for row in 0..n {
            if row == pivot {
                continue;
            }
            let factor = a[row][pivot];
            for (col, value) in a[row].iter_mut().enumerate().skip(pivot) {
                *value -= factor * pivot_a[col];
            }
            for (value, pivot_value) in b[row].iter_mut().zip(&pivot_b) {
                *value -= factor * pivot_value;
            }
        }
    }
    b
}

fn main() {
    let centers = [-1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2];
    let train_x: Vec<f64> = (0..41).map(|i| -1.0 + i as f64 * 0.05).collect();
    let train_y: Vec<f64> = train_x.iter().map(|&x| dynamics(x)).collect();

    let x_features: Vec<_> = train_x.iter().map(|&x| feature(x, &centers)).collect();
    let y_features: Vec<_> = train_y.iter().map(|&y| feature(y, &centers)).collect();
    let operator = fit_operator(&x_features, &y_features);

    let test_x: Vec<f64> = [-0.93, -0.37, 0.18, 0.71].into();
    let mut identity_error = 0.0;
    let mut dmd_error = 0.0;
    for x in test_x {
        let phi_x = feature(x, &centers);
        let phi_y = feature(dynamics(x), &centers);
        let predicted = apply_operator(&phi_x, &operator);
        identity_error += l2(&phi_x, &phi_y);
        dmd_error += l2(&predicted, &phi_y);
    }
    identity_error /= 4.0;
    dmd_error /= 4.0;

    println!("RBF centers: {centers:?}");
    println!("mean one-step feature error, identity: {identity_error:.6}");
    println!("mean one-step feature error, kernel DMD: {dmd_error:.6}");

    assert!(dmd_error < identity_error * 0.35);
}
