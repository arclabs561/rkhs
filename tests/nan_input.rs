//! Quantile and sliced-Wasserstein paths sort their inputs. NaN samples must
//! give a NaN or finite result, not a panic from the sort comparator.

use rkhs::{qmmd, quantile_gram_matrix};

fn k(a: f64, b: f64) -> f64 {
    (-(a - b) * (a - b)).exp()
}

#[test]
fn qmmd_with_nan_samples_does_not_panic() {
    let p = [0.3, f64::NAN, 1.2, -0.5, f64::NAN, 2.0];
    let q = [1.0, 0.1, f64::NAN, 0.7];
    let _ = qmmd(&p, &q, k, 8);
}

#[test]
fn quantile_gram_matrix_with_nan_samples_does_not_panic() {
    let s = [0.3, f64::NAN, 1.2, -0.5, 2.0];
    let _ = quantile_gram_matrix(&s, 0.5, k);
}

#[test]
#[allow(deprecated)]
fn sliced_wasserstein_graph_kernel_with_nan_features_does_not_panic() {
    use rkhs::sliced_wasserstein_graph_kernel;
    let f1 = vec![vec![0.0, 1.0], vec![f64::NAN, 2.0], vec![1.0, 0.5]];
    let f2 = vec![vec![1.0, f64::NAN], vec![0.0, 0.0]];
    let _ = sliced_wasserstein_graph_kernel(&f1, &f2, 16, 1.0, 5);
}
