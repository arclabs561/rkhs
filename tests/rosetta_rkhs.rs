//! Rosetta correctness fixtures: rkhs kernels, Gram matrices, and MMD asserted
//! against scikit-learn.
//!
//! Reference values in `fixtures/rosetta/rkhs_kernels.json` come from
//! `gen_rkhs.py` (their provenance). Gram matrices are exact sklearn oracles
//! (rbf/polynomial/linear/laplacian via sklearn.metrics.pairwise, Matern via
//! sklearn.gaussian_process.kernels). MMD has no sklearn function, so its
//! reference is computed in numpy from the sklearn RBF kernel matrices using the
//! canonical estimator; that is a cross-implementation check, not a library
//! oracle. The KDE-family kernels (epanechnikov, triangle, ...) are excluded
//! because no external library implements them.
//!
//! Regenerate the fixture: `uv run tests/fixtures/rosetta/gen_rkhs.py`.

use ndarray::Array2;
use serde::Deserialize;

const FIXTURE: &str = include_str!("fixtures/rosetta/rkhs_kernels.json");

#[derive(Deserialize)]
struct Fixture {
    params: Params,
    x: Vec<Vec<f64>>,
    y: Vec<Vec<f64>>,
    expected: Expected,
}

#[derive(Deserialize)]
struct Params {
    sigma: f64,
    poly_degree: u32,
    poly_gamma: f64,
    poly_coef0: f64,
    laplacian_sigma: f64,
    matern_lengthscale: f64,
}

#[derive(Deserialize)]
struct Expected {
    gram_rbf: Vec<Vec<f64>>,
    gram_poly: Vec<Vec<f64>>,
    gram_linear: Vec<Vec<f64>>,
    gram_laplacian: Vec<Vec<f64>>,
    gram_matern12: Vec<Vec<f64>>,
    gram_matern32: Vec<Vec<f64>>,
    gram_matern52: Vec<Vec<f64>>,
    mmd_biased_rbf: f64,
    mmd_unbiased_rbf: f64,
}

fn close(got: f64, want: f64, label: &str) {
    let tol = 1e-9 * (1.0 + want.abs());
    let diff = (got - want).abs();
    assert!(
        diff <= tol,
        "{label}: rkhs={got} sklearn={want} diff={diff} tol={tol}"
    );
}

fn mat_close(got: &Array2<f64>, want: &[Vec<f64>], label: &str) {
    assert_eq!(got.nrows(), want.len(), "{label}: row count");
    for (i, row) in want.iter().enumerate() {
        assert_eq!(got.ncols(), row.len(), "{label}: col count");
        for (j, &w) in row.iter().enumerate() {
            close(got[[i, j]], w, &format!("{label}[{i}][{j}]"));
        }
    }
}

fn to_array2(rows: &[Vec<f64>]) -> Array2<f64> {
    let d = rows[0].len();
    let mut a = Array2::zeros((rows.len(), d));
    for (i, r) in rows.iter().enumerate() {
        for (j, &v) in r.iter().enumerate() {
            a[[i, j]] = v;
        }
    }
    a
}

#[test]
fn rosetta_kernels_match_sklearn() {
    let fx: Fixture = serde_json::from_str(FIXTURE).expect("parse rosetta fixture");
    let p = &fx.params;
    let x = &fx.x;
    let e = &fx.expected;

    // RBF Gram via both rkhs paths (scalar-closure and the ndarray expansion).
    mat_close(
        &rkhs::kernel_matrix(x, |a, b| rkhs::rbf(a, b, p.sigma)),
        &e.gram_rbf,
        "gram_rbf_closure",
    );
    mat_close(
        &rkhs::rbf_kernel_matrix_ndarray(to_array2(x).view(), p.sigma),
        &e.gram_rbf,
        "gram_rbf_ndarray",
    );

    mat_close(
        &rkhs::kernel_matrix(x, |a, b| {
            rkhs::polynomial(a, b, p.poly_degree, p.poly_gamma, p.poly_coef0)
        }),
        &e.gram_poly,
        "gram_poly",
    );
    mat_close(
        &rkhs::kernel_matrix(x, rkhs::linear),
        &e.gram_linear,
        "gram_linear",
    );
    mat_close(
        &rkhs::kernel_matrix(x, |a, b| rkhs::laplacian(a, b, p.laplacian_sigma)),
        &e.gram_laplacian,
        "gram_laplacian",
    );
    mat_close(
        &rkhs::kernel_matrix(x, |a, b| rkhs::matern_12(a, b, p.matern_lengthscale)),
        &e.gram_matern12,
        "gram_matern12",
    );
    mat_close(
        &rkhs::kernel_matrix(x, |a, b| rkhs::matern_32(a, b, p.matern_lengthscale)),
        &e.gram_matern32,
        "gram_matern32",
    );
    mat_close(
        &rkhs::kernel_matrix(x, |a, b| rkhs::matern_52(a, b, p.matern_lengthscale)),
        &e.gram_matern52,
        "gram_matern52",
    );

    // MMD (cross-implementation check vs the numpy estimator over sklearn kernels).
    let rbf = |a: &[f64], b: &[f64]| rkhs::rbf(a, b, p.sigma);
    close(
        rkhs::mmd_biased(x, &fx.y, rbf),
        e.mmd_biased_rbf,
        "mmd_biased_rbf",
    );
    close(
        rkhs::mmd_unbiased(x, &fx.y, rbf),
        e.mmd_unbiased_rbf,
        "mmd_unbiased_rbf",
    );
}
