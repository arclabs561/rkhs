//! Nonlinear KQD oracle generated from the authors' pinned implementation.
//!
//! Regenerate with `uv run scripts/generate_kqd_reference.py`. Normal test
//! runs consume only the JSON fixture and do not require Python or JAX.

use ndarray::{Array2, ArrayView1};
use rkhs::kqd::{gaussian_ekqd_from_draws, GaussianEkqdConfig, KqdWeights};
use serde::Deserialize;

const FIXTURE: &str = include_str!("fixtures/kqd_reference.json");
const UPSTREAM_COMMIT: &str = "34ecaf75090f0482ab3fc6603d008d5ef3909b11";

#[derive(Deserialize)]
struct Fixture {
    provenance: Provenance,
    params: Params,
    x: Vec<Vec<f64>>,
    y: Vec<Vec<f64>>,
    landmarks: Vec<Vec<f64>>,
    coefficients: Vec<Vec<f64>>,
    weights: Vec<f64>,
    expected: Expected,
}

#[derive(Deserialize)]
struct Provenance {
    upstream_commit: String,
}

#[derive(Deserialize)]
struct Params {
    bandwidth: f64,
    power: f64,
    normalize: bool,
}

#[derive(Deserialize)]
struct Expected {
    landmark_gram: Vec<Vec<f64>>,
    directions: Vec<Direction>,
    distance: f64,
}

#[derive(Deserialize)]
struct Direction {
    norm_squared: f64,
    sorted_x: Vec<f64>,
    sorted_y: Vec<f64>,
    tau_power: f64,
}

fn array(rows: &[Vec<f64>]) -> Array2<f64> {
    let columns = rows.first().expect("non-empty fixture").len();
    Array2::from_shape_vec(
        (rows.len(), columns),
        rows.iter().flatten().copied().collect(),
    )
    .expect("rectangular fixture")
}

fn close(actual: f64, expected: f64, label: &str) {
    let tolerance = 2e-12 * (1.0 + expected.abs());
    assert!(
        (actual - expected).abs() <= tolerance,
        "{label}: rkhs={actual:.17} upstream={expected:.17}"
    );
}

#[test]
fn nonlinear_kqd_matches_pinned_reference_directions_and_distance() {
    let fixture: Fixture = serde_json::from_str(FIXTURE).expect("parse KQD fixture");
    assert_eq!(fixture.provenance.upstream_commit, UPSTREAM_COMMIT);

    let x = array(&fixture.x);
    let y = array(&fixture.y);
    let landmarks = array(&fixture.landmarks);
    let coefficients = array(&fixture.coefficients);
    let bandwidth = fixture.params.bandwidth;
    let kernel = |left: ArrayView1<'_, f64>, right: ArrayView1<'_, f64>| {
        let squared_distance: f64 = left.iter().zip(right).map(|(a, b)| (a - b).powi(2)).sum();
        (-squared_distance / (2.0 * bandwidth.powi(2))).exp()
    };

    let mut gram = Array2::zeros((landmarks.nrows(), landmarks.nrows()));
    for row in 0..landmarks.nrows() {
        for column in 0..landmarks.nrows() {
            gram[[row, column]] = kernel(landmarks.row(row), landmarks.row(column));
            close(
                gram[[row, column]],
                fixture.expected.landmark_gram[row][column],
                &format!("landmark Gram [{row}, {column}]"),
            );
        }
    }

    for (index, lambda) in coefficients.rows().into_iter().enumerate() {
        let expected = &fixture.expected.directions[index];
        let norm_squared = lambda.dot(&gram.dot(&lambda));
        close(
            norm_squared,
            expected.norm_squared,
            "direction norm squared",
        );

        let project = |samples: &Array2<f64>| {
            let mut values: Vec<_> = samples
                .rows()
                .into_iter()
                .map(|sample| {
                    lambda
                        .iter()
                        .zip(landmarks.rows())
                        .map(|(coefficient, landmark)| coefficient * kernel(landmark, sample))
                        .sum::<f64>()
                        / norm_squared.sqrt()
                })
                .collect();
            values.sort_unstable_by(f64::total_cmp);
            values
        };
        let projected_x = project(&x);
        let projected_y = project(&y);
        for (sample, (&actual, &upstream)) in projected_x.iter().zip(&expected.sorted_x).enumerate()
        {
            close(actual, upstream, &format!("direction {index} x[{sample}]"));
        }
        for (sample, (&actual, &upstream)) in projected_y.iter().zip(&expected.sorted_y).enumerate()
        {
            close(actual, upstream, &format!("direction {index} y[{sample}]"));
        }
        let tau_power: f64 = fixture
            .weights
            .iter()
            .zip(projected_x.iter().zip(&projected_y))
            .map(|(weight, (left, right))| weight * (left - right).abs().powf(fixture.params.power))
            .sum();
        close(
            tau_power,
            expected.tau_power,
            &format!("direction {index} tau"),
        );
    }

    let actual = gaussian_ekqd_from_draws(
        x.view(),
        y.view(),
        landmarks.view(),
        coefficients.view(),
        kernel,
        &GaussianEkqdConfig {
            power: fixture.params.power,
            normalize: fixture.params.normalize,
            weights: KqdWeights::Discrete(fixture.weights),
        },
    )
    .expect("evaluate fixed nonlinear KQD case");
    close(actual, fixture.expected.distance, "final KQD distance");
}
