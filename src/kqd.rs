//! Kernel quantile discrepancies from caller-supplied Gaussian draws.
//!
//! This module implements the empirical Gaussian expected kernel quantile
//! discrepancy (e-KQD) of Naslidnyk, Chau, Briol, and Muandet. Randomness is
//! kept outside the core routine: callers supply the landmarks and Gaussian
//! coefficients, which makes a computation reproducible and directly testable.

use ndarray::{ArrayView1, ArrayView2};
use std::fmt;

/// Quantile masses used to aggregate paired order statistics.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum KqdWeights {
    /// Equal mass on every order statistic.
    #[default]
    Uniform,
    /// Non-negative masses, normalized internally to sum to one.
    Discrete(Vec<f64>),
}

/// Configuration for [`gaussian_ekqd_from_draws`].
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianEkqdConfig {
    /// Wasserstein power. Must be finite and at least one.
    pub power: f64,
    /// Normalize every sampled RKHS direction to unit norm.
    pub normalize: bool,
    /// Mass assigned to the paired empirical order statistics.
    pub weights: KqdWeights,
}

impl Default for GaussianEkqdConfig {
    fn default() -> Self {
        Self {
            power: 2.0,
            normalize: true,
            weights: KqdWeights::Uniform,
        }
    }
}

/// Errors returned by empirical KQD evaluation.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum KqdError {
    /// One of the sample, landmark, or projection dimensions is empty.
    EmptyInput(&'static str),
    /// The two samples do not have the same shape.
    SampleShapeMismatch {
        /// Shape of the first sample.
        x: (usize, usize),
        /// Shape of the second sample.
        y: (usize, usize),
    },
    /// The feature dimension does not match the landmark dimension.
    LandmarkDimensionMismatch {
        /// Sample feature dimension.
        samples: usize,
        /// Landmark feature dimension.
        landmarks: usize,
    },
    /// The coefficient width does not match the number of landmarks.
    CoefficientShapeMismatch {
        /// Number of landmarks.
        landmarks: usize,
        /// Coefficients per projection.
        coefficients: usize,
    },
    /// A numeric input contains a NaN or infinity.
    NonFiniteInput(&'static str),
    /// The requested Wasserstein power is not finite and at least one.
    InvalidPower(f64),
    /// Explicit weights do not match the common sample size.
    WeightCountMismatch {
        /// Common sample size.
        samples: usize,
        /// Number of weights supplied.
        weights: usize,
    },
    /// An explicit weight is negative or non-finite.
    InvalidWeight {
        /// Index of the invalid weight.
        index: usize,
        /// Invalid value.
        value: f64,
    },
    /// Explicit weights do not have a finite, positive sum.
    InvalidWeightSum(f64),
    /// A kernel evaluation returned a NaN or infinity.
    NonFiniteKernelValue,
    /// A sampled direction cannot be normalized to unit RKHS norm.
    DegenerateDirection {
        /// Row index in the coefficient matrix.
        projection: usize,
        /// Computed squared RKHS norm.
        norm_squared: f64,
    },
    /// The final aggregation produced a NaN or infinity.
    NonFiniteResult,
}

impl fmt::Display for KqdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput(name) => write!(f, "{name} must be non-empty"),
            Self::SampleShapeMismatch { x, y } => {
                write!(f, "sample shapes must match, got {x:?} and {y:?}")
            }
            Self::LandmarkDimensionMismatch { samples, landmarks } => write!(
                f,
                "sample and landmark dimensions must match, got {samples} and {landmarks}"
            ),
            Self::CoefficientShapeMismatch {
                landmarks,
                coefficients,
            } => write!(
                f,
                "each projection needs {landmarks} coefficients, got {coefficients}"
            ),
            Self::NonFiniteInput(name) => write!(f, "{name} must contain only finite values"),
            Self::InvalidPower(power) => {
                write!(f, "power must be finite and at least 1, got {power}")
            }
            Self::WeightCountMismatch { samples, weights } => {
                write!(f, "expected {samples} weights, got {weights}")
            }
            Self::InvalidWeight { index, value } => {
                write!(
                    f,
                    "weight {index} must be finite and non-negative, got {value}"
                )
            }
            Self::InvalidWeightSum(sum) => {
                write!(f, "weights must have a finite, positive sum, got {sum}")
            }
            Self::NonFiniteKernelValue => write!(f, "kernel returned a non-finite value"),
            Self::DegenerateDirection {
                projection,
                norm_squared,
            } => write!(
                f,
                "projection {projection} has invalid squared RKHS norm {norm_squared}"
            ),
            Self::NonFiniteResult => write!(f, "KQD aggregation produced a non-finite result"),
        }
    }
}

impl std::error::Error for KqdError {}

/// Estimate Gaussian expected KQD from fixed landmarks and Gaussian draws.
///
/// `x` and `y` must have the same shape `(n, d)`. `landmarks` has shape
/// `(m, d)`, and each row of `coefficients` is one direction with `m`
/// coefficients. For coefficient row `lambda`, the projected value at `x_j`
/// is
///
/// `sum_r lambda_r k(z_r, x_j) / sqrt(m)`.
///
/// The projected values for the two samples are sorted separately and paired
/// by rank. Their weighted `power`-distance is averaged over coefficient rows,
/// followed by the `1 / power` root. With `normalize = true`, each direction
/// is divided by its RKHS norm computed from the landmark Gram matrix.
pub fn gaussian_ekqd_from_draws<F>(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    landmarks: ArrayView2<'_, f64>,
    coefficients: ArrayView2<'_, f64>,
    kernel: F,
    config: &GaussianEkqdConfig,
) -> Result<f64, KqdError>
where
    F: Fn(ArrayView1<'_, f64>, ArrayView1<'_, f64>) -> f64,
{
    validate_shapes(x, y, landmarks, coefficients)?;
    validate_finite(x, "x")?;
    validate_finite(y, "y")?;
    validate_finite(landmarks, "landmarks")?;
    validate_finite(coefficients, "coefficients")?;
    if !config.power.is_finite() || config.power < 1.0 {
        return Err(KqdError::InvalidPower(config.power));
    }
    let weights = normalized_weights(&config.weights, x.nrows())?;
    let landmark_count = landmarks.nrows();
    let landmark_scale = (landmark_count as f64).sqrt();

    let mut kzz = None;
    if config.normalize {
        let mut gram = vec![0.0; landmark_count * landmark_count];
        for r in 0..landmark_count {
            for s in 0..landmark_count {
                let value = kernel(landmarks.row(r), landmarks.row(s));
                if !value.is_finite() {
                    return Err(KqdError::NonFiniteKernelValue);
                }
                gram[r * landmark_count + s] = value;
            }
        }
        kzz = Some(gram);
    }

    let mut total = 0.0;
    for (projection, lambda) in coefficients.rows().into_iter().enumerate() {
        let denominator = if let Some(gram) = &kzz {
            let mut norm_squared = 0.0;
            for r in 0..landmark_count {
                for s in 0..landmark_count {
                    norm_squared += lambda[r] * gram[r * landmark_count + s] * lambda[s];
                }
            }
            norm_squared /= landmark_count as f64;
            if !norm_squared.is_finite() || norm_squared <= 0.0 {
                return Err(KqdError::DegenerateDirection {
                    projection,
                    norm_squared,
                });
            }
            landmark_scale * norm_squared.sqrt()
        } else {
            landmark_scale
        };

        let mut projected_x = project(x, landmarks, lambda, denominator, &kernel)?;
        let mut projected_y = project(y, landmarks, lambda, denominator, &kernel)?;
        projected_x.sort_unstable_by(f64::total_cmp);
        projected_y.sort_unstable_by(f64::total_cmp);

        total += weights
            .iter()
            .zip(projected_x.iter().zip(&projected_y))
            .map(|(weight, (left, right))| weight * (left - right).abs().powf(config.power))
            .sum::<f64>();
    }

    let result = (total / coefficients.nrows() as f64).powf(1.0 / config.power);
    if result.is_finite() {
        Ok(result)
    } else {
        Err(KqdError::NonFiniteResult)
    }
}

fn validate_shapes(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    landmarks: ArrayView2<'_, f64>,
    coefficients: ArrayView2<'_, f64>,
) -> Result<(), KqdError> {
    for (name, rows, columns) in [
        ("x", x.nrows(), x.ncols()),
        ("y", y.nrows(), y.ncols()),
        ("landmarks", landmarks.nrows(), landmarks.ncols()),
        ("coefficients", coefficients.nrows(), coefficients.ncols()),
    ] {
        if rows == 0 || columns == 0 {
            return Err(KqdError::EmptyInput(name));
        }
    }
    if x.dim() != y.dim() {
        return Err(KqdError::SampleShapeMismatch {
            x: x.dim(),
            y: y.dim(),
        });
    }
    if x.ncols() != landmarks.ncols() {
        return Err(KqdError::LandmarkDimensionMismatch {
            samples: x.ncols(),
            landmarks: landmarks.ncols(),
        });
    }
    if landmarks.nrows() != coefficients.ncols() {
        return Err(KqdError::CoefficientShapeMismatch {
            landmarks: landmarks.nrows(),
            coefficients: coefficients.ncols(),
        });
    }
    Ok(())
}

fn validate_finite(values: ArrayView2<'_, f64>, name: &'static str) -> Result<(), KqdError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(KqdError::NonFiniteInput(name))
    }
}

fn normalized_weights(weights: &KqdWeights, samples: usize) -> Result<Vec<f64>, KqdError> {
    match weights {
        KqdWeights::Uniform => Ok(vec![1.0 / samples as f64; samples]),
        KqdWeights::Discrete(weights) => {
            if weights.len() != samples {
                return Err(KqdError::WeightCountMismatch {
                    samples,
                    weights: weights.len(),
                });
            }
            for (index, &value) in weights.iter().enumerate() {
                if !value.is_finite() || value < 0.0 {
                    return Err(KqdError::InvalidWeight { index, value });
                }
            }
            let sum: f64 = weights.iter().sum();
            if !sum.is_finite() || sum <= 0.0 {
                return Err(KqdError::InvalidWeightSum(sum));
            }
            Ok(weights.iter().map(|weight| weight / sum).collect())
        }
    }
}

fn project<F>(
    samples: ArrayView2<'_, f64>,
    landmarks: ArrayView2<'_, f64>,
    coefficients: ArrayView1<'_, f64>,
    denominator: f64,
    kernel: &F,
) -> Result<Vec<f64>, KqdError>
where
    F: Fn(ArrayView1<'_, f64>, ArrayView1<'_, f64>) -> f64,
{
    samples
        .rows()
        .into_iter()
        .map(|sample| {
            let mut value = 0.0;
            for (coefficient, landmark) in coefficients.iter().zip(landmarks.rows()) {
                let kernel_value = kernel(landmark, sample);
                if !kernel_value.is_finite() {
                    return Err(KqdError::NonFiniteKernelValue);
                }
                value += coefficient * kernel_value;
            }
            let value = value / denominator;
            if value.is_finite() {
                Ok(value)
            } else {
                Err(KqdError::NonFiniteResult)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, ArrayView1};

    fn linear(left: ArrayView1<'_, f64>, right: ArrayView1<'_, f64>) -> f64 {
        left.iter().zip(right).map(|(a, b)| a * b).sum()
    }

    fn evaluate(
        x: ArrayView2<'_, f64>,
        y: ArrayView2<'_, f64>,
        landmarks: ArrayView2<'_, f64>,
        coefficients: ArrayView2<'_, f64>,
        config: &GaussianEkqdConfig,
    ) -> Result<f64, KqdError> {
        gaussian_ekqd_from_draws(x, y, landmarks, coefficients, linear, config)
    }

    #[test]
    fn hand_oracle_includes_landmark_scaling_and_direction_norm() {
        let x = array![[1.0], [3.0]];
        let y = array![[2.0], [7.0]];
        let landmarks = array![[1.0], [3.0]];
        let coefficients = array![[1.0, 1.0]];

        let normalized = evaluate(
            x.view(),
            y.view(),
            landmarks.view(),
            coefficients.view(),
            &GaussianEkqdConfig::default(),
        )
        .unwrap();
        // The normalized linear direction is x itself, so W_2 is
        // sqrt((|1-2|^2 + |3-7|^2) / 2) = sqrt(8.5).
        assert!((normalized - 8.5_f64.sqrt()).abs() < 1e-12);

        let unnormalized = evaluate(
            x.view(),
            y.view(),
            landmarks.view(),
            coefficients.view(),
            &GaussianEkqdConfig {
                normalize: false,
                ..GaussianEkqdConfig::default()
            },
        )
        .unwrap();
        assert!((unnormalized - normalized * 8.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn matches_projected_one_dimensional_wasserstein_distance() {
        let x = array![[0.0, 3.0], [2.0, -1.0], [5.0, 8.0]];
        let y = array![[1.0, 9.0], [4.0, 2.0], [8.0, -4.0]];
        let landmarks = array![[1.0, 0.0], [0.0, 1.0]];
        let coefficients = array![[2.0, 0.0]];
        let actual = evaluate(
            x.view(),
            y.view(),
            landmarks.view(),
            coefficients.view(),
            &GaussianEkqdConfig::default(),
        )
        .unwrap();
        let expected = ((1.0_f64 + 4.0 + 9.0) / 3.0).sqrt();
        assert!((actual - expected).abs() < 1e-12);
    }

    #[test]
    fn identity_symmetry_permutation_and_positive_scale_invariance() {
        let x = array![[3.0], [-1.0], [2.0]];
        let permuted = array![[2.0], [3.0], [-1.0]];
        let y = array![[0.0], [4.0], [8.0]];
        let landmarks = array![[1.0], [2.0]];
        let coefficients = array![[0.5, -2.0], [1.0, 3.0]];
        let scaled = &coefficients * 7.0;
        let config = GaussianEkqdConfig::default();

        assert_eq!(
            evaluate(
                x.view(),
                permuted.view(),
                landmarks.view(),
                coefficients.view(),
                &config
            ),
            Ok(0.0)
        );
        let xy = evaluate(
            x.view(),
            y.view(),
            landmarks.view(),
            coefficients.view(),
            &config,
        )
        .unwrap();
        let yx = evaluate(
            y.view(),
            x.view(),
            landmarks.view(),
            coefficients.view(),
            &config,
        )
        .unwrap();
        let scaled_value =
            evaluate(x.view(), y.view(), landmarks.view(), scaled.view(), &config).unwrap();
        assert!((xy - yx).abs() < 1e-12);
        assert!((xy - scaled_value).abs() < 1e-12);
    }

    #[test]
    fn explicit_equal_weights_match_uniform_weights() {
        let x = array![[0.0], [2.0]];
        let y = array![[1.0], [5.0]];
        let landmarks = array![[1.0]];
        let coefficients = array![[1.0]];
        let uniform = evaluate(
            x.view(),
            y.view(),
            landmarks.view(),
            coefficients.view(),
            &GaussianEkqdConfig::default(),
        )
        .unwrap();
        let explicit = evaluate(
            x.view(),
            y.view(),
            landmarks.view(),
            coefficients.view(),
            &GaussianEkqdConfig {
                weights: KqdWeights::Discrete(vec![2.0, 2.0]),
                ..GaussianEkqdConfig::default()
            },
        )
        .unwrap();
        assert_eq!(uniform, explicit);
    }

    #[test]
    fn rejects_invalid_contracts() {
        let x = array![[0.0], [1.0]];
        let y = array![[0.0], [1.0]];
        let landmarks = array![[1.0]];
        let coefficients = array![[1.0]];

        let invalid_power = GaussianEkqdConfig {
            power: 0.5,
            ..GaussianEkqdConfig::default()
        };
        assert_eq!(
            evaluate(
                x.view(),
                y.view(),
                landmarks.view(),
                coefficients.view(),
                &invalid_power
            ),
            Err(KqdError::InvalidPower(0.5))
        );

        let invalid_weights = GaussianEkqdConfig {
            weights: KqdWeights::Discrete(vec![1.0, -1.0]),
            ..GaussianEkqdConfig::default()
        };
        assert!(matches!(
            evaluate(
                x.view(),
                y.view(),
                landmarks.view(),
                coefficients.view(),
                &invalid_weights
            ),
            Err(KqdError::InvalidWeight { index: 1, .. })
        ));

        let zero_coefficients = array![[0.0]];
        assert!(matches!(
            evaluate(
                x.view(),
                y.view(),
                landmarks.view(),
                zero_coefficients.view(),
                &GaussianEkqdConfig::default()
            ),
            Err(KqdError::DegenerateDirection { projection: 0, .. })
        ));

        let bad_y = array![[0.0, 1.0], [1.0, 2.0]];
        assert!(matches!(
            evaluate(
                x.view(),
                bad_y.view(),
                landmarks.view(),
                coefficients.view(),
                &GaussianEkqdConfig::default()
            ),
            Err(KqdError::SampleShapeMismatch { .. })
        ));

        let nan_x = array![[f64::NAN], [1.0]];
        assert_eq!(
            evaluate(
                nan_x.view(),
                y.view(),
                landmarks.view(),
                coefficients.view(),
                &GaussianEkqdConfig::default()
            ),
            Err(KqdError::NonFiniteInput("x"))
        );

        assert_eq!(
            gaussian_ekqd_from_draws(
                x.view(),
                y.view(),
                landmarks.view(),
                coefficients.view(),
                |_, _| f64::NAN,
                &GaussianEkqdConfig::default()
            ),
            Err(KqdError::NonFiniteKernelValue)
        );
    }
}
