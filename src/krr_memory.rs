//! Kernel Ridge Regression associative memory (Hopfield network), per
//! "Kernel Ridge Regression for Efficient Learning of High-Capacity Hopfield
//! Networks" (arXiv:2504.12561) and its KLR sibling (arXiv:2504.07633).
//!
//! Classical Hebbian associative memory stores only about `0.14 N` patterns. A
//! kernel-trained Hopfield network lifts this far higher: it learns dual
//! coefficients by ridge regression in an RBF feature space and reliably stores
//! and recalls well over `N` patterns (the paper reports `P/N` up to ~1.5).
//!
//! Training is a single closed-form solve. With stored bipolar patterns
//! `X` (`P x N`, entries in `{-1, +1}`) and the `P x P` Gram matrix
//! `K[i,j] = k(xi, xj)`:
//!
//! ```text
//! (K + lambda I) alpha = X        (paper Eq. 1)
//! ```
//!
//! `K + lambda I` is symmetric positive-definite, so the solve is a Cholesky
//! factorization plus forward/back substitution (no linear-algebra backend
//! required). Retrieval iterates the synchronous update
//! `s <- sign(k_s . alpha)`, where `k_s = [k(s, x1), ..., k(s, xP)]`.

use crate::rbf;
use ndarray::{Array1, Array2};

/// A Kernel Ridge Regression Hopfield associative memory over an RBF kernel.
#[derive(Debug, Clone)]
pub struct KrrMemory {
    patterns: Vec<Vec<f64>>,
    alpha: Array2<f64>, // P x N learned dual coefficients
    sigma: f64,
}

impl KrrMemory {
    /// Train on bipolar patterns (each entry ideally in `{-1, +1}`), an RBF
    /// bandwidth `sigma`, and ridge parameter `lambda`. Solves
    /// `(K + lambda I) alpha = X` by Cholesky. Returns `None` if the patterns are
    /// empty, ragged, or the regularized Gram matrix is not positive-definite.
    ///
    /// To match the paper's `K(x, y) = exp(-||x - y||^2 / N)`, pass
    /// `sigma = (N / 2).sqrt()` (since `rbf` uses `exp(-||x-y||^2 / (2 sigma^2))`).
    pub fn train(patterns: &[Vec<f64>], sigma: f64, lambda: f64) -> Option<Self> {
        let p = patterns.len();
        if p == 0 || sigma <= 0.0 {
            return None;
        }
        let n = patterns[0].len();
        if n == 0 || patterns.iter().any(|x| x.len() != n) {
            return None;
        }

        // Gram matrix K (P x P) plus ridge: A = K + lambda I.
        let mut a = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            for j in 0..p {
                let mut v = rbf(&patterns[i], &patterns[j], sigma);
                if i == j {
                    v += lambda;
                }
                a[[i, j]] = v;
            }
        }

        // Targets X (P x N): the stored patterns themselves.
        let mut x = Array2::<f64>::zeros((p, n));
        for (i, pat) in patterns.iter().enumerate() {
            for (j, &v) in pat.iter().enumerate() {
                x[[i, j]] = v;
            }
        }

        let alpha = cholesky_solve(&a, &x)?;
        Some(Self {
            patterns: patterns.to_vec(),
            alpha,
            sigma,
        })
    }

    /// Number of stored patterns.
    pub fn len(&self) -> usize {
        self.patterns.len()
    }

    /// Whether the memory is empty.
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    /// One synchronous update: `sign(k_s . alpha)`.
    pub fn step(&self, state: &[f64]) -> Vec<f64> {
        let p = self.patterns.len();
        let mut k = Array1::<f64>::zeros(p);
        for (mu, pat) in self.patterns.iter().enumerate() {
            k[mu] = rbf(state, pat, self.sigma);
        }
        let h = k.dot(&self.alpha); // length N
        h.iter()
            .map(|&v| if v >= 0.0 { 1.0 } else { -1.0 })
            .collect()
    }

    /// Retrieve by iterating the synchronous update until a fixed point or
    /// `max_iters` is reached.
    pub fn retrieve(&self, query: &[f64], max_iters: usize) -> Vec<f64> {
        let mut s = query.to_vec();
        for _ in 0..max_iters {
            let next = self.step(&s);
            if next == s {
                break;
            }
            s = next;
        }
        s
    }
}

/// Solve `A X = B` for `X`, where `A` is symmetric positive-definite, via
/// Cholesky factorization (`A = L L^T`) plus forward/back substitution. Returns
/// `None` if `A` is not positive-definite (a non-positive pivot appears).
fn cholesky_solve(a: &Array2<f64>, b: &Array2<f64>) -> Option<Array2<f64>> {
    let n = a.nrows();
    let cols = b.ncols();

    // Cholesky: A = L L^T, L lower-triangular.
    let mut l = Array2::<f64>::zeros((n, n));
    for j in 0..n {
        let mut diag = a[[j, j]];
        for k in 0..j {
            diag -= l[[j, k]] * l[[j, k]];
        }
        if diag <= 0.0 {
            return None;
        }
        let ljj = diag.sqrt();
        l[[j, j]] = ljj;
        for i in (j + 1)..n {
            let mut s = a[[i, j]];
            for k in 0..j {
                s -= l[[i, k]] * l[[j, k]];
            }
            l[[i, j]] = s / ljj;
        }
    }

    // Solve for each column of B: L y = b, then L^T x = y.
    let mut x = Array2::<f64>::zeros((n, cols));
    for c in 0..cols {
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            let mut s = b[[i, c]];
            for k in 0..i {
                s -= l[[i, k]] * y[k];
            }
            y[i] = s / l[[i, i]];
        }
        for i in (0..n).rev() {
            let mut s = y[i];
            for k in (i + 1)..n {
                s -= l[[k, i]] * x[[k, c]];
            }
            x[[i, c]] = s / l[[i, i]];
        }
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*seed >> 40) as f64 / (1u64 << 24) as f64
    }

    #[test]
    fn cholesky_solve_matches_known_system() {
        // SPD A = [[4,1],[1,3]], B = [[1],[2]]. Solve A x = B.
        // x = A^{-1} b = (1/11) [[3,-1],[-1,4]] [1,2]^T = [1/11, 7/11].
        let a = ndarray::array![[4.0, 1.0], [1.0, 3.0]];
        let b = ndarray::array![[1.0], [2.0]];
        let x = cholesky_solve(&a, &b).unwrap();
        assert!((x[[0, 0]] - 1.0 / 11.0).abs() < 1e-12);
        assert!((x[[1, 0]] - 7.0 / 11.0).abs() < 1e-12);
        // Residual A x - b is ~0.
        let r = a.dot(&x) - &b;
        assert!(r.iter().all(|v| v.abs() < 1e-12));
    }

    fn random_bipolar(p: usize, n: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut s = seed;
        (0..p)
            .map(|_| {
                (0..n)
                    .map(|_| if lcg(&mut s) < 0.5 { -1.0 } else { 1.0 })
                    .collect()
            })
            .collect()
    }

    fn overlap(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>() / a.len() as f64
    }

    #[test]
    fn stores_far_above_the_hebbian_limit() {
        // N neurons, P = N patterns: load factor P/N = 1.0, ~7x the Hebbian
        // limit of 0.14. Every stored pattern must be an exact fixed point at
        // this load, which a Hebbian network cannot do. Fails if the ridge solve
        // or retrieval is wrong.
        let n = 16usize;
        let patterns = random_bipolar(n, n, 12345);
        let sigma = (n as f64 / 2.0).sqrt(); // matches K = exp(-||.||^2 / N)
        let mem = KrrMemory::train(&patterns, sigma, 0.01).unwrap();
        for pat in &patterns {
            assert!(
                overlap(&mem.retrieve(pat, 25), pat) > 0.95,
                "stored pattern not a fixed point at P/N=1.0"
            );
        }
    }

    #[test]
    fn recovers_from_noise_at_moderate_load() {
        // P/N = 0.5 (8 patterns in 16 neurons), still ~3.5x the Hebbian limit.
        // At this load the attractor basins are large enough to correct a flipped
        // bit, which is the associative-memory (error-correcting) property.
        let n = 16usize;
        let patterns = random_bipolar(8, n, 999);
        let sigma = (n as f64 / 2.0).sqrt();
        let mem = KrrMemory::train(&patterns, sigma, 0.01).unwrap();
        for pat in &patterns {
            let mut noisy = pat.clone();
            noisy[0] = -noisy[0]; // one flipped bit
            assert!(
                overlap(&mem.retrieve(&noisy, 25), pat) > 0.95,
                "did not recover pattern from a one-bit flip at P/N=0.5"
            );
        }
    }

    #[test]
    fn fixed_point_property_holds_across_dimensions_and_seeds() {
        // Property generalization of `stores_far_above_the_hebbian_limit`: at a
        // moderate load (P/N = 0.5) every stored pattern must be an exact fixed
        // point, regardless of the ambient dimension or the random pattern set.
        // Sweeping the (n, seed) grid risks the ridge solve and retrieval far
        // more than a single fixture: a dimension- or configuration-specific
        // failure in the Cholesky solve or the kernel scaling would surface here
        // but pass the fixed-seed test.
        for &n in &[8usize, 12, 20, 32] {
            let sigma = (n as f64 / 2.0).sqrt();
            for seed in [1u64, 7, 42, 101, 2024] {
                let p = n / 2; // P/N = 0.5
                let patterns = random_bipolar(p, n, seed);
                let mem = KrrMemory::train(&patterns, sigma, 0.01)
                    .expect("train should succeed on valid bipolar patterns");
                for pat in &patterns {
                    let recalled = mem.retrieve(pat, 25);
                    assert!(
                        overlap(&recalled, pat) > 0.95,
                        "stored pattern not a fixed point at n={n}, seed={seed}, P/N=0.5"
                    );
                }
            }
        }
    }

    #[test]
    fn rejects_bad_input() {
        assert!(KrrMemory::train(&[], 1.0, 0.01).is_none());
        assert!(KrrMemory::train(&[vec![1.0, -1.0]], 0.0, 0.01).is_none());
        assert!(KrrMemory::train(&[vec![1.0], vec![1.0, -1.0]], 1.0, 0.01).is_none());
    }
}
