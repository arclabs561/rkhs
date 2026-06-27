# /// script
# requires-python = ">=3.10"
# dependencies = ["scikit-learn", "numpy"]
# ///
"""Rosetta fixture generator for rkhs kernels, Gram matrices, and MMD.

Provenance for rkhs_kernels.json. Kernel Gram matrices are computed by
scikit-learn (rbf/polynomial/linear/laplacian via sklearn.metrics.pairwise,
Matern via sklearn.gaussian_process.kernels). These are exact external oracles.

MMD has no scikit-learn function, so mmd_biased / mmd_unbiased are computed in
numpy from the sklearn RBF kernel matrices using the canonical estimators
(Gretton et al. 2012). That is a cross-implementation check (numpy vs rkhs both
implementing the documented formula over the same kernel), weaker than a library
oracle but still catches a transcription error in either side.

Regenerate: uv run tests/fixtures/rosetta/gen_rkhs.py

Excluded (no external library oracle): the KDE-family kernels (epanechnikov,
triangle, cosine, quartic, triweight, tricube), already covered by closed-form
unit tests in the crate.
"""

import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics.pairwise import (
    laplacian_kernel,
    linear_kernel,
    polynomial_kernel,
    rbf_kernel,
)

SEED = 0
rng = np.random.default_rng(SEED)

# Two small point sets, values O(1) so the squared-norm expansion sklearn uses
# does not lose precision against rkhs's direct (xi-yi)^2 summation.
X = rng.normal(0.0, 1.0, size=(6, 3))
Y = rng.normal(0.5, 1.2, size=(5, 3))

# Parameters. rkhs rbf uses exp(-||x-y||^2/(2 sigma^2)); sklearn rbf_kernel uses
# exp(-gamma ||x-y||^2), so gamma = 1/(2 sigma^2). rkhs laplacian uses
# exp(-||x-y||_1/sigma); sklearn laplacian_kernel uses exp(-gamma ||x-y||_1), so
# gamma = 1/sigma.
sigma = 1.3
rbf_gamma = 1.0 / (2.0 * sigma * sigma)
poly_degree = 3
poly_gamma = 0.5
poly_coef0 = 1.0
laplacian_sigma = 0.8
laplacian_gamma = 1.0 / laplacian_sigma
matern_lengthscale = 1.1

gram_rbf = rbf_kernel(X, gamma=rbf_gamma)
gram_poly = polynomial_kernel(X, degree=poly_degree, gamma=poly_gamma, coef0=poly_coef0)
gram_linear = linear_kernel(X)
gram_laplacian = laplacian_kernel(X, gamma=laplacian_gamma)
gram_matern12 = Matern(length_scale=matern_lengthscale, nu=0.5)(X)
gram_matern32 = Matern(length_scale=matern_lengthscale, nu=1.5)(X)
gram_matern52 = Matern(length_scale=matern_lengthscale, nu=2.5)(X)

# MMD from RBF kernel matrices (same gamma/sigma as above).
kxx = rbf_kernel(X, X, gamma=rbf_gamma)
kyy = rbf_kernel(Y, Y, gamma=rbf_gamma)
kxy = rbf_kernel(X, Y, gamma=rbf_gamma)
m, n = len(X), len(Y)
mmd_biased = max(0.0, kxx.mean() + kyy.mean() - 2.0 * kxy.mean())
mmd_unbiased = (
    (kxx.sum() - np.trace(kxx)) / (m * (m - 1))
    + (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
    - 2.0 * kxy.mean()
)

fixture = {
    "provenance": {
        "generator": "gen_rkhs.py",
        "library": "scikit-learn",
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "python": platform.python_version(),
        "seed": SEED,
        "note": "Gram matrices are sklearn oracles; MMD is numpy-from-sklearn-kernel.",
    },
    "params": {
        "sigma": sigma,
        "poly_degree": poly_degree,
        "poly_gamma": poly_gamma,
        "poly_coef0": poly_coef0,
        "laplacian_sigma": laplacian_sigma,
        "matern_lengthscale": matern_lengthscale,
    },
    "x": X.tolist(),
    "y": Y.tolist(),
    "expected": {
        "gram_rbf": gram_rbf.tolist(),
        "gram_poly": gram_poly.tolist(),
        "gram_linear": gram_linear.tolist(),
        "gram_laplacian": gram_laplacian.tolist(),
        "gram_matern12": gram_matern12.tolist(),
        "gram_matern32": gram_matern32.tolist(),
        "gram_matern52": gram_matern52.tolist(),
        "mmd_biased_rbf": float(mmd_biased),
        "mmd_unbiased_rbf": float(mmd_unbiased),
    },
}

out = Path(__file__).parent / "rkhs_kernels.json"
out.write_text(json.dumps(fixture, indent=2) + "\n")
print(f"mmd_biased_rbf   {mmd_biased:.12f}")
print(f"mmd_unbiased_rbf {mmd_unbiased:.12f}")
print(f"wrote {out}")
