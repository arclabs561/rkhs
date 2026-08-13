# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "jax[cpu]==0.4.38",
# ]
# ///
"""Generate the nonlinear KQD reference fixture from the authors' code.

The upstream package is pinned to commit 34ecaf7. This script deliberately
uses only its ``rkhs_function`` and ``rkhs_norm_sq`` primitives, then applies
Algorithm 1 locally. It does not call the package's top-level e-KQD routine.

Regenerate with ``uv run scripts/generate_kqd_reference.py``. JAX is needed
only for regeneration; the Rust test reads the resulting JSON fixture.
"""

import ast
import hashlib
import json
from pathlib import Path
from typing import Callable
from urllib.request import urlopen

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, jit, vmap

UPSTREAM_COMMIT = "34ecaf75090f0482ab3fc6603d008d5ef3909b11"
UPSTREAM_SOURCE_SHA256 = (
    "f59942f2d4893269186a72452f0399f6f01da0a14647c509cd5dbe16b80ef87e"
)
UPSTREAM_SOURCE = (
    f"https://raw.githubusercontent.com/MashaNaslidnyk/kqe/{UPSTREAM_COMMIT}/kqe/kqd.py"
)

jax.config.update("jax_enable_x64", True)

X = jnp.array([[-1.0, 0.5], [0.25, 2.0], [2.0, -0.75]], dtype=jnp.float64)
Y = jnp.array([[-0.5, 1.5], [1.0, -1.0], [2.5, 0.25]], dtype=jnp.float64)
LANDMARKS = jnp.array([[-1.5, 0.0], [0.5, 1.0], [2.0, -1.5]], dtype=jnp.float64)
COEFFICIENTS = jnp.array([[0.75, -1.25, 0.5], [-0.4, 0.9, 1.1]], dtype=jnp.float64)
WEIGHTS = np.array([0.2, 0.3, 0.5], dtype=np.float64)
BANDWIDTH = 1.3
POWER = 2.0


def load_upstream_primitives() -> tuple[Callable, Callable]:
    """Load only the two upstream primitives from the pinned source file."""
    with urlopen(UPSTREAM_SOURCE) as response:  # noqa: S310 (constant HTTPS URL)
        source = response.read()
    actual_hash = hashlib.sha256(source).hexdigest()
    if actual_hash != UPSTREAM_SOURCE_SHA256:
        raise RuntimeError(
            f"expected upstream source {UPSTREAM_SOURCE_SHA256}, got {actual_hash}"
        )

    module = ast.parse(source, filename=UPSTREAM_SOURCE)
    wanted = {"rkhs_function", "rkhs_norm_sq"}
    definitions = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    if {node.name for node in definitions} != wanted:
        raise RuntimeError(
            "pinned upstream source does not define both RKHS primitives"
        )
    namespace = {
        "Array": Array,
        "Callable": Callable,
        "jit": jit,
        "jnp": jnp,
        "vmap": vmap,
    }
    exec(
        compile(ast.Module(body=definitions, type_ignores=[]), UPSTREAM_SOURCE, "exec"),
        namespace,
    )  # noqa: S102
    # The pinned decorators do not mark ``kernel_fn`` static, so the public JIT
    # wrappers reject a Python callable. Execute the exact wrapped function
    # bodies instead; these are the two upstream numerical primitives.
    return namespace["rkhs_function"].__wrapped__, namespace["rkhs_norm_sq"].__wrapped__


def rbf(left: jax.Array, right: jax.Array) -> jax.Array:
    squared_distance = jnp.sum((left - right) ** 2)
    return jnp.exp(-squared_distance / (2.0 * BANDWIDTH**2))


def main() -> None:
    rkhs_function, rkhs_norm_sq = load_upstream_primitives()

    gram = np.asarray(
        jax.vmap(lambda left: jax.vmap(lambda right: rbf(left, right))(LANDMARKS))(
            LANDMARKS
        )
    )
    directions = []
    tau_power_sum = 0.0
    for coefficients in COEFFICIENTS:
        norm_squared = float(rkhs_norm_sq(coefficients, rbf, LANDMARKS))
        projected_x = np.sort(
            np.asarray(rkhs_function(coefficients, rbf, LANDMARKS, X))
            / np.sqrt(norm_squared)
        )
        projected_y = np.sort(
            np.asarray(rkhs_function(coefficients, rbf, LANDMARKS, Y))
            / np.sqrt(norm_squared)
        )
        tau_power = float(np.dot(WEIGHTS, np.abs(projected_x - projected_y) ** POWER))
        tau_power_sum += tau_power
        directions.append(
            {
                "norm_squared": norm_squared,
                "sorted_x": projected_x.tolist(),
                "sorted_y": projected_y.tolist(),
                "tau_power": tau_power,
            }
        )

    fixture = {
        "provenance": {
            "generator": "scripts/generate_kqd_reference.py",
            "upstream": "https://github.com/MashaNaslidnyk/kqe",
            "upstream_commit": UPSTREAM_COMMIT,
            "method": (
                "unwrapped kqe.rkhs_function/rkhs_norm_sq followed by a local "
                "implementation of Algorithm 1"
            ),
        },
        "params": {"bandwidth": BANDWIDTH, "power": POWER, "normalize": True},
        "x": np.asarray(X).tolist(),
        "y": np.asarray(Y).tolist(),
        "landmarks": np.asarray(LANDMARKS).tolist(),
        "coefficients": np.asarray(COEFFICIENTS).tolist(),
        "weights": WEIGHTS.tolist(),
        "expected": {
            "landmark_gram": gram.tolist(),
            "directions": directions,
            "distance": float((tau_power_sum / len(COEFFICIENTS)) ** (1.0 / POWER)),
        },
    }

    output = Path(__file__).parents[1] / "tests" / "fixtures" / "kqd_reference.json"
    output.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
