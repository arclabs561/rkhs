# rkhs examples

Examples for kernel distances, kernel thinning, compatibility re-exports from
`hopfield`, and small spectral diagnostics.

## Running

```sh
cargo run --example distribution_mmd
cargo run --example clam_vs_kmeans
cargo run --example kernel_thinning
cargo run --example kernel_dmd
cargo run --example kernel_rmt
```

Use `cargo test --examples` to compile every example.

## Task map

| Goal | Example | What to inspect |
|---|---|---|
| Compare two sets of samples | `distribution_mmd` | MMD over topic distributions with a Jensen-Shannon kernel. The output compares a science-heavy corpus against an arts-heavy corpus and a held-out science-heavy corpus. |
| See dense associative-memory retrieval | `associative_memory` | LSE and LSR energies on stored 2D memories, including corrupted-pattern recovery and compact-support behavior. |
| Compare AM clustering with k-means | `clam_vs_kmeans` | Hard-label agreement plus a beta sweep showing where soft assignments expose uncertainty. |
| Walk through ClAM mechanics | `clam_clustering` | Centroid memories, soft assignments, contraction steps, and the loss used for differentiable clustering. |
| Select a representative coreset | `kernel_thinning` | Kernel thinning on two Gaussian clusters, compared with the best random subset from repeated trials. |
| Fit a one-step nonlinear dynamics proxy | `kernel_dmd` | RBF features plus a ridge-fitted linear operator predict the next feature state better than identity. |
| Diagnose a Gram matrix spectrum | `gram_spectrum_rmt` | Signal dimensions above the Marchenko-Pastur noise floor for clustered versus noisy data. |
| Study high-dimensional MMD limits | `kernel_rmt` | How the Gram spectrum and MMD detection rate change as the dimension/sample ratio changes. |

## Reading path

Start with `distribution_mmd` for kernel distances between empirical
distributions. Use the `clam_*` examples only if you need the compatibility
re-exports from `hopfield`. Use the RMT examples when you need to understand
why a kernel matrix has stopped separating signal from noise.
