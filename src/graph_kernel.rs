//! Graph kernels: similarity measures between labeled graphs.
//!
//! These kernels measure structural similarity between graphs, connecting
//! RKHS methods to graph-structured data.
//!
//! Graphs are represented as adjacency lists (`&[Vec<usize>]`) with
//! optional node labels (`&[u64]`). Node indices are 0-based.

use std::collections::HashMap;

/// Weisfeiler-Leman subtree kernel (1-WL).
///
/// Counts matching subtree patterns at each depth up to `max_depth`.
/// At each iteration, node labels are refined by hashing the multiset
/// of neighbor labels, capturing increasingly large subtree structure.
///
/// The kernel value is the inner product of the aggregated label histograms
/// across all depths.
///
/// # Arguments
///
/// * `adj1` - Adjacency list for graph 1
/// * `labels1` - Initial node labels for graph 1
/// * `adj2` - Adjacency list for graph 2
/// * `labels2` - Initial node labels for graph 2
/// * `max_depth` - Number of WL refinement iterations
///
/// # Example
///
/// ```rust
/// use rkhs::wl_subtree_kernel;
///
/// // Triangle graph
/// let adj = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
/// let labels = vec![1, 1, 1];
///
/// let k = wl_subtree_kernel(&adj, &labels, &adj, &labels, 2);
/// assert!(k > 0.0);
/// ```
pub fn wl_subtree_kernel(
    adj1: &[Vec<usize>],
    labels1: &[u64],
    adj2: &[Vec<usize>],
    labels2: &[u64],
    max_depth: usize,
) -> f64 {
    let n1 = adj1.len();
    let n2 = adj2.len();

    if n1 == 0 && n2 == 0 {
        return 0.0;
    }

    let mut current1: Vec<u64> = labels1.to_vec();
    let mut current2: Vec<u64> = labels2.to_vec();

    let mut next_id: u64 = {
        // Start IDs after the max initial label to avoid collisions
        let max_label = labels1
            .iter()
            .chain(labels2.iter())
            .copied()
            .max()
            .unwrap_or(0);
        max_label + 1
    };

    // Signature -> compressed label mapping, shared across both graphs
    // so that identical structural signatures get the same label.
    let mut signature_map: HashMap<(u64, Vec<u64>), u64> = HashMap::new();

    let mut total = 0.0;

    // Count matching labels at depth 0
    total += histogram_dot(&current1, &current2);

    for _depth in 0..max_depth {
        let new1 = refine_labels(adj1, &current1, &mut signature_map, &mut next_id);
        let new2 = refine_labels(adj2, &current2, &mut signature_map, &mut next_id);

        current1 = new1;
        current2 = new2;

        total += histogram_dot(&current1, &current2);
    }

    total
}

/// Refine node labels by hashing each node's label with its sorted neighbor labels.
///
/// The `signature_map` must be shared across both graphs within a kernel call
/// so that identical structural patterns receive the same compressed label.
fn refine_labels(
    adj: &[Vec<usize>],
    labels: &[u64],
    signature_map: &mut HashMap<(u64, Vec<u64>), u64>,
    next_id: &mut u64,
) -> Vec<u64> {
    let n = adj.len();
    let mut new_labels = Vec::with_capacity(n);

    for i in 0..n {
        let mut neighbor_labels: Vec<u64> = adj[i].iter().map(|&j| labels[j]).collect();
        neighbor_labels.sort_unstable();

        let sig = (labels[i], neighbor_labels);

        let new_label = *signature_map.entry(sig).or_insert_with(|| {
            let id = *next_id;
            *next_id += 1;
            id
        });

        new_labels.push(new_label);
    }

    new_labels
}

/// Inner product of two label histograms: sum over all labels of count_1(l) * count_2(l).
fn histogram_dot(labels1: &[u64], labels2: &[u64]) -> f64 {
    let mut counts1: HashMap<u64, u64> = HashMap::new();
    for &l in labels1 {
        *counts1.entry(l).or_insert(0) += 1;
    }

    let mut counts2: HashMap<u64, u64> = HashMap::new();
    for &l in labels2 {
        *counts2.entry(l).or_insert(0) += 1;
    }

    let mut dot = 0u64;
    for (&label, &c1) in &counts1 {
        if let Some(&c2) = counts2.get(&label) {
            dot += c1 * c2;
        }
    }

    dot as f64
}

/// Random walk kernel via direct product graph.
///
/// Counts matching random walks between two graphs with geometric decay.
/// `k(G1, G2) = sum_{l=0}^{max_len} lambda^l * (matching walks of length l)`
///
/// A matching walk of length `l` is a sequence of node pairs
/// `(u0,v0), (u1,v1), ..., (ul,vl)` where each `(ui, ui+1)` is an edge
/// in G1 and `(vi, vi+1)` is an edge in G2.
///
/// # Arguments
///
/// * `adj1` - Adjacency list for graph 1
/// * `adj2` - Adjacency list for graph 2
/// * `max_len` - Maximum walk length
/// * `lambda` - Decay factor per step (should be < 1 / max(degree) for convergence)
///
/// # Example
///
/// ```rust
/// use rkhs::random_walk_kernel;
///
/// // Two edges
/// let adj1 = vec![vec![1], vec![0]];
/// let adj2 = vec![vec![1], vec![0]];
///
/// let k = random_walk_kernel(&adj1, &adj2, 3, 0.1);
/// assert!(k > 0.0);
/// ```
pub fn random_walk_kernel(
    adj1: &[Vec<usize>],
    adj2: &[Vec<usize>],
    max_len: usize,
    lambda: f64,
) -> f64 {
    let n1 = adj1.len();
    let n2 = adj2.len();

    if n1 == 0 || n2 == 0 {
        return 0.0;
    }

    let np = n1 * n2; // product graph size

    // Build product graph adjacency.
    // Product node (i, j) has index i * n2 + j.
    // Edge (i,j) -> (i',j') exists iff (i,i') in E1 and (j,j') in E2.
    let mut prod_adj: Vec<Vec<usize>> = vec![Vec::new(); np];
    for (i, neighbors_i) in adj1.iter().enumerate() {
        for &i_prime in neighbors_i {
            for (j, neighbors_j) in adj2.iter().enumerate() {
                for &j_prime in neighbors_j {
                    let from = i * n2 + j;
                    let to = i_prime * n2 + j_prime;
                    prod_adj[from].push(to);
                }
            }
        }
    }

    // Count walks using iterative BFS-style propagation.
    // walk_count[node] = number of walks ending at this node at current length.
    let mut walk_count = vec![1.0_f64; np]; // length 0: one walk per product node
    let mut total: f64 = walk_count.iter().sum::<f64>(); // lambda^0 * count

    for l in 1..=max_len {
        let mut next_count = vec![0.0_f64; np];
        for (node, neighbors) in prod_adj.iter().enumerate() {
            for &neighbor in neighbors {
                next_count[neighbor] += walk_count[node];
            }
        }
        let factor = lambda.powi(l as i32);
        total += factor * next_count.iter().sum::<f64>();
        walk_count = next_count;
    }

    total
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // WL Subtree Kernel
    // =========================================================================

    #[test]
    fn wl_identical_graphs() {
        // Triangle
        let adj = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
        let labels = vec![1, 1, 1];

        let k_same = wl_subtree_kernel(&adj, &labels, &adj, &labels, 3);

        // Path graph (different structure)
        let adj2 = vec![vec![1], vec![0, 2], vec![1]];
        let labels2 = vec![1, 1, 1];

        let k_diff = wl_subtree_kernel(&adj, &labels, &adj2, &labels2, 3);

        assert!(
            k_same > k_diff,
            "identical graphs should have higher kernel than different, got same={k_same} diff={k_diff}"
        );
    }

    #[test]
    fn wl_symmetric() {
        let adj1 = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
        let labels1 = vec![1, 2, 1];

        let adj2 = vec![vec![1], vec![0, 2], vec![1]];
        let labels2 = vec![1, 1, 2];

        let k_12 = wl_subtree_kernel(&adj1, &labels1, &adj2, &labels2, 2);
        let k_21 = wl_subtree_kernel(&adj2, &labels2, &adj1, &labels1, 2);

        assert!(
            (k_12 - k_21).abs() < 1e-10,
            "WL kernel should be symmetric, got {k_12} vs {k_21}"
        );
    }

    #[test]
    fn wl_empty_graphs() {
        let adj: Vec<Vec<usize>> = vec![];
        let labels: Vec<u64> = vec![];

        let k = wl_subtree_kernel(&adj, &labels, &adj, &labels, 3);
        assert!((k - 0.0).abs() < 1e-10);
    }

    #[test]
    fn wl_different_labels_matter() {
        let adj = vec![vec![1], vec![0]];
        let labels_a = vec![1, 1];
        let labels_b = vec![1, 2];

        let k_same = wl_subtree_kernel(&adj, &labels_a, &adj, &labels_a, 1);
        let k_diff = wl_subtree_kernel(&adj, &labels_a, &adj, &labels_b, 1);

        assert!(
            k_same > k_diff,
            "same labels should give higher kernel than different labels"
        );
    }

    // =========================================================================
    // Random Walk Kernel
    // =========================================================================

    #[test]
    fn rw_self_geq_other() {
        // Triangle
        let adj1 = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
        // Path
        let adj2 = vec![vec![1], vec![0, 2], vec![1]];

        let k_self = random_walk_kernel(&adj1, &adj1, 3, 0.1);
        let k_other = random_walk_kernel(&adj1, &adj2, 3, 0.1);

        assert!(
            k_self >= k_other,
            "self-kernel should be >= cross-kernel, got self={k_self} other={k_other}"
        );
    }

    #[test]
    fn rw_symmetric() {
        let adj1 = vec![vec![1, 2], vec![0], vec![0]];
        let adj2 = vec![vec![1], vec![0, 2], vec![1]];

        let k_12 = random_walk_kernel(&adj1, &adj2, 3, 0.1);
        let k_21 = random_walk_kernel(&adj2, &adj1, 3, 0.1);

        assert!(
            (k_12 - k_21).abs() < 1e-10,
            "random walk kernel should be symmetric, got {k_12} vs {k_21}"
        );
    }

    #[test]
    fn rw_empty_graph() {
        let adj1: Vec<Vec<usize>> = vec![];
        let adj2 = vec![vec![1], vec![0]];

        let k = random_walk_kernel(&adj1, &adj2, 3, 0.1);
        assert!((k - 0.0).abs() < 1e-10);
    }

    #[test]
    fn rw_disconnected_nodes() {
        // Two isolated nodes (no edges)
        let adj = vec![vec![], vec![]];

        let k = random_walk_kernel(&adj, &adj, 3, 0.1);
        // Only length-0 walks exist: 2*2 = 4 product nodes, each with 1 walk
        assert!(
            (k - 4.0).abs() < 1e-10,
            "isolated nodes should have kernel = n1*n2, got {k}"
        );
    }
}
