//! MMD between sets of distributions using distribution kernels.
//!
//! Scenario: compare "corpora" represented as sets of topic distributions.
//! Each document is a distribution over 6 topics. Two corpora are similar
//! if their document-topic distributions are similar.
//!
//! Uses the Jensen-Shannon kernel so that MMD operates in a distribution RKHS
//! rather than a pointwise RKHS.
//!
//! Run: cargo run --example distribution_mmd

use rkhs::{jensen_shannon_kernel, mmd_biased};

/// Generate a distribution peaked at `peak` over `n_topics` topics.
///
/// Assigns `concentration` weight to `peak`, spreads the rest uniformly.
fn peaked_distribution(peak: usize, n_topics: usize, concentration: f64) -> Vec<f64> {
    let residual = (1.0 - concentration) / (n_topics - 1) as f64;
    let mut dist = vec![residual; n_topics];
    dist[peak] = concentration;
    dist
}

/// Build a corpus of `n` distributions peaked at topics in `peak_range`,
/// cycling through the peaks.
fn make_corpus(peak_range: &[usize], n: usize, n_topics: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| {
            let peak = peak_range[i % peak_range.len()];
            // Vary concentration slightly so documents aren't identical.
            let concentration = 0.55 + 0.03 * (i as f64);
            peaked_distribution(peak, n_topics, concentration)
        })
        .collect()
}

fn main() {
    let n_topics = 6;

    // Corpus A: science-heavy (topics 0, 1, 2)
    let corpus_a = make_corpus(&[0, 1, 2], 10, n_topics);
    // Corpus B: arts-heavy (topics 3, 4, 5)
    let corpus_b = make_corpus(&[3, 4, 5], 10, n_topics);
    // Corpus C: another science corpus, slightly different concentrations.
    let corpus_c: Vec<Vec<f64>> = (0..10)
        .map(|i| {
            let peak = [0, 1, 2][i % 3];
            let concentration = 0.45 + 0.02 * (i as f64);
            peaked_distribution(peak, n_topics, concentration)
        })
        .collect();

    let lambda = 1.0;
    let kernel = |a: &[f64], b: &[f64]| jensen_shannon_kernel(a, b, lambda);

    let mmd_ab = mmd_biased(&corpus_a, &corpus_b, kernel);
    let mmd_ac = mmd_biased(&corpus_a, &corpus_c, kernel);
    let mmd_aa = mmd_biased(&corpus_a, &corpus_a, kernel);

    println!("=== MMD Between Corpora of Topic Distributions ===");
    println!();
    println!("Each corpus: 10 documents, 6 topics, Jensen-Shannon kernel (lambda={lambda})");
    println!("  Corpus A: science-heavy (peaked at topics 0-2)");
    println!("  Corpus B: arts-heavy    (peaked at topics 3-5)");
    println!("  Corpus C: science-heavy (peaked at topics 0-2, same structure as A)");
    println!();
    println!("MMD^2(A, B) = {mmd_ab:.6}   (different topic profiles -> large)");
    println!("MMD^2(A, C) = {mmd_ac:.6}   (similar topic profiles  -> small)");
    println!("MMD^2(A, A) = {mmd_aa:.6}   (self-comparison         -> zero)");
    println!();

    assert!(
        mmd_ab > mmd_ac,
        "cross-domain MMD should exceed within-domain MMD"
    );
    assert!(mmd_aa < 1e-12, "self-MMD should be zero, got {mmd_aa}");

    // Show a few sample distributions for context.
    println!("Sample distributions (first 3 per corpus):");
    for (name, corpus) in [("A", &corpus_a), ("B", &corpus_b), ("C", &corpus_c)] {
        println!("  Corpus {name}:");
        for (i, doc) in corpus.iter().take(3).enumerate() {
            let formatted: Vec<String> = doc.iter().map(|v| format!("{v:.3}")).collect();
            println!("    doc {i}: [{}]", formatted.join(", "));
        }
    }
}
