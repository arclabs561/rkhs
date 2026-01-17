//! Scaling benchmark for RKHS Gram matrix: Old Vec<Vec> vs New Optimized ndarray.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::Array2;
use rand::prelude::*;
use rkhs::{kernel_matrix, rbf, rbf_kernel_matrix_ndarray};

fn create_data(n: usize, dim: usize) -> Array2<f64> {
    let mut rng = rand::rng();
    Array2::from_shape_fn((n, dim), |_| rng.random::<f64>())
}

fn bench_gram_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("rbf_gram_scaling");
    let dim = 128;

    for n in [128usize, 256, 512, 1024] {
        let data = create_data(n, dim);
        let sigma = 1.0;

        group.throughput(Throughput::Elements(n as u64));

        // 1. Old path: requires Vec<Vec<f64>>
        let vecs: Vec<Vec<f64>> = data.rows().into_iter().map(|r| r.to_vec()).collect();
        group.bench_with_input(BenchmarkId::new("old_vecvec", n), &n, |b, _| {
            b.iter(|| black_box(kernel_matrix(black_box(&vecs), |x, y| rbf(x, y, sigma))))
        });

        // 2. New optimized path: ndarray view
        group.bench_with_input(BenchmarkId::new("new_ndarray", n), &n, |b, _| {
            b.iter(|| black_box(rbf_kernel_matrix_ndarray(black_box(data.view()), sigma)))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_gram_scaling);
criterion_main!(benches);
