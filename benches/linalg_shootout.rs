use criterion::{black_box, criterion_group, criterion_main, Criterion};
use faer::{Mat, Parallelism};
use ndarray::Array2;

fn bench_matmul(c: &mut Criterion) {
    let sizes = [128, 512, 1024];
    let mut group = c.benchmark_group("Matrix Multiplication");

    for &n in &sizes {
        // ndarray setup
        let nd_a = Array2::<f64>::zeros((n, n));
        let nd_b = Array2::<f64>::zeros((n, n));

        // faer setup
        let faer_a = Mat::<f64>::zeros(n, n);
        let faer_b = Mat::<f64>::zeros(n, n);

        group.bench_function(format!("ndarray_{n}"), |b| {
            b.iter(|| black_box(nd_a.dot(&nd_b)))
        });

        group.bench_function(format!("faer_{n}_serial"), |b| {
            b.iter(|| {
                let mut c = Mat::<f64>::zeros(n, n);
                faer::linalg::matmul::matmul(
                    c.as_mut(),
                    faer_a.as_ref(),
                    faer_b.as_ref(),
                    None,
                    1.0,
                    Parallelism::None,
                );
                black_box(c)
            })
        });

        group.bench_function(format!("faer_{n}_parallel"), |b| {
            b.iter(|| {
                let mut c = Mat::<f64>::zeros(n, n);
                faer::linalg::matmul::matmul(
                    c.as_mut(),
                    faer_a.as_ref(),
                    faer_b.as_ref(),
                    None,
                    1.0,
                    Parallelism::Rayon(0), // 0 = auto-detect
                );
                black_box(c)
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_matmul);
criterion_main!(benches);
