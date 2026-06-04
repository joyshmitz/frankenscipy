use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fsci_ndimage::{BoundaryMode, NdArray, maximum_filter, minimum_filter};
use std::hint::black_box;

fn image(side: usize) -> NdArray {
    let data: Vec<f64> = (0..side * side)
        .map(|i| {
            let x = i as f64;
            (x * 0.017).sin() * 100.0 + (x * 0.0031).cos() * 37.0 + (i % 53) as f64
        })
        .collect();
    NdArray::new(data, vec![side, side]).expect("image")
}

fn bench_minmax_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("minmax_filter");
    let img = image(256);
    for &size in &[7usize, 15, 31] {
        group.bench_function(BenchmarkId::new("minimum_256x256", size), |b| {
            b.iter(|| {
                minimum_filter(black_box(&img), size, BoundaryMode::Reflect, 0.0).expect("min")
            })
        });
        group.bench_function(BenchmarkId::new("maximum_256x256", size), |b| {
            b.iter(|| {
                maximum_filter(black_box(&img), size, BoundaryMode::Reflect, 0.0).expect("max")
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_minmax_filter);
criterion_main!(benches);
