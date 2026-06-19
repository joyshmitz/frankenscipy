use criterion::{Criterion, criterion_group, criterion_main};
use fsci_cluster::{gaussian_mixture, kmeans, kmeans2};

/// Deterministic blobs: `n` points in `d` dims drawn around 4 cluster centres.
fn blobs(n: usize, d: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| {
            let centre = (i % 4) as f64;
            (0..d)
                .map(|j| {
                    let t = (i * (j + 1)) as f64;
                    centre * 5.0 + (t * 0.013).sin() * 0.5 + ((i + j) % 7) as f64 * 0.05
                })
                .collect()
        })
        .collect()
}

/// kmeans (Lloyd) and kmeans2 (double-buffered Lloyd loop, frankenscipy-4ylee).
fn bench_kmeans(c: &mut Criterion) {
    let data = blobs(2000, 4);
    let init: Vec<Vec<f64>> = (0..4).map(|k| vec![k as f64 * 5.0; 4]).collect();
    let mut group = c.benchmark_group("kmeans");
    group.bench_function("k4/n2000", |b| b.iter(|| kmeans(&data, 4, 50, 42)));
    group.bench_function("kmeans2/k4/n2000", |b| b.iter(|| kmeans2(&data, &init, 50)));
    group.finish();
}

/// Gaussian-mixture EM (per-point logp/log_norm hoisted, frankenscipy-5ufms).
fn bench_gmm(c: &mut Criterion) {
    let data = blobs(1000, 3);
    let mut group = c.benchmark_group("gmm");
    group.bench_function("k3/n1000", |b| {
        b.iter(|| gaussian_mixture(&data, 3, 50, 1e-4, 1e-6, 42))
    });
    group.finish();
}

/// Hierarchical clustering: NN-chain linkage + cophenetic distances (the cophenet
/// member-list move-instead-of-clone win, frankenscipy-jphzn).
fn bench_hierarchical(c: &mut Criterion) {
    use fsci_cluster::{LinkageMethod, cophenet, linkage};
    let data = blobs(400, 4);
    let z = linkage(&data, LinkageMethod::Average).expect("linkage");
    let mut group = c.benchmark_group("hierarchical");
    group.bench_function("linkage_average/n400", |b| {
        b.iter(|| linkage(&data, LinkageMethod::Average))
    });
    group.bench_function("cophenet/n400", |b| b.iter(|| cophenet(&z)));
    group.finish();
}

criterion_group!(benches, bench_kmeans, bench_gmm, bench_hierarchical);
criterion_main!(benches);
