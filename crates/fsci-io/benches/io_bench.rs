use criterion::{Criterion, criterion_group, criterion_main};
use fsci_io::{loadtxt, mmread, mmwrite, savetxt};

fn matrix(rows: usize, cols: usize) -> Vec<f64> {
    (0..rows * cols).map(|i| i as f64 * 0.001 + 1.0).collect()
}

/// Whitespace-delimited text round trip. savetxt writes cells with `write!` into
/// the buffer (frankenscipy-d1uxy); loadtxt's serial path parses straight into the
/// output buffer (frankenscipy-fwnb1) — both exercised here at <4096 rows.
fn bench_text_io(c: &mut Criterion) {
    let (rows, cols) = (500usize, 20usize);
    let data = matrix(rows, cols);
    let text = savetxt(rows, cols, &data, " ").expect("savetxt");
    let mut group = c.benchmark_group("text_io");
    group.bench_function("savetxt/500x20", |b| {
        b.iter(|| savetxt(rows, cols, &data, " "))
    });
    group.bench_function("loadtxt/500x20", |b| b.iter(|| loadtxt(&text)));
    group.finish();
}

/// MatrixMarket array round trip. mmwrite uses `writeln!` per entry; mmread parses
/// off the split iterator (frankenscipy-1f4yh for coordinate).
fn bench_matrix_market(c: &mut Criterion) {
    let (rows, cols) = (100usize, 100usize);
    let data = matrix(rows, cols);
    let mm = mmwrite(rows, cols, &data).expect("mmwrite");
    let mut group = c.benchmark_group("matrix_market");
    group.bench_function("mmwrite/100x100", |b| b.iter(|| mmwrite(rows, cols, &data)));
    group.bench_function("mmread/100x100", |b| b.iter(|| mmread(&mm)));
    group.finish();
}

criterion_group!(benches, bench_text_io, bench_matrix_market);
criterion_main!(benches);
