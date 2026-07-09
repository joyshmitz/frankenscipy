use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fsci_sparse::{
    COO_SUM_DUPLICATES_RADIX_DISABLE, CooMatrix, CscMatrix, CsrMatrix, FormatConvertible,
    IluOptions, Shape2D, SolveOptions, add_csr, block_diag, diags, eye, kron, random, scale_csr,
    spilu, spmm, spmv_csr, spsolve,
};
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Duration;

/// Matrix configurations: (rows/cols, density).
const CONFIGS: &[(usize, f64)] = &[
    (100, 0.05),     // 100×100, 5%
    (1_000, 0.01),   // 1000×1000, 1%
    (10_000, 0.001), // 10000×10000, 0.1%
];
const TINY_DENSITY_CASES: &[(usize, f64)] = &[(1_000_000_000, 1e-19), (2_000_000_000, 1e-20)];

const SEED: u64 = 0xBEEF_CAFE;

fn make_random_csr(n: usize, density: f64) -> CsrMatrix {
    random(Shape2D::new(n, n), density, SEED)
        .expect("random coo")
        .to_csr()
        .expect("coo->csr")
}

fn make_random_rect_csr(rows: usize, cols: usize, density: f64, seed: u64) -> CsrMatrix {
    random(Shape2D::new(rows, cols), density, seed)
        .expect("random coo")
        .to_csr()
        .expect("coo->csr")
}

fn make_vector(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i as f64) * 0.01 - 0.5).collect()
}

fn bench_csr_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_csr_construction");

    for &(n, density) in CONFIGS {
        let coo = random(Shape2D::new(n, n), density, SEED).expect("random coo");
        group.bench_with_input(
            BenchmarkId::new(format!("{n}x{n}_d{}", (density * 100.0) as u32), n),
            &coo,
            |b, coo| {
                b.iter(|| {
                    let csr = black_box(coo).to_csr().expect("coo->csr");
                    black_box(csr.nnz());
                });
            },
        );
    }

    group.finish();
}

fn bench_spmv(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_spmv");

    for &(n, density) in CONFIGS {
        let csr = make_random_csr(n, density);
        let vec = make_vector(n);
        group.bench_with_input(
            BenchmarkId::new(
                format!("{n}x{n}_d{}_nnz{}", (density * 100.0) as u32, csr.nnz()),
                n,
            ),
            &(csr, vec),
            |b, (csr, vec)| {
                b.iter(|| {
                    let result = spmv_csr(black_box(csr), black_box(vec)).expect("spmv");
                    black_box(result.len());
                });
            },
        );
    }

    group.finish();
}

fn bench_format_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_format_conversion");

    for &(n, density) in CONFIGS {
        let csr = make_random_csr(n, density);
        let label = format!("{n}x{n}_d{}", (density * 100.0) as u32);

        group.bench_with_input(
            BenchmarkId::new(format!("{label}_csr_to_csc"), n),
            &csr,
            |b, csr: &CsrMatrix| {
                b.iter(|| {
                    let csc = black_box(csr).to_csc().expect("csr->csc");
                    black_box(csc.nnz());
                });
            },
        );

        let csc = csr.to_csc().expect("csr->csc");
        group.bench_with_input(
            BenchmarkId::new(format!("{label}_csc_to_csr"), n),
            &csc,
            |b, csc: &CscMatrix| {
                b.iter(|| {
                    let csr = black_box(csc).to_csr().expect("csc->csr");
                    black_box(csr.nnz());
                });
            },
        );
    }

    group.finish();
}

fn bench_arithmetic(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_arithmetic");

    for &(n, density) in CONFIGS {
        let a = make_random_csr(n, density);
        let b = random(Shape2D::new(n, n), density, SEED ^ 0xFF)
            .expect("random b")
            .to_csr()
            .expect("b csr");
        let label = format!("{n}x{n}_d{}", (density * 100.0) as u32);

        group.bench_with_input(
            BenchmarkId::new(format!("{label}_add"), n),
            &(a.clone(), b),
            |b_iter, (a, b)| {
                b_iter.iter(|| {
                    let sum = add_csr(black_box(a), black_box(b)).expect("add");
                    black_box(sum.nnz());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("{label}_scale"), n),
            &a,
            |b_iter, a| {
                b_iter.iter(|| {
                    let scaled = scale_csr(black_box(a), black_box(2.5)).expect("scale");
                    black_box(scaled.nnz());
                });
            },
        );
    }

    group.finish();
}

fn bench_eye(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_eye");

    for &n in &[100, 1_000, 10_000] {
        group.bench_with_input(BenchmarkId::new("eye", n), &n, |b, &n| {
            b.iter(|| {
                let id = eye(black_box(n)).expect("eye");
                black_box(id.nnz());
            });
        });
    }

    group.finish();
}

fn bench_diags(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_diags");

    for &n in &[100, 1_000, 10_000] {
        group.bench_with_input(BenchmarkId::new("tridiag", n), &n, |b, &n: &usize| {
            let sub = vec![-1.0; n.saturating_sub(1)];
            let main = vec![2.0; n];
            let sup = vec![-1.0; n.saturating_sub(1)];
            b.iter(|| {
                let csr = diags(
                    black_box(&[sub.clone(), main.clone(), sup.clone()]),
                    &[-1, 0, 1],
                    Some(Shape2D::new(n, n)),
                )
                .expect("tridiag");
                black_box(csr.nnz());
            });
        });
    }

    group.finish();
}

fn bench_spmm(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_spmm");
    for &(n, density) in &[(500usize, 0.02f64), (1_000, 0.01), (2_000, 0.01)] {
        let a = make_random_csr(n, density);
        let b = random(Shape2D::new(n, n), density, SEED ^ 0x1234)
            .expect("b coo")
            .to_csr()
            .expect("b csr");
        let label = format!("{n}x{n}_d{}", (density * 100.0) as u32);
        group.bench_with_input(BenchmarkId::new(label, n), &(a, b), |bi, (a, b)| {
            bi.iter(|| {
                let prod = spmm(black_box(a), black_box(b));
                black_box(prod.nnz());
            });
        });
    }
    group.finish();
}

fn bench_kron(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_kron");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));

    let a = make_random_rect_csr(400, 400, 0.02, SEED);
    let b = make_random_rect_csr(120, 120, 0.05, SEED ^ 0x5C1F);
    let label = format!("400x400d2_120x120d5_nnz{}_{}", a.nnz(), b.nnz());

    group.bench_with_input(
        BenchmarkId::new(label, a.nnz() * b.nnz()),
        &(a, b),
        |bi, (a, b)| {
            bi.iter(|| {
                let product = kron(black_box(a), black_box(b)).expect("kron");
                black_box(product.nnz());
            });
        },
    );

    group.finish();
}

fn make_spilu_banded_csc(n: usize, half_bandwidth: usize) -> CscMatrix {
    let entries_per_row = half_bandwidth.saturating_mul(2).saturating_add(1);
    let mut data = Vec::with_capacity(n.saturating_mul(entries_per_row));
    let mut rows = Vec::with_capacity(data.capacity());
    let mut cols = Vec::with_capacity(data.capacity());

    for row in 0..n {
        let start = row.saturating_sub(half_bandwidth);
        let end = row.saturating_add(half_bandwidth).min(n.saturating_sub(1));
        for col in start..=end {
            rows.push(row);
            cols.push(col);
            if row == col {
                data.push(entries_per_row as f64 + 2.0 + (row % 17) as f64 * 0.001);
            } else {
                data.push(-1.0 / (row.abs_diff(col) + 1) as f64);
            }
        }
    }

    CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false)
        .expect("spilu banded coo")
        .to_csc()
        .expect("spilu banded csc")
}

fn make_laplacian_2d(k: usize) -> CsrMatrix {
    let n = k * k;
    let mut rows = Vec::with_capacity(n * 5);
    let mut cols = Vec::with_capacity(n * 5);
    let mut data = Vec::with_capacity(n * 5);
    let idx = |r: usize, c: usize| r * k + c;
    for r in 0..k {
        for c in 0..k {
            let i = idx(r, c);
            rows.push(i);
            cols.push(i);
            data.push(4.001);
            for (dr, dc) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                let nr = r as i64 + dr;
                let nc = c as i64 + dc;
                if nr >= 0 && nr < k as i64 && nc >= 0 && nc < k as i64 {
                    rows.push(i);
                    cols.push(idx(nr as usize, nc as usize));
                    data.push(-1.0);
                }
            }
        }
    }
    CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false)
        .expect("laplacian coo")
        .to_csr()
        .expect("laplacian csr")
}

fn bench_spilu(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_spilu");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));

    for &(n, half_bandwidth) in &[(512usize, 16usize), (1_024, 32)] {
        let matrix = make_spilu_banded_csc(n, half_bandwidth);
        group.bench_with_input(
            BenchmarkId::new(format!("{n}_bw{half_bandwidth}"), n),
            &matrix,
            |b, matrix| {
                b.iter(|| {
                    let ilu = spilu(black_box(matrix), IluOptions::default()).expect("spilu");
                    black_box(ilu.shape);
                });
            },
        );
    }

    group.finish();
}

fn bench_spsolve_laplacian(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_spsolve_laplacian");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));

    for &side in &[40usize, 70] {
        let matrix = make_laplacian_2d(side);
        let n = side * side;
        let rhs: Vec<f64> = (0..n).map(|i| 1.0 + (i % 13) as f64 * 0.5).collect();
        group.bench_with_input(
            BenchmarkId::new("spsolve", n),
            &(matrix, rhs),
            |b, (a, rhs)| {
                b.iter(|| {
                    let result = spsolve(black_box(a), black_box(rhs), SolveOptions::default())
                        .expect("spsolve laplacian");
                    black_box(result.solution.len());
                });
            },
        );
    }

    group.finish();
}

fn bench_random_tiny_density(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_random_tiny_density");
    for &(n, density) in TINY_DENSITY_CASES {
        group.bench_with_input(
            BenchmarkId::new(format!("{n}x{n}_d{density:.0e}"), n),
            &(n, density),
            |b, &(n, density)| {
                b.iter(|| {
                    let coo = random(
                        Shape2D::new(black_box(n), black_box(n)),
                        black_box(density),
                        black_box(SEED),
                    )
                    .expect("random tiny density");
                    black_box(coo.nnz());
                });
            },
        );
    }
    group.finish();
}

fn bench_coo_to_csr(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_coo_to_csr");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));

    for (n, nnz) in [(100_000usize, 2_000_000usize), (20_000, 2_000_000)] {
        // Deterministic unsorted triplets (LCG), with some duplicates.
        let mut st = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (st >> 33) as usize
        };
        let mut rows = Vec::with_capacity(nnz);
        let mut cols = Vec::with_capacity(nnz);
        let mut data = Vec::with_capacity(nnz);
        for _ in 0..nnz {
            rows.push(next() % n);
            cols.push(next() % n);
            data.push((next() % 1000) as f64 + 1.0);
        }
        let coo =
            CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false).expect("coo");
        group.bench_with_input(
            BenchmarkId::new(format!("n{n}_nnz{nnz}"), nnz),
            &coo,
            |bi, coo| {
                bi.iter(|| {
                    let csr = FormatConvertible::to_csr(black_box(coo)).expect("to_csr");
                    black_box(csr.nnz());
                });
            },
        );
    }
    group.finish();
}

fn bench_coo_sum_duplicates(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_coo_sum_duplicates");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));

    for (n, nnz) in [(100_000usize, 2_000_000usize), (20_000, 2_000_000)] {
        let mut st = 0xC0FF_EE00_1234_5678u64 ^ (n as u64);
        let mut next = || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (st >> 33) as usize
        };
        let rows: Vec<usize> = (0..nnz).map(|_| next() % n).collect();
        let cols: Vec<usize> = (0..nnz).map(|_| next() % n).collect();
        let data: Vec<f64> = (0..nnz).map(|_| (next() % 1000) as f64 + 1.0).collect();
        let fixture = (rows, cols, data);
        for (label, disable_radix) in [("current", false), ("legacy_original", true)] {
            group.bench_with_input(
                BenchmarkId::new(format!("{label}_n{n}_nnz{nnz}"), nnz),
                &fixture,
                |bi, (rows, cols, data)| {
                    bi.iter(|| {
                        COO_SUM_DUPLICATES_RADIX_DISABLE.store(disable_radix, Ordering::Relaxed);
                        let coo = CooMatrix::from_triplets(
                            Shape2D::new(n, n),
                            black_box(data).clone(),
                            black_box(rows).clone(),
                            black_box(cols).clone(),
                            true,
                        )
                        .expect("coo");
                        black_box(coo.nnz());
                    });
                },
            );
        }
        COO_SUM_DUPLICATES_RADIX_DISABLE.store(false, Ordering::Relaxed);
    }
    group.finish();
}

fn bench_coo_to_csc(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_coo_to_csc");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));

    for (n, nnz) in [(100_000usize, 2_000_000usize), (20_000, 2_000_000)] {
        let mut st = 0x0FE1_DEAD_5678_1234u64 ^ (n as u64);
        let mut next = || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (st >> 33) as usize
        };
        let mut rows = Vec::with_capacity(nnz);
        let mut cols = Vec::with_capacity(nnz);
        let mut data = Vec::with_capacity(nnz);
        for _ in 0..nnz {
            rows.push(next() % n);
            cols.push(next() % n);
            data.push((next() % 1000) as f64 + 1.0);
        }
        let coo =
            CooMatrix::from_triplets(Shape2D::new(n, n), data, rows, cols, false).expect("coo");
        group.bench_with_input(
            BenchmarkId::new(format!("n{n}_nnz{nnz}"), nnz),
            &coo,
            |bi, coo| {
                bi.iter(|| {
                    let csc = FormatConvertible::to_csc(black_box(coo)).expect("to_csc");
                    black_box(csc.nnz());
                });
            },
        );
    }
    group.finish();
}

fn bench_block_diag(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_block_diag");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));

    for (nblocks, bs, dens) in [(2000usize, 40usize, 0.1f64), (500, 120, 0.05)] {
        let blocks: Vec<CsrMatrix> = (0..nblocks)
            .map(|i| make_random_rect_csr(bs, bs, dens, SEED ^ (i as u64).wrapping_mul(0x9E37)))
            .collect();
        let refs: Vec<&CsrMatrix> = blocks.iter().collect();
        let nnz: usize = blocks.iter().map(CsrMatrix::nnz).sum();
        group.bench_with_input(
            BenchmarkId::new(format!("nblocks{nblocks}_bs{bs}_nnz{nnz}"), nnz),
            &refs,
            |bi, refs| {
                bi.iter(|| {
                    let out = block_diag(black_box(refs)).expect("block_diag");
                    black_box(out.nnz());
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_coo_sum_duplicates,
    bench_coo_to_csr,
    bench_coo_to_csc,
    bench_block_diag,
    bench_csr_construction,
    bench_spmv,
    bench_format_conversion,
    bench_arithmetic,
    bench_eye,
    bench_diags,
    bench_spmm,
    bench_kron,
    bench_spilu,
    bench_spsolve_laplacian,
    bench_random_tiny_density
);
criterion_main!(benches);
