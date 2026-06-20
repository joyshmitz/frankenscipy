//! Same-process A/B for parallel CSR SpMV (the inner kernel of every Krylov
//! solver, eigsh/eigs/svds, onenormest). Each output row is an independent dot
//! product, so parallelizing across row chunks is byte-identical. This settles
//! whether SpMV is bandwidth-bound (<2x) or scales on this 64-core box.
//! Run: `cargo run --profile release-perf -p fsci-sparse --bin perf_csr_matvec`.

use std::hint::black_box;
use std::time::Instant;

use fsci_sparse::{CsrMatrix, FormatConvertible, Shape2D, random, spmv_csr};

fn legacy_public_spmv_csr(a: &CsrMatrix, x: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0; a.shape().rows];
    for (row, output) in result.iter_mut().enumerate().take(a.shape().rows) {
        for idx in a.indptr()[row]..a.indptr()[row + 1] {
            *output += a.data()[idx] * x[a.indices()[idx]];
        }
    }
    result
}

fn public_spmv_ab() {
    println!("public spmv_csr legacy-row-sweep vs current; bit identity + timing\n");
    for &(n, density, reps) in &[
        (100usize, 0.05f64, 200_000u32),
        (1_000, 0.01, 50_000),
        (10_000, 0.001, 5_000),
    ] {
        let a = random(Shape2D::new(n, n), density, 0xA11CE ^ n as u64)
            .unwrap()
            .to_csr()
            .unwrap();
        let x: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01 - 0.5).collect();

        let legacy = legacy_public_spmv_csr(&a, &x);
        let current = spmv_csr(&a, &x).unwrap();
        let identical = legacy
            .iter()
            .zip(&current)
            .all(|(a, b)| a.to_bits() == b.to_bits());

        let mut acc = 0.0;
        let _ = legacy_public_spmv_csr(&a, &x);
        let _ = spmv_csr(&a, &x).unwrap();
        let t0 = Instant::now();
        for _ in 0..reps {
            acc += legacy_public_spmv_csr(black_box(&a), black_box(&x))[0];
        }
        let legacy_time = t0.elapsed();
        let t1 = Instant::now();
        for _ in 0..reps {
            acc += spmv_csr(black_box(&a), black_box(&x)).unwrap()[0];
        }
        let current_time = t1.elapsed();
        println!(
            "n={n:>5} nnz={:>7} reps={reps:>6} identical={identical} legacy={:>9.3?} current={:>9.3?} ratio={:>5.2}x (acc={acc:.3})",
            a.nnz(),
            legacy_time / reps,
            current_time / reps,
            legacy_time.as_secs_f64() / current_time.as_secs_f64(),
        );
    }
}

fn serial(a: &CsrMatrix, x: &[f64]) -> Vec<f64> {
    let n = a.shape().rows;
    let (indptr, indices, data) = (a.indptr(), a.indices(), a.data());
    let mut out = vec![0.0; n];
    for i in 0..n {
        let mut s = 0.0;
        for idx in indptr[i]..indptr[i + 1] {
            s += data[idx] * x[indices[idx]];
        }
        out[i] = s;
    }
    out
}

fn parallel(a: &CsrMatrix, x: &[f64]) -> Vec<f64> {
    let n = a.shape().rows;
    let (indptr, indices, data) = (a.indptr(), a.indices(), a.data());
    let nnz = data.len();
    // Scale workers by WORK (nnz), ~128K nnz/thread, so medium matrices don't
    // over-spawn; serial below ~256K nnz where spawn cost isn't amortized.
    let nthreads = if nnz < 1 << 18 || n < 256 {
        1
    } else {
        std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(1)
            .min(nnz / (1 << 17))
            .max(1)
    };
    if nthreads <= 1 {
        return serial(a, x);
    }
    let mut out = vec![0.0; n];
    let chunk = n.div_ceil(nthreads);
    std::thread::scope(|scope| {
        for (t, slot) in out.chunks_mut(chunk).enumerate() {
            let base = t * chunk;
            scope.spawn(move || {
                for (r, o) in slot.iter_mut().enumerate() {
                    let i = base + r;
                    let mut s = 0.0;
                    for idx in indptr[i]..indptr[i + 1] {
                        s += data[idx] * x[indices[idx]];
                    }
                    *o = s;
                }
            });
        }
    });
    out
}

fn main() {
    if std::env::var_os("FSCI_PUBLIC_SPMV_AB").is_some() {
        public_spmv_ab();
        return;
    }

    println!("nproc check below; A/B byte-identity + timing\n");
    for &(n, density) in &[
        (20_000usize, 0.0005f64),
        (50_000, 0.0004),
        (200_000, 0.0001),
        (500_000, 0.00004),
    ] {
        let a = random(Shape2D::new(n, n), density, 0xC0FFEE ^ n as u64)
            .unwrap()
            .to_csr()
            .unwrap();
        let nnz = a.data().len();
        let mut x = vec![0.0; n];
        let mut s = 0x1234_5678u64;
        for xi in x.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            *xi = (s >> 11) as f64 / (1u64 << 53) as f64;
        }

        // Byte-identity
        let ser = serial(&a, &x);
        let par = parallel(&a, &x);
        let identical = ser
            .iter()
            .zip(&par)
            .all(|(a, b)| a.to_bits() == b.to_bits());

        let reps = 200;
        let _ = parallel(&a, &x);
        let t0 = Instant::now();
        let mut acc = 0.0;
        for _ in 0..reps {
            acc += serial(black_box(&a), black_box(&x))[0];
        }
        let ts = t0.elapsed();
        let t1 = Instant::now();
        for _ in 0..reps {
            acc += parallel(black_box(&a), black_box(&x))[0];
        }
        let tp = t1.elapsed();
        println!(
            "n={n:>7} nnz={nnz:>9} identical={identical}  serial={:>9.3?}  parallel={:>9.3?}  ratio={:>5.1}x (acc={acc:.3})",
            ts / reps,
            tp / reps,
            ts.as_secs_f64() / tp.as_secs_f64()
        );
    }
    println!("\n--- end-to-end (lib eigsh, uses lib csr_matvec) ---");
    end_to_end();
}

// End-to-end: eigsh (power iteration) does many matvecs; the lib csr_matvec
// change speeds them. Run this binary after stashing the lib change to compare.
fn end_to_end() {
    use fsci_sparse::{EigsOptions, eigsh};
    for &(n, density) in &[(100_000usize, 0.00015f64), (300_000, 0.00005)] {
        // Symmetric-ish: A + A^T would be ideal but random is fine for matvec timing.
        let a = random(Shape2D::new(n, n), density, 0xBEEF ^ n as u64)
            .unwrap()
            .to_csr()
            .unwrap();
        let t0 = Instant::now();
        let r = eigsh(
            &a,
            2,
            EigsOptions {
                tol: 1e-8,
                max_iter: 80,
            },
        );
        let dt = t0.elapsed();
        println!(
            "eigsh n={n} nnz={} -> {:?} in {:?}",
            a.data().len(),
            r.is_ok(),
            dt
        );
    }
}
