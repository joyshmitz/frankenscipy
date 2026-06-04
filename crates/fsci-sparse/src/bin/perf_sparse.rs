//! Profiling-only harness for sparse hot paths.
//!
//! NOT a product binary. It exists so RCH, hyperfine, and sha256 checks can
//! attach to deterministic sparse arithmetic scenarios.
//!
//! Usage:
//!   `perf_sparse add-csr <n> <density> <repeats>`
//!   `perf_sparse add-csr-golden [path]`

use std::fmt::Write as _;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use fsci_sparse::{CooMatrix, CsrMatrix, FormatConvertible, Shape2D, add_csr, diags, random};

const SEED: u64 = 0xBEEF_CAFE;

fn make_add_inputs(n: usize, density: f64) -> (CsrMatrix, CsrMatrix) {
    let shape = Shape2D::new(n, n);
    let lhs = random(shape, density, SEED)
        .expect("random lhs")
        .to_csr()
        .expect("lhs csr");
    let rhs = random(shape, density, SEED ^ 0x5EED_1234)
        .expect("random rhs")
        .to_csr()
        .expect("rhs csr");
    (lhs, rhs)
}

fn cancellation_inputs() -> (CsrMatrix, CsrMatrix) {
    let shape = Shape2D::new(3, 4);
    let lhs = CooMatrix::from_triplets(
        shape,
        vec![1.0, 2.0, -4.0, 5.0],
        vec![0, 1, 1, 2],
        vec![1, 0, 3, 2],
        false,
    )
    .expect("lhs coo")
    .to_csr()
    .expect("lhs csr");
    let rhs = CooMatrix::from_triplets(
        shape,
        vec![3.0, 4.0, -5.0, 6.0],
        vec![0, 1, 2, 2],
        vec![2, 3, 2, 3],
        false,
    )
    .expect("rhs coo")
    .to_csr()
    .expect("rhs csr");
    (lhs, rhs)
}

fn write_csr(output: &mut String, label: &str, matrix: &CsrMatrix) {
    let meta = matrix.canonical_meta();
    write!(
        output,
        "case={label} shape={}x{} nnz={} sorted={} deduplicated={} indptr=",
        matrix.shape().rows,
        matrix.shape().cols,
        matrix.nnz(),
        meta.sorted_indices,
        meta.deduplicated,
    )
    .expect("write header");
    for value in matrix.indptr() {
        write!(output, "{value},").expect("write indptr");
    }
    output.push_str(" indices=");
    for value in matrix.indices() {
        write!(output, "{value},").expect("write indices");
    }
    output.push_str(" data=");
    for value in matrix.data() {
        write!(output, "{:016x},", value.to_bits()).expect("write data");
    }
    output.push('\n');
}

fn add_csr_golden_text() -> String {
    let mut output = String::new();
    let cases = [(8usize, 0.25), (64, 0.05), (1024, 0.001)];
    for (n, density) in cases {
        let (lhs, rhs) = make_add_inputs(n, density);
        let sum = add_csr(&lhs, &rhs).expect("add csr");
        write_csr(&mut output, &format!("random-{n}-{density}"), &sum);
    }
    let (lhs, rhs) = cancellation_inputs();
    let sum = add_csr(&lhs, &rhs).expect("add csr cancellation");
    write_csr(&mut output, "cancellation", &sum);
    output
}

fn diags_golden_text() -> String {
    let mut output = String::new();

    let small = diags(
        &[
            vec![-1.0, -1.0, -1.0, -1.0, -1.0],
            vec![2.0; 6],
            vec![-1.0, -1.0, -1.0, -1.0, -1.0],
        ],
        &[-1, 0, 1],
        Some(Shape2D::new(6, 6)),
    )
    .expect("small tridiag");
    write_csr(&mut output, "diags-tridiag-6", &small);

    let rectangular = diags(
        &[vec![0.0, 3.0, -2.0], vec![4.0, 0.0]],
        &[1, -2],
        Some(Shape2D::new(4, 5)),
    )
    .expect("rectangular explicit-zero diags");
    write_csr(&mut output, "diags-rect-explicit-zero", &rectangular);

    let n = 10_000usize;
    let sub = vec![-1.0; n - 1];
    let main = vec![2.0; n];
    let sup = vec![-1.0; n - 1];
    let large =
        diags(&[sub, main, sup], &[-1, 0, 1], Some(Shape2D::new(n, n))).expect("large tridiag");
    write_csr(&mut output, "diags-tridiag-10000", &large);

    output
}

fn write_or_print_golden(output: String, path: Option<&str>) {
    if let Some(path) = path {
        let path = Path::new(path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create golden artifact parent");
        }
        std::fs::write(path, output).expect("write golden artifact");
    } else {
        print!("{output}");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("add-csr");
    if mode == "add-csr-golden" {
        write_or_print_golden(add_csr_golden_text(), args.get(2).map(String::as_str));
        return;
    }
    if mode == "diags-golden" {
        write_or_print_golden(diags_golden_text(), args.get(2).map(String::as_str));
        return;
    }
    if mode != "add-csr" {
        eprintln!("unknown mode: {mode}");
        std::process::exit(2);
    }

    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10_000);
    let density: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.001);
    let repeats: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(20);
    let (lhs, rhs) = make_add_inputs(n, density);

    let t0 = Instant::now();
    let mut checksum = 0.0_f64;
    for _ in 0..repeats {
        let sum = add_csr(black_box(&lhs), black_box(&rhs)).expect("add csr");
        checksum += sum.data().iter().sum::<f64>() + sum.nnz() as f64;
        black_box(&sum);
    }
    let elapsed = t0.elapsed();
    let total_ms = elapsed.as_secs_f64() * 1e3;
    let per_call_ms = total_ms / repeats as f64;
    println!(
        "{{\"mode\":\"{mode}\",\"n\":{n},\"density\":{density},\"repeats\":{repeats},\"total_ms\":{total_ms:.3},\"per_call_ms\":{per_call_ms:.6},\"checksum\":{checksum:.12e}}}",
    );
}
