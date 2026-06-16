// A/B for clarkson_woodruff_transform: the IMPLICIT CountSketch (one O(m·n) hashing pass)
// vs the naive EXPLICIT application — materialise the dense sketch matrix S (sketch×m,
// one ±1 per column) and compute S·A via matmul, O(sketch·m·n). Same sketch, so the
// outputs agree to rounding; the speedup is why scipy.linalg ships the implicit form.
use fsci_linalg::{clarkson_woodruff_transform, matmul};
use std::hint::black_box;
use std::time::Instant;

// Replay the same SplitMix64 stream the transform uses, to build S explicitly.
fn build_sketch_matrix(m: usize, sketch: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
    let mut next = || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    let mut s = vec![vec![0.0f64; m]; sketch];
    for col in 0..m {
        let bucket = (next() % sketch as u64) as usize;
        let sign = if next() & 1 == 0 { 1.0 } else { -1.0 };
        s[bucket][col] = sign;
    }
    s
}

fn main() {
    let m = 5000usize;
    let n = 50usize;
    let sketch = 1200usize;
    let seed = 7u64;
    let mut st: u64 = 0x243f_6a88_85a3_08d3;
    let mut rng = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64) / (1u64 << 53) as f64 - 0.5
    };
    let a: Vec<Vec<f64>> = (0..m).map(|_| (0..n).map(|_| rng()).collect()).collect();

    let implicit = clarkson_woodruff_transform(&a, sketch, seed).expect("cwt");
    let s = build_sketch_matrix(m, sketch, seed);
    let explicit = matmul(&s, &a).expect("matmul");
    let mut max_abs = 0.0f64;
    for (ri, re) in implicit.iter().zip(&explicit) {
        for (&x, &y) in ri.iter().zip(re) {
            max_abs = max_abs.max((x - y).abs());
        }
    }
    println!("max_abs_diff(implicit vs explicit)={max_abs:.3e}");

    let trials = 5;
    let mut ti = Vec::new();
    let mut te = Vec::new();
    for _ in 0..trials {
        let t = Instant::now();
        black_box(clarkson_woodruff_transform(&a, sketch, seed).unwrap());
        ti.push(t.elapsed().as_secs_f64());
        let t = Instant::now();
        let s = build_sketch_matrix(m, sketch, seed);
        black_box(matmul(&s, &a).unwrap());
        te.push(t.elapsed().as_secs_f64());
    }
    ti.sort_by(|a, b| a.partial_cmp(b).unwrap());
    te.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let imp = ti[trials / 2] * 1e3;
    let exp = te[trials / 2] * 1e3;
    println!(
        "explicit(form S + matmul) {exp:.2} ms | implicit CountSketch {imp:.2} ms | speedup {:.1}x  (m={m} n={n} sketch={sketch})",
        exp / imp
    );
}
