//! Same-process timing + bit-identity harness for the KDTree dual-tree queries
//! (count_neighbors, query_ball_tree, query_pairs).
//!
//! Each anchor's query is an independent tree descent; count_neighbors is an exact integer
//! sum, query_ball_tree stores each result at its own index, and query_pairs sorts the
//! collected set — all bit-identical when parallelized across anchors. Dumps digests and
//! times them. Run: `cargo run -p fsci-spatial --bin perf_kdtree_queries`.

use std::hint::black_box;
use std::time::Instant;

use fsci_spatial::KDTree;

fn points(n: usize, d: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            (0..d)
                .map(|_| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (s >> 11) as f64 / (1u64 << 53) as f64
                })
                .collect()
        })
        .collect()
}

fn digest_pairs(pairs: &[(usize, usize)]) -> u64 {
    pairs.iter().fold(1469598103934665603u64, |h, &(a, b)| {
        let h = (h ^ a as u64).wrapping_mul(1099511628211);
        (h ^ b as u64).wrapping_mul(1099511628211)
    })
}

fn digest_ball(res: &[Vec<usize>]) -> u64 {
    res.iter()
        .flat_map(|v| v.iter())
        .fold(1469598103934665603u64, |h, &x| {
            (h ^ x as u64).wrapping_mul(1099511628211)
        })
}

fn main() {
    let d = 3usize;
    let sizes = [5000usize, 15000, 40000];
    let r = 0.12;

    println!("===GOLDEN_PAYLOAD_BEGIN===");
    for &n in &sizes {
        let a = KDTree::new(&points(n, d, 1)).unwrap();
        let b = KDTree::new(&points(n / 2, d, 99)).unwrap();
        let cn = a.count_neighbors(&b, r).unwrap();
        let qbt = a.query_ball_tree(&b, r).unwrap();
        let qp = a.query_pairs(r).unwrap();
        println!(
            "n={n} count={cn} ball={:016x} pairs_len={} pairs={:016x}",
            digest_ball(&qbt),
            qp.len(),
            digest_pairs(&qp)
        );
    }
    println!("===GOLDEN_PAYLOAD_END===");

    for &n in &sizes {
        let a = KDTree::new(&points(n, d, 1)).unwrap();
        let b = KDTree::new(&points(n / 2, d, 99)).unwrap();
        let reps = 5;
        macro_rules! time {
            ($name:expr, $body:expr) => {{
                let t0 = Instant::now();
                let mut acc = 0usize;
                for _ in 0..reps {
                    acc = acc.wrapping_add($body);
                }
                println!(
                    "n={n:>6} {:<12} {:>9.3?}/call (acc={acc})",
                    $name,
                    t0.elapsed() / reps
                );
            }};
        }
        time!(
            "count_neigh",
            black_box(&a).count_neighbors(black_box(&b), r).unwrap()
        );
        time!(
            "ball_tree",
            black_box(&a)
                .query_ball_tree(black_box(&b), r)
                .unwrap()
                .len()
        );
        time!("query_pairs", black_box(&a).query_pairs(r).unwrap().len());
    }
}
