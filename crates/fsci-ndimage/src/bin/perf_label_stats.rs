//! Same-process A/B for label-indexed measurements (`mean`/`sum`/...).
//!
//! The shared core `measurement_label_groups` used to bucket each of the N
//! input elements with an O(K) linear `position` scan over the K requested
//! labels — O(N*K). It now builds a label->position map once and buckets in
//! O(1) (O(N+K)). This bin reconstructs the old linear-scan grouping verbatim,
//! proves byte-identical means, and times both against the new public `mean`.
//! Run: `cargo run --release -p fsci-ndimage --bin perf_label_stats`.

use fsci_ndimage::{NdArray, mean};
use std::time::Instant;

struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
}

/// Verbatim old behavior: per element, linear `position` scan over `index`.
fn old_mean_by_label(input: &NdArray, labels: &NdArray, index: &[usize]) -> Vec<f64> {
    let mut groups: Vec<Vec<f64>> = vec![Vec::new(); index.len()];
    for (&value, &label_value) in input.data.iter().zip(&labels.data) {
        if let Some(pos) = index
            .iter()
            .position(|&wanted| label_value == wanted as f64)
        {
            groups[pos].push(value);
        }
    }
    groups
        .iter()
        .map(|g| {
            if g.is_empty() {
                0.0
            } else {
                g.iter().sum::<f64>() / g.len() as f64
            }
        })
        .collect()
}

fn main() {
    for &(side, k) in &[(256usize, 512usize), (512, 1024), (512, 2048), (768, 4096)] {
        let n = side * side;
        let mut r = Lcg(0xA11CE ^ (side as u64) ^ ((k as u64) << 20));
        // Random label in 1..=k for every cell; random values.
        let labels_data: Vec<f64> = (0..n)
            .map(|_| (1 + (r.next_u64() as usize) % k) as f64)
            .collect();
        let values: Vec<f64> = (0..n).map(|_| (r.next_u64() >> 11) as f64 * 1e-9).collect();
        let input = NdArray::new(values, vec![side, side]).unwrap();
        let labels = NdArray::new(labels_data, vec![side, side]).unwrap();
        let index: Vec<usize> = (1..=k).collect();

        // Correctness: byte-identical to the old linear-scan grouping.
        let new_means = mean(&input, Some(&labels), Some(&index)).unwrap();
        let old_means = old_mean_by_label(&input, &labels, &index);
        let mismatches = new_means
            .iter()
            .zip(&old_means)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();

        let t0 = Instant::now();
        let mut acc = 0.0f64;
        for _ in 0..3 {
            acc += old_mean_by_label(&input, &labels, &index)[0];
        }
        let old_t = t0.elapsed() / 3;

        let t1 = Instant::now();
        for _ in 0..3 {
            acc += mean(&input, Some(&labels), Some(&index)).unwrap()[0];
        }
        let new_t = t1.elapsed() / 3;

        let ratio = old_t.as_secs_f64() / new_t.as_secs_f64();
        println!(
            "N={:>7} K={:>5}  old(O(N*K))={:>10.3?}  new(O(N+K))={:>10.3?}  ratio={ratio:>7.1}x  mism={mismatches}  (acc={acc:.3})",
            n, k, old_t, new_t
        );
    }
}
