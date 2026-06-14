// A/B for nystroem_spectral_embedding (data-based, O(n·m) — never forms the n×n affinity) vs
// the full path: build the dense n×n RBF affinity, then spectral_embedding (randomized eigh).
// Both produce an n×k Laplacian-eigenmap embedding that separates the blobs; the speedup is
// the wall-clock ratio (the win is the O(n²)→O(n·m) drop from skipping the dense affinity).
use fsci_cluster::{kmeans, nystroem_spectral_embedding, spectral_embedding};
use std::hint::black_box;
use std::time::Instant;

fn purity(labels: &[usize], truth: &[usize], k: usize) -> f64 {
    let n = labels.len();
    let mut correct = 0usize;
    for pred in 0..k {
        let mut counts = vec![0usize; k];
        for i in 0..n {
            if labels[i] == pred {
                counts[truth[i]] += 1;
            }
        }
        correct += counts.iter().copied().max().unwrap_or(0);
    }
    correct as f64 / n as f64
}

fn build_aff(pts: &[Vec<f64>], gamma: f64) -> Vec<Vec<f64>> {
    let n = pts.len();
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    let d2: f64 = pts[i].iter().zip(&pts[j]).map(|(&a, &b)| (a - b) * (a - b)).sum();
                    (-gamma * d2).exp()
                })
                .collect()
        })
        .collect()
}

fn main() {
    let k = 4usize;
    let per = 700usize;
    let n = k * per;
    let m = 48usize;
    let gamma = 0.5f64;
    let mut st: u64 = 0x243f_6a88_85a3_08d3;
    let mut rng = || {
        st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((st >> 11) as f64) / (1u64 << 53) as f64 - 0.5
    };
    let mut pts = Vec::new();
    let mut truth = Vec::new();
    for c in 0..k {
        for _ in 0..per {
            pts.push(vec![15.0 * c as f64 + rng(), rng()]);
            truth.push(c);
        }
    }

    let nys = nystroem_spectral_embedding(&pts, k, m, gamma, 7).expect("nys-embed");
    let nys_labels = kmeans(&nys.embedding, k, 100, 7).expect("km").labels;
    println!("nystroem_spectral_embedding purity={:.3}  (n={n} m={m})", purity(&nys_labels, &truth, k));

    let trials = 3;
    let mut tn = Vec::new();
    let mut tf = Vec::new();
    for _ in 0..trials {
        let t = Instant::now();
        black_box(nystroem_spectral_embedding(&pts, k, m, gamma, 7).unwrap());
        tn.push(t.elapsed().as_secs_f64());
        let t = Instant::now();
        let aff = build_aff(&pts, gamma);
        black_box(spectral_embedding(&aff, k, 7).unwrap());
        tf.push(t.elapsed().as_secs_f64());
    }
    tn.sort_by(|a, b| a.partial_cmp(b).unwrap());
    tf.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n_ms = tn[trials / 2] * 1e3;
    let f_ms = tf[trials / 2] * 1e3;
    println!("full affinity+spectral_embedding {f_ms:.2} ms | nystroem_spectral_embedding {n_ms:.2} ms | speedup {:.2}x  (n={n} m={m} k={k})", f_ms / n_ms);
}
