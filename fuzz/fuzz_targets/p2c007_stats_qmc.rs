#![no_main]

use arbitrary::Arbitrary;
use fsci_stats::qmc::{HaltonSampler, LatinHypercubeSampler, SobolSampler};
use libfuzzer_sys::fuzz_target;

// QMC sampler robustness oracle:
//   For HaltonSampler, SobolSampler (≤2-D), and LatinHypercubeSampler:
//     1. sample(n) returns exactly n × dim values
//     2. all values are finite and in [0, 1)
//     3. determinism — two independently-constructed samplers with the
//        same configuration produce bit-identical sequences
//     4. for samplers with skip(): post-skip sample matches a slice of
//        a single longer sample
//
// Catches regressions in radical_inverse, the prime list, the Sobol
// direction numbers, the LHS scrambling, or any per-call state mutation.
//
// Resolves frankenscipy-tts8f.

const MAX_DIM: usize = 8;
const MAX_N: usize = 64;

#[derive(Debug, Arbitrary)]
struct QmcInput {
    sampler_choice: u8,
    dim: u8,
    n: u8,
    seed: u64,
}

fn assert_in_unit_interval(name: &str, values: &[f64], expected_len: usize) {
    assert_eq!(
        values.len(),
        expected_len,
        "{name}: length {} != expected {expected_len}",
        values.len()
    );
    for (i, &v) in values.iter().enumerate() {
        assert!(
            v.is_finite(),
            "{name}: non-finite value at index {i}: {v}"
        );
        assert!(
            (0.0..1.0).contains(&v),
            "{name}: value {v} at index {i} not in [0, 1)"
        );
    }
}

fuzz_target!(|input: QmcInput| {
    let dim = 1 + (input.dim as usize % MAX_DIM);
    let n = 1 + (input.n as usize % MAX_N);

    match input.sampler_choice % 3 {
        0 => {
            // HaltonSampler — supports any dim ≥ 1
            let Ok(mut a) = HaltonSampler::new(dim) else {
                return;
            };
            let Ok(mut b) = HaltonSampler::new(dim) else {
                return;
            };
            let out_a = a.sample(n);
            let out_b = b.sample(n);
            assert_in_unit_interval("halton.sample", &out_a, n * dim);
            assert_eq!(out_a, out_b, "halton: same seed must yield same sequence");

            // skip(k) + sample(n) ≡ sample(k+n)[k..]
            let k = (input.seed as usize % 8) + 1;
            let Ok(mut c) = HaltonSampler::new(dim) else {
                return;
            };
            c.skip(k as u64);
            let post_skip = c.sample(n);
            let Ok(mut d) = HaltonSampler::new(dim) else {
                return;
            };
            let combined = d.sample(k + n);
            assert_eq!(
                post_skip,
                combined[k * dim..],
                "halton: skip({k})+sample({n}) != sample({}, k..k+n)",
                k + n
            );
        }
        1 => {
            // SobolSampler — only supports dim 1..=2
            let sobol_dim = 1 + (input.dim as usize % 2);
            let Ok(mut a) = SobolSampler::new(sobol_dim) else {
                return;
            };
            let Ok(mut b) = SobolSampler::new(sobol_dim) else {
                return;
            };
            let out_a = a.sample(n);
            let out_b = b.sample(n);
            assert_in_unit_interval("sobol.sample", &out_a, n * sobol_dim);
            assert_eq!(out_a, out_b, "sobol: deterministic on same dim");
        }
        _ => {
            // LatinHypercubeSampler — supports any dim ≥ 1, takes seed
            let seed = input.seed;
            let Ok(mut a) = LatinHypercubeSampler::new(dim, seed) else {
                return;
            };
            let Ok(mut b) = LatinHypercubeSampler::new(dim, seed) else {
                return;
            };
            let out_a = a.sample(n);
            let out_b = b.sample(n);
            assert_in_unit_interval("lhs.sample", &out_a, n * dim);
            assert_eq!(out_a, out_b, "lhs: same seed must yield same sequence");
        }
    }
});
