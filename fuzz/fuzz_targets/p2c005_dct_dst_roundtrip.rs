#![no_main]

use arbitrary::Arbitrary;
use fsci_fft::{FftOptions, Normalization, dct, dst_ii, dst_iii, idct};
use libfuzzer_sys::fuzz_target;

// DCT/DST roundtrip metamorphic oracle:
//   For any real input x:
//     idct(dct(x, norm), norm) ≈ x
//     dst_iii(dst_ii(x, norm), norm) ≈ x  (for Backward, scaled by 2N)
//
// Catches regressions like [frankenscipy-v0vm5] where idct hardcoded
// 1/(2N) and broke under Ortho/Forward normalizations.

const MAX_LEN: usize = 256;
const REL_TOL: f64 = 1e-9;
const ABS_TOL: f64 = 1e-12;

#[derive(Debug, Arbitrary)]
struct DctDstInput {
    samples: Vec<f64>,
    norm_variant: u8,
    transform_variant: u8,
}

fn sanitize(x: f64) -> f64 {
    if x.is_finite() {
        x.clamp(-1e6, 1e6)
    } else {
        0.0
    }
}

fn close_enough(a: f64, b: f64) -> bool {
    if !a.is_finite() || !b.is_finite() {
        return true;
    }
    (a - b).abs() <= ABS_TOL + REL_TOL * a.abs().max(b.abs())
}

fuzz_target!(|input: DctDstInput| {
    let n = input.samples.len().min(MAX_LEN);
    if n == 0 {
        return;
    }
    let original: Vec<f64> = input
        .samples
        .iter()
        .take(n)
        .map(|&v| sanitize(v))
        .collect();

    let norm = match input.norm_variant % 3 {
        0 => Normalization::Backward,
        1 => Normalization::Forward,
        _ => Normalization::Ortho,
    };
    let opts = FftOptions {
        normalization: norm,
        ..FftOptions::default()
    };

    match input.transform_variant % 2 {
        0 => {
            // DCT-II → DCT-III roundtrip via dct/idct
            let Ok(forward) = dct(&original, &opts) else {
                return;
            };
            let Ok(recovered) = idct(&forward, &opts) else {
                return;
            };
            if recovered.len() != n {
                panic!("idct(dct(x)) length {} != {}", recovered.len(), n);
            }
            for (i, (orig, rec)) in original.iter().zip(recovered.iter()).enumerate() {
                if !close_enough(*orig, *rec) {
                    panic!(
                        "DCT roundtrip failed at index {i}: original={} recovered={} \
                         len={n} norm={norm:?}",
                        orig, rec
                    );
                }
            }
        }
        _ => {
            // DST-II → DST-III roundtrip. dst_iii(dst_ii(x)) recovers
            // x scaled by the convention-dependent factor:
            //   Backward: 2N·x
            //   Ortho:    1·x
            //   Forward:  (1/(2N))·x
            let Ok(forward) = dst_ii(&original, &opts) else {
                return;
            };
            let Ok(recovered) = dst_iii(&forward, &opts) else {
                return;
            };
            if recovered.len() != n {
                panic!("dst_iii(dst_ii(x)) length {} != {}", recovered.len(), n);
            }
            let nf = n as f64;
            let scale = match norm {
                Normalization::Backward => 2.0 * nf,
                Normalization::Ortho => 1.0,
                Normalization::Forward => 1.0 / (2.0 * nf),
            };
            for (i, (orig, rec)) in original.iter().zip(recovered.iter()).enumerate() {
                let expected = orig * scale;
                if !close_enough(expected, *rec) {
                    panic!(
                        "DST roundtrip failed at index {i}: original={} \
                         expected={} recovered={} len={n} norm={norm:?}",
                        orig, expected, rec
                    );
                }
            }
        }
    }
});
