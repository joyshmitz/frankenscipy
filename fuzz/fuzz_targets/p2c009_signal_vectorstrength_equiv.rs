#![no_main]

use arbitrary::Arbitrary;
use fsci_signal::vectorstrength;
use libfuzzer_sys::fuzz_target;

// vectorstrength equivalence oracle for [frankenscipy-dw9fd].
//
// After the 76714d4 perf fix (single-pass loop with sin_cos),
// verify random inputs:
//   1. r matches a naive dual-pass reference within rel < 1e-9.
//   2. pvalue matches the naive computation similarly.
//   3. 0 ≤ r ≤ 1  (Rayleigh statistic bound).
//   4. 0 ≤ pvalue ≤ 1.

const BOUND: f64 = 1.0e6;
const MIN_N: usize = 1;
const MAX_N: usize = 128;

#[derive(Debug, Arbitrary)]
struct VectorStrengthInput {
    events: Vec<f64>,
    period: f64,
}

fn sanitize_event(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(-BOUND, BOUND)
    } else {
        0.0
    }
}

fn sanitize_period(value: f64) -> f64 {
    if value.is_finite() && value > 0.0 {
        value.clamp(1e-6, BOUND)
    } else {
        1.0
    }
}

fn naive_vectorstrength(events: &[f64], period: f64) -> (f64, f64) {
    if events.is_empty() || period <= 0.0 || !period.is_finite() {
        return (0.0, 1.0);
    }
    let two_pi = 2.0 * std::f64::consts::PI;
    let n = events.len() as f64;
    let sin_sum: f64 = events.iter().map(|&t| (two_pi * t / period).sin()).sum();
    let cos_sum: f64 = events.iter().map(|&t| (two_pi * t / period).cos()).sum();
    let r = (sin_sum * sin_sum + cos_sum * cos_sum).sqrt() / n;
    let p = (-n * r * r).exp().clamp(0.0, 1.0);
    (r, p)
}

fuzz_target!(|input: VectorStrengthInput| {
    let events: Vec<f64> = input
        .events
        .iter()
        .take(MAX_N)
        .copied()
        .map(sanitize_event)
        .collect();
    if events.len() < MIN_N {
        return;
    }
    let period = sanitize_period(input.period);

    let (r_opt, p_opt) = vectorstrength(&events, period);
    let (r_nav, p_nav) = naive_vectorstrength(&events, period);

    // Bounds.
    if !(r_opt.is_finite() && (0.0..=1.0).contains(&r_opt)) {
        panic!("r out of bounds: r={r_opt} for {} events", events.len());
    }
    if !(p_opt.is_finite() && (0.0..=1.0).contains(&p_opt)) {
        panic!("pvalue out of bounds: p={p_opt}");
    }

    // Equivalence (large tolerance because trig accuracy can drift slightly
    // between the two summation orderings).
    let r_scale = r_opt.abs().max(r_nav.abs()).max(1e-12);
    if (r_opt - r_nav).abs() > 1e-9 * r_scale {
        panic!(
            "r mismatch: opt={r_opt}, naive={r_nav} for events.len()={}, period={period}",
            events.len()
        );
    }
    let p_scale = p_opt.abs().max(p_nav.abs()).max(1e-12);
    if (p_opt - p_nav).abs() > 1e-9 * p_scale {
        panic!("pvalue mismatch: opt={p_opt}, naive={p_nav}");
    }
});
