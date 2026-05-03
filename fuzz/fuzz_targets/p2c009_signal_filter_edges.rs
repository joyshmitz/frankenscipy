#![no_main]

use arbitrary::Arbitrary;
use fsci_signal::{SosSection, lfilter, lfilter_zi, sosfilt, sosfilt_zi, sosfiltfilt};
use libfuzzer_sys::fuzz_target;

const MAX_COEFF: usize = 8;
const MAX_SIGNAL: usize = 96;
const MAX_SOS: usize = 4;

#[derive(Debug, Arbitrary)]
struct SignalFilterInput {
    b_len: u8,
    a_len: u8,
    x_len: u8,
    sos_len: u8,
    b_raw: Vec<f64>,
    a_raw: Vec<f64>,
    x_raw: Vec<f64>,
    zi_raw: Vec<f64>,
    sos_raw: Vec<f64>,
    use_bad_zi: bool,
}

fn finite(value: f64, bound: f64) -> f64 {
    if value.is_finite() {
        value.clamp(-bound, bound)
    } else {
        0.0
    }
}

fn coeffs(raw: &[f64], requested: u8, default_first: f64) -> Vec<f64> {
    let len = usize::from(requested) % (MAX_COEFF + 1);
    let mut values: Vec<f64> = (0..len)
        .map(|idx| finite(raw.get(idx).copied().unwrap_or(0.0), 2.0))
        .collect();
    if let Some(first) = values.first_mut()
        && first.abs() < 1.0e-9
    {
        *first = default_first;
    }
    values
}

fn signal(raw: &[f64], requested: u8) -> Vec<f64> {
    let len = usize::from(requested) % (MAX_SIGNAL + 1);
    (0..len)
        .map(|idx| finite(raw.get(idx).copied().unwrap_or(0.0), 8.0))
        .collect()
}

fn zi(raw: &[f64], len: usize, bad: bool) -> Vec<f64> {
    let target = if bad { len.saturating_add(1) } else { len };
    (0..target)
        .map(|idx| finite(raw.get(idx).copied().unwrap_or(0.0), 2.0))
        .collect()
}

fn sos_sections(input: &SignalFilterInput) -> Vec<SosSection> {
    let count = usize::from(input.sos_len) % (MAX_SOS + 1);
    (0..count)
        .map(|section| {
            let base = section * 6;
            let b0 = finite(input.sos_raw.get(base).copied().unwrap_or(1.0), 1.0);
            let b1 = finite(input.sos_raw.get(base + 1).copied().unwrap_or(0.0), 1.0);
            let b2 = finite(input.sos_raw.get(base + 2).copied().unwrap_or(0.0), 1.0);
            let mut a0 = finite(input.sos_raw.get(base + 3).copied().unwrap_or(1.0), 1.0);
            let a1 = finite(input.sos_raw.get(base + 4).copied().unwrap_or(0.0), 0.75);
            let a2 = finite(input.sos_raw.get(base + 5).copied().unwrap_or(0.0), 0.75);
            if a0.abs() < 1.0e-9 {
                a0 = 1.0;
            }
            [b0, b1, b2, a0, a1, a2]
        })
        .collect()
}

fn assert_output_shape_and_finiteness(name: &str, output: &[f64], expected_len: usize) {
    assert_eq!(output.len(), expected_len, "{name}: output length mismatch");
    for (idx, value) in output.iter().enumerate() {
        assert!(
            value.is_finite(),
            "{name}: non-finite output at index {idx}: {value}"
        );
    }
}

fuzz_target!(|input: SignalFilterInput| {
    let b = coeffs(&input.b_raw, input.b_len, 1.0);
    let a = coeffs(&input.a_raw, input.a_len, 1.0);
    let x = signal(&input.x_raw, input.x_len);

    let _ = lfilter_zi(&b, &a);
    let nfilt = b.len().max(a.len()).saturating_sub(1);
    let zi_values = zi(&input.zi_raw, nfilt, input.use_bad_zi);
    if let Ok(output) = lfilter(&b, &a, &x, Some(&zi_values)) {
        assert_output_shape_and_finiteness("lfilter", &output, x.len());
    }
    if let Ok(output) = lfilter(&b, &a, &x, None) {
        assert_output_shape_and_finiteness("lfilter(no zi)", &output, x.len());
    }

    let sos = sos_sections(&input);
    let _ = sosfilt_zi(&sos);
    if let Ok(output) = sosfilt(&sos, &x) {
        assert_output_shape_and_finiteness("sosfilt", &output, x.len());
    }
    if x.len() >= 3
        && let Ok(output) = sosfiltfilt(&sos, &x)
    {
        assert_output_shape_and_finiteness("sosfiltfilt", &output, x.len());
    }
});
