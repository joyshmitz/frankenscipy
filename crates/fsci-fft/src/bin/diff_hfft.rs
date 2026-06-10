use fsci_fft::{hfft, hfft2, ihfft, ihfft2, Complex64, FftOptions, Normalization};
fn opts(n: Normalization) -> FftOptions {
    FftOptions::default().with_normalization(n)
}
fn dumpr(name: &str, nl: &str, v: &[f64]) {
    for (i, &x) in v.iter().enumerate() {
        println!("{name},{nl},{i},re,{x:.17e}");
    }
}
fn dumpc(name: &str, nl: &str, v: &[Complex64]) {
    for (i, c) in v.iter().enumerate() {
        println!("{name},{nl},{i},re,{:.17e}", c.0);
        println!("{name},{nl},{i},im,{:.17e}", c.1);
    }
}
fn main() {
    // Hermitian-ish complex spectrum for hfft (len 4 -> default n=6).
    let spec: Vec<Complex64> = vec![(3.0, 0.0), (0.7, -1.2), (-0.4, 0.9), (1.1, 0.0)];
    // Real signal for ihfft (len 6).
    let sig: Vec<f64> = (0..6)
        .map(|k| {
            let t = k as f64;
            (0.5 * t).cos() + 0.3 * t - 0.05 * t * t + 0.8
        })
        .collect();

    for (nl, nrm) in [
        ("backward", Normalization::Backward),
        ("ortho", Normalization::Ortho),
        ("forward", Normalization::Forward),
    ] {
        let o = opts(nrm);
        // default n
        if let Ok(v) = hfft(&spec, None, &o) {
            dumpr("hfft", nl, &v);
        }
        // explicit n = 7 (odd)
        if let Ok(v) = hfft(&spec, Some(7), &o) {
            dumpr("hfft_n7", nl, &v);
        }
        if let Ok(v) = ihfft(&sig, None, &o) {
            dumpc("ihfft", nl, &v);
        }
        if let Ok(v) = ihfft(&sig, Some(5), &o) {
            dumpc("ihfft_n5", nl, &v);
        }
        // 2-D Hermitian. hfft2 input shape (3,3) -> output (3,4) [last axis 2*(3-1)].
        let spec2: Vec<Complex64> = (0..9)
            .map(|k| {
                let t = k as f64;
                ((0.4 * t).cos() + 0.5, (0.3 * t).sin() * 0.6)
            })
            .collect();
        if let Ok(v) = hfft2(&spec2, (3, 4), &o) {
            dumpr("hfft2", nl, &v);
        }
        // ihfft2 real input shape (3,4).
        let sig2: Vec<f64> = (0..12)
            .map(|k| {
                let t = k as f64;
                (0.5 * t).cos() + 0.2 * t - 0.03 * t * t + 0.9
            })
            .collect();
        if let Ok(v) = ihfft2(&sig2, (3, 4), &o) {
            dumpc("ihfft2", nl, &v);
        }
    }
}
