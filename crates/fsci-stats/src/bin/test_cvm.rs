fn main() {
    // Reference values from scipy.stats._hypotests._cdf_cvm
    let cases = [
        (0.1_f64, 5_usize, 0.3968736863295094),
        (0.5, 5, 0.9644416400590435),
        (1.0, 5, 0.9991041957882961),
        (0.1, 10, 0.4060001239613562),
        (0.5, 10, 0.9623042112467172),
        (0.1, 14, 0.4086076775704552),
        (0.5, 50, 0.9605942681968561),
        (0.1, 100, 0.4142139178300182),
    ];
    let mut max_diff: f64 = 0.0;
    for (x, n, expected) in cases {
        let got = fsci_stats::cvm_cdf_n(x, n);
        let diff = (got - expected).abs();
        max_diff = max_diff.max(diff);
        println!("cvm_cdf({}, n={}) = {:.16}  scipy: {:.16}  diff {:.2e}", x, n, got, expected, diff);
    }
    println!("max_diff = {:.2e}", max_diff);
}
