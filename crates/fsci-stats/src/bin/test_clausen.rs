fn main() {
    use std::f64::consts::PI;
    // Reference Cl2 values computed via direct sum to N=10^6 (stable
    // to ~1e-12 between N=10^5 and N=10^6).
    let cases = [
        (0.1, 0.330272346694),
        (0.5, 0.848311869671),
        (1.0, 1.013959139535),
        (PI / 3.0, 1.014941606410),
        (PI / 2.0, 0.915965594177), // Catalan
        (2.0 * PI / 3.0, 0.676627737607),
        (PI, 0.0),
        (3.0 * PI / 2.0, -0.915965594177),
    ];
    let mut max_diff: f64 = 0.0;
    for (t, expected) in cases {
        let v = fsci_special::clausen(t);
        let diff = (v - expected).abs();
        max_diff = max_diff.max(diff);
        println!(
            "Cl2({:.6}) = {:.12}  exp: {:.12}  diff {:.2e}",
            t, v, expected, diff
        );
    }
    println!("max_diff = {:.2e}", max_diff);
}
