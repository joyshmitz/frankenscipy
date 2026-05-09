fn main() {
    let cases = [
        (1.0_f64, 0.99_f64),
        (1.0, 0.999),
        (0.0, 0.99),
        (5.0, 0.5),
    ];
    println!("=== pdtri ===");
    for (k, p) in cases {
        let got = fsci_special::pdtri(k, p);
        println!("pdtri({}, {}) = {}", k, p, got);
    }
    println!("=== chdtri ===");
    for &(v, p) in &[(3.0_f64, 0.99_f64), (3.0, 0.5), (5.0, 0.95), (1.0, 0.999)] {
        let got = fsci_special::chdtri(v, p);
        println!("chdtri({}, {}) = {}", v, p, got);
    }
}
