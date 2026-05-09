fn main() {
    use fsci_runtime::RuntimeMode;
    use fsci_special::types::SpecialTensor;
    let cases: [(f64, f64, f64, f64); 12] = [
        (1.0, 2.0, 3.0, -0.99),
        (1.0, 2.0, 3.0, -0.95),
        (1.0, 2.0, 3.0, -0.9),
        (1.0, 2.0, 3.0, -0.5),
        (1.0, 2.0, 3.0, -0.1),
        (1.0, 2.0, 3.0, -0.01),
        (1.0, 2.0, 3.0, -0.001),
        (1.0, 2.0, 3.0, -1e-6),
        (1.0, 2.0, 3.0, 0.001),
        (1.0, 2.0, 3.0, 0.5),
        (1.0, 2.0, 3.0, 0.95),
        (1.0, 2.0, 3.0, 0.99),
    ];
    for (a, b, c, z) in cases {
        let a_t = SpecialTensor::RealScalar(a);
        let b_t = SpecialTensor::RealScalar(b);
        let c_t = SpecialTensor::RealScalar(c);
        let z_t = SpecialTensor::RealScalar(z);
        let res = fsci_special::hyp2f1(&a_t, &b_t, &c_t, &z_t, RuntimeMode::Strict);
        let got = match res {
            Ok(SpecialTensor::RealScalar(v)) => v,
            Ok(other) => {
                println!("ERR unexpected output {:?}", other);
                continue;
            }
            Err(e) => {
                println!("ERR {:?}", e);
                continue;
            }
        };
        let expected = -2.0 * (z + (1.0 - z).ln()) / (z * z);
        let diff = (got - expected).abs();
        println!(
            "hyp2f1({:>4}, {:>4}, {:>4}, {:>9}) = {:>20.15} (closed: {:>20.15}, diff {:.2e})",
            a, b, c, z, got, expected, diff
        );
    }
}
