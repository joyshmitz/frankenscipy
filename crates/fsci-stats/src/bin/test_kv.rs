fn main() {
    use fsci_runtime::RuntimeMode;
    use fsci_special::types::SpecialTensor;
    let kv = |order: f64, x: f64| -> f64 {
        let r = fsci_special::kv(
            &SpecialTensor::RealScalar(order),
            &SpecialTensor::RealScalar(x),
            RuntimeMode::Strict,
        );
        match r {
            Ok(SpecialTensor::RealScalar(v)) => v,
            _ => f64::NAN
        }
    };
    println!("K_{{0.25}}(0.197) = {}", kv(0.25, 0.197));
    println!("K_{{0.75}}(0.197) = {}", kv(0.75, 0.197));
    println!("K_{{1.25}}(0.197) = {} (NaN due to iv sign bug)", kv(1.25, 0.197));
    // Workaround: K_{ν+1}(z) = K_{ν-1}(z) + (2ν/z) K_ν(z), with ν=1/4:
    // K_{5/4}(z) = K_{3/4}(z) + (1/(2z)) K_{1/4}(z)
    let z = 0.197f64;
    let workaround = kv(0.75, z) + kv(0.25, z) / (2.0 * z);
    println!("K_{{1.25}}(0.197) via recurrence = {}", workaround);
    println!("expected scipy = 8.000746979880");
}
