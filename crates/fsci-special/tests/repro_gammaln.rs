use fsci_special::{gammaln, SpecialTensor};
use fsci_runtime::RuntimeMode;

#[test]
fn test_gammaln_overflow() {
    // gammaln(200) should be finite
    let x = SpecialTensor::RealScalar(200.0);
    let res = gammaln(&x, RuntimeMode::Strict).expect("gammaln(200)");
    if let SpecialTensor::RealScalar(v) = res {
        println!("gammaln(200) = {}", v);
        assert!(v.is_finite(), "gammaln(200) should be finite, got {}", v);
        assert!((v - 857.933).abs() < 1.0, "gammaln(200) ≈ 857.9, got {}", v);
    } else {
        panic!("expected scalar");
    }
}
