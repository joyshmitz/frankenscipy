use fsci_special::{rel_entr, SpecialTensor};
use fsci_runtime::RuntimeMode;

#[test]
fn test_repro_sos() {
    let nan = f64::NAN;
    let x = SpecialTensor::RealScalar(nan);
    let y = SpecialTensor::RealScalar(1.0);
    let res = rel_entr(&x, &y, RuntimeMode::Strict).expect("rel_entr(NaN, 1)");
    if let SpecialTensor::RealScalar(v) = res {
        assert!(v.is_nan());
    } else {
        panic!("expected scalar");
    }

    let x_vec = SpecialTensor::RealVec(vec![nan, 1.0]);
    let y_vec = SpecialTensor::RealVec(vec![1.0, nan]);
    let res_vec = rel_entr(&x_vec, &y_vec, RuntimeMode::Strict).expect("rel_entr([NaN, 1], [1, NaN])");
    if let SpecialTensor::RealVec(vs) = res_vec {
        assert!(vs[0].is_nan());
        assert!(vs[1].is_nan());
    } else {
        panic!("expected vector");
    }
    
    /*
    // Check broadcasting
    let y_scalar = SpecialTensor::RealScalar(1.0);
    let res_broad = rel_entr(&x_vec, &y_scalar, RuntimeMode::Strict);
    assert!(res_broad.is_ok(), "broadcasting should be supported: {:?}", res_broad.err());
    */

    // Check infinity cases
    let inf = f64::INFINITY;
    let res_inf = rel_entr(&SpecialTensor::RealScalar(1.0), &SpecialTensor::RealScalar(inf), RuntimeMode::Strict).expect("rel_entr(1, inf)");
    if let SpecialTensor::RealScalar(v) = res_inf {
        assert!(v.is_infinite(), "rel_entr(1, inf) should be inf, got {}", v);
    }

    let res_kl_inf = fsci_special::kl_div(1.0, inf);
    assert!(res_kl_inf.is_infinite(), "kl_div(1, inf) should be inf, got {}", res_kl_inf);
}
