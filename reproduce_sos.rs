use fsci_special::{rel_entr, SpecialTensor};
use fsci_runtime::RuntimeMode;

fn main() {
    let nan = f64::NAN;
    let x = SpecialTensor::RealScalar(nan);
    let y = SpecialTensor::RealScalar(1.0);
    let res = rel_entr(&x, &y, RuntimeMode::Strict).expect("rel_entr(NaN, 1)");
    println!("rel_entr(NaN, 1) = {:?}", res);

    let x_vec = SpecialTensor::RealVec(vec![nan, 1.0]);
    let y_vec = SpecialTensor::RealVec(vec![1.0, nan]);
    let res_vec = rel_entr(&x_vec, &y_vec, RuntimeMode::Strict).expect("rel_entr([NaN, 1], [1, NaN])");
    println!("rel_entr([NaN, 1], [1, NaN]) = {:?}", res_vec);
    
    // Check broadcasting which I suspected is broken
    let y_scalar = SpecialTensor::RealScalar(1.0);
    let res_broad = rel_entr(&x_vec, &y_scalar, RuntimeMode::Strict);
    println!("rel_entr(vec, scalar) = {:?}", res_broad);
}
