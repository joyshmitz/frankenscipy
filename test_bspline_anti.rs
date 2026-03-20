use fsci_interpolate::{BSpline};
fn main() {
    let t = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0];
    let c = vec![1.0, 2.0, 3.0, 4.0, 0.0];
    let k = 2;
    let b = BSpline::new(t, c, k).unwrap();
    let b_anti = b.antiderivative(1).unwrap();
    println!("rust anti c: {:?}", b_anti.c());
    println!("rust anti t: {:?}", b_anti.t());
}
