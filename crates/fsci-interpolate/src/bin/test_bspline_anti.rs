use fsci_interpolate::BSpline;

fn main() {
    let t = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let c = vec![1.0, 2.0, 3.0];
    let k = 2;
    let spl = BSpline::new(t, c, k).unwrap();
    let anti = spl.antiderivative(1).unwrap();
    println!("Knots: {:?}", anti.knots());
    println!("Coeffs: {:?}", anti.coeffs());
    println!("Degree: {}", anti.degree());
}
