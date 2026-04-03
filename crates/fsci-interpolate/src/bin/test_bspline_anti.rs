use fsci_interpolate::BSpline;

fn main() {
    let t = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let c = vec![1.0, f64::NAN, 3.0];
    let k = 2;
    let bspline = BSpline::new(t, c, k).unwrap();
    println!("{:?}", bspline.eval(0.5));
    println!("{:?}", bspline.eval(f64::NAN));
}
