use fsci_spatial::Delaunay;
fn main() {
    let points = vec![(0.0, 0.0), (1.0, 0.0), (f64::NAN, f64::NAN), (0.0, 1.0)];
    let delaunay = Delaunay::new(&points);
    println!("{:?}", delaunay.is_err());
}
