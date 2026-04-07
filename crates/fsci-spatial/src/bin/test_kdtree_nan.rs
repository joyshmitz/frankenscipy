use fsci_spatial::KDTree;
fn main() {
    let points = vec![vec![1.0, 2.0], vec![f64::NAN, f64::NAN], vec![3.0, 4.0]];
    let tree = KDTree::new(&points);
    println!("{:?}", tree.is_err());
}
