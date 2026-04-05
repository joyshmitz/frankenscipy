use fsci_cluster::dbscan;

fn main() {
    let points = vec![vec![1.0, 2.0], vec![f64::NAN, f64::NAN], vec![3.0, 4.0]];
    let res = dbscan(&points, 1.0, 1);
    println!("{:?}", res);
}
