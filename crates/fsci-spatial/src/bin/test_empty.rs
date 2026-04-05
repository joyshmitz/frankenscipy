use fsci_spatial::KDTree;

fn main() {
    let points = vec![vec![1.0, 2.0], vec![f64::NAN, f64::NAN], vec![3.0, 4.0]];
    let tree = KDTree::new(&points).unwrap();
    let res = tree.query(&[2.0, 3.0]);
    println!("query: {:?}", res);
    let res_k = tree.query_k(&[2.0, 3.0], 2);
    println!("query_k: {:?}", res_k);
    let res_ball = tree.query_ball_point(&[2.0, 3.0], 10.0);
    println!("query_ball: {:?}", res_ball);
}
