use fsci_spatial::{DistanceMetric, pdist};
fn main() {
    println!("Testing pdist with empty vector");
    let res1 = pdist(&[], DistanceMetric::Euclidean);
    println!("res1: {:?}", res1);

    println!("Testing pdist with single empty vector");
    let res2 = pdist(&[vec![]], DistanceMetric::Euclidean);
    println!("res2: {:?}", res2);

    println!("Testing pdist with two empty vectors");
    let res3 = pdist(&[vec![], vec![]], DistanceMetric::Euclidean);
    println!("res3: {:?}", res3);
}
