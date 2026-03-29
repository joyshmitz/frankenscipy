use fsci_stats::{ContinuousDistribution, LaplaceAsymmetric};

fn main() {
    let a = LaplaceAsymmetric::new(2.0);
    println!("pdf = {}", a.pdf(0.5));
    println!("cdf = {}", a.cdf(0.5));
    println!("mean = {}", a.mean());
    println!("var = {}", a.var());
}
