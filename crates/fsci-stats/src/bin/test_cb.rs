use fsci_stats::{ContinuousDistribution, CrystalBall};

fn main() {
    let cb = CrystalBall::new(2.0, 3.0);
    println!("pdf = {}", cb.pdf(0.0));
    println!("cdf = {}", cb.cdf(0.0));
}
