use fsci_stats::{Argus, ContinuousDistribution};

fn main() {
    let a = Argus::new(1.0);
    println!("pdf = {}", a.pdf(0.5));
    println!("cdf = {}", a.cdf(0.5));
}
