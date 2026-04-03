use fsci_stats::{ContinuousDistribution, Rice};

fn main() {
    let r = Rice::new(1.0);
    let cdf = r.cdf(f64::NAN);
    println!("CDF is: {:?}", cdf);
    assert!(cdf.is_nan());
}
