use fsci_stats::ChiSquared;
use fsci_stats::ContinuousDistribution;

fn main() {
    let chi2 = ChiSquared::new(1.0);
    println!("cdf(2.0) = {}", chi2.cdf(2.0));
    println!("sf(2.0)  = {}", chi2.sf(2.0));
    let chi2_large = ChiSquared::new(10.0);
    println!("cdf(15.0) = {}", chi2_large.cdf(15.0));
    println!("sf(15.0)  = {}", chi2_large.sf(15.0));
}
