fn main() {
    use fsci_stats::{ContinuousDistribution, RecipInvGauss};
    for mu in [0.5_f64, 1.0, 2.0, 5.0] {
        let d = RecipInvGauss::new(mu);
        println!("mu={mu}: mean={:.10} var={:.10} skew={:.10} kurt={:.10}",
            d.mean(), d.var(), d.skewness(), d.kurtosis());
    }
}
