fn main() {
    use fsci_stats::{ContinuousDistribution, TukeyLambda};
    for lam in [-0.2_f64, 0.0, 0.5, 1.0, -0.1] {
        let d = TukeyLambda::new(lam);
        println!("λ={lam}: mean={:.10} var={:.10} skew={:.10} kurt={:.10}",
            d.mean(), d.var(), d.skewness(), d.kurtosis());
    }
}
