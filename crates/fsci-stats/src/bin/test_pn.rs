fn main() {
    use fsci_stats::{ContinuousDistribution, GenHalfLogistic};
    for c in [0.5_f64, 1.0, 2.0] {
        let p = GenHalfLogistic::new(c);
        println!("c={c}: mean={:.10} var={:.10} skew={:.10} kurt={:.10}",
            p.mean(), p.var(), p.skewness(), p.kurtosis());
    }
}
