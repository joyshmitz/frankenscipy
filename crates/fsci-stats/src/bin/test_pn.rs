fn main() {
    use fsci_stats::{ContinuousDistribution, PowerNorm};
    for c in [0.5_f64, 1.0, 2.0, 4.0] {
        let p = PowerNorm::new(c);
        println!("c={c}: mean={:.6} var={:.6} skew={:.6} kurt={:.6}",
            p.mean(), p.var(), p.skewness(), p.kurtosis());
    }
}
