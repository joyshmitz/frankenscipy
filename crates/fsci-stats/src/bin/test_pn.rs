fn main() {
    use fsci_stats::{ContinuousDistribution, TruncWeibullMin};
    for (c, a, b) in [(1.5_f64, 0.5_f64, 5.0_f64), (2.5, 0.1, 3.0), (3.0, 1.0, 10.0)] {
        let d = TruncWeibullMin::new(c, a, b);
        println!("c={c}, a={a}, b={b}: mean={:.10} var={:.10} skew={:.10} kurt={:.10}",
            d.mean(), d.var(), d.skewness(), d.kurtosis());
    }
}
