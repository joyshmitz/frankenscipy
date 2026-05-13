fn main() {
    use fsci_stats::{ContinuousDistribution, KsTwoBign};
    let d = KsTwoBign;
    println!("KsTwoBign: mean={:.10} var={:.10} skew={:.10} kurt={:.10}",
        d.mean(), d.var(), d.skewness(), d.kurtosis());
}
