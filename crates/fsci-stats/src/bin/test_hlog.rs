fn main() {
    use fsci_stats::{ContinuousDistribution, HalfLogistic};
    let d = HalfLogistic;
    println!("mean={:.12}  scipy=1.3862943611198906", d.mean());
    println!("var={:.12}   scipy=1.3680560780236473", d.var());
    println!("skew={:.12}  scipy=1.5403288034048790", d.skewness());
    println!("kurt={:.12}  scipy=3.5837356644567153", d.kurtosis());
}
