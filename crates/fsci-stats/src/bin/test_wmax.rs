fn main() {
    use fsci_stats::{ContinuousDistribution, WeibullMax};
    let d = WeibullMax::new(2.0);
    println!("c=2: skew={:.10} (scipy -0.6311106578189344)  kurt={:.10} (scipy 0.24508930068764556)", d.skewness(), d.kurtosis());
    let d = WeibullMax::new(3.0);
    println!("c=3: skew={:.10}  kurt={:.10}", d.skewness(), d.kurtosis());
}
