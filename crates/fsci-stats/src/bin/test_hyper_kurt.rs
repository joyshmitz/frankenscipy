fn main() {
    use fsci_stats::{DiscreteDistribution, Hypergeometric};
    let cases = [
        (50_u64, 20_u64, 10_u64, -0.13162123226950354_f64),
        (20, 7, 12, -0.043_f64),
        (52, 4, 5, -0.1_f64),
        (10, 3, 5, -0.5_f64),
    ];
    for (m, n, big_n, _expected) in cases {
        let h = Hypergeometric::new(m, n, big_n);
        println!("Hypergeometric({}, {}, {}).kurtosis() = {}", m, n, big_n, h.kurtosis());
    }
}
