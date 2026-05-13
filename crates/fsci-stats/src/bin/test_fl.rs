fn main() {
    use fsci_stats::{ContinuousDistribution, FatigueLife};
    for (c, ws, wk) in [
        (1.5_f64, 3.098072701327122_f64, 14.468691212039774_f64),
        (1.0, 2.30940108_f64, 7.66666667_f64),
        (0.5, 1.7320508_f64, 4.3137255_f64),
    ] {
        let d = FatigueLife::new(c);
        println!("c={c}: skew={:.10} (want {ws:.10}) ds={:.2e}  kurt={:.10} (want {wk:.10}) dk={:.2e}",
            d.skewness(), d.kurtosis(), (d.skewness()-ws).abs(), (d.kurtosis()-wk).abs());
    }
}
