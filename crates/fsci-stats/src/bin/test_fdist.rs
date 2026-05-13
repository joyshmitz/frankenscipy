fn main() {
    use fsci_stats::{ContinuousDistribution, FDistribution, InverseGamma};
    println!("--- F-distribution ---");
    for (d1, d2, s, k) in [
        (5.0_f64, 10.0_f64, 3.867020319812938_f64, 50.86153846153847_f64),
        (10.0, 20.0, 1.8351920959819217, 6.893877551020408),
    ] {
        let f = FDistribution::new(d1, d2);
        println!("F({d1},{d2}) skew={:.10} (scipy {s:.10})  kurt={:.10} (scipy {k:.10})  ds={:.2e} dk={:.2e}",
            f.skewness(), f.kurtosis(), (f.skewness()-s).abs(), (f.kurtosis()-k).abs());
    }
    println!("--- InverseGamma ---");
    for (a, s, k) in [
        (4.0_f64, 5.656854249492381_f64, f64::NAN),
        (5.0, 3.4641016151377544, 42.0),
    ] {
        let g = InverseGamma::new(a);
        let ds = (g.skewness() - s).abs();
        let dk = if k.is_nan() { f64::from(!g.kurtosis().is_nan()) } else { (g.kurtosis()-k).abs() };
        println!("InvGamma({a}) skew={:.10} (scipy {s:.10})  kurt={:.10} (scipy {k:.10})  ds={ds:.2e} dk={dk:.2e}",
            g.skewness(), g.kurtosis());
    }
}
