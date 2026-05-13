fn main() {
    use fsci_stats::{ContinuousDistribution, Kappa3};
    for (a, em, ev) in [
        (0.5_f64, f64::NAN, f64::NAN),
        (1.0, f64::NAN, f64::NAN),
        (2.0, 2.0_f64.sqrt(), f64::NAN),
        (5.0, 0.7761272901, 0.3172223381),
        (10.0, 0.6492002378, 0.1587648372),
    ] {
        let d = Kappa3::new(a);
        let m = d.mean();
        let v = d.var();
        let dm = if em.is_nan() { f64::from(!m.is_nan()) } else { (m - em).abs() };
        let dv = if ev.is_nan() { f64::from(!v.is_nan()) } else { (v - ev).abs() };
        println!("a={a}: mean={m:.10} (scipy {em:.10})  var={v:.10} (scipy {ev:.10})  dm={dm:.2e} dv={dv:.2e}");
    }
}
