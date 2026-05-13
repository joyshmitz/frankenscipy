fn main() {
    use fsci_stats::{ContinuousDistribution, Burr3};
    for (c, d, em, ev) in [
        (3.0_f64, 2.0_f64, 1.6122661015415272_f64, 1.431263271673902_f64),
        (5.0, 3.0, 1.0820883294_f64, 0.1554116499_f64),
        (2.5, 1.5, 1.5707963267948966_f64, f64::NAN),
        (1.5, 2.0, f64::NAN, f64::NAN),
        (0.5, 2.0, f64::NAN, f64::NAN),
    ] {
        let dist = Burr3::new(c, d);
        let m = dist.mean();
        let v = dist.var();
        let dm = if em.is_nan() { f64::from(!m.is_nan()) } else { (m - em).abs() };
        let dv = if ev.is_nan() { f64::from(!v.is_nan()) } else { (v - ev).abs() };
        println!("c={c} d={d}: mean={m:.10} (scipy {em:.10})  var={v:.10} (scipy {ev:.10})  dm={dm:.2e} dv={dv:.2e}");
    }
}
