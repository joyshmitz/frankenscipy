fn main() {
    use fsci_stats::{ContinuousDistribution, ExponPow};
    for (b, em, ev) in [
        (1.5_f64, 0.6648334123684801_f64, 0.11514883812803839_f64),
        (2.0, 0.7157241036160373, 0.0840863698206612),
        (1.0, 0.5963473623231939, 0.1925187894854262),
        (3.0, 0.7842008137_f64, 0.0521810502_f64),
    ] {
        let d = ExponPow::new(b);
        let m = d.mean();
        let v = d.var();
        println!("b={b}: mean={m:.10} (scipy {em:.10})  var={v:.10} (scipy {ev:.10})  dm={:.2e} dv={:.2e}",
            (m-em).abs(), (v-ev).abs());
    }
}
