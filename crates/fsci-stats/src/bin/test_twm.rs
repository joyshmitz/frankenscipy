fn main() {
    use fsci_stats::{ContinuousDistribution, TruncWeibullMin};
    for (c, a, b, em, ev) in [
        (2.0_f64, 0.1_f64, 2.0_f64, 0.8693606096581452_f64, 0.1790069975796863_f64),
        (1.0, 0.0, 1.0, 0.4180232931306739_f64, 0.0772943535975068_f64),
        (3.0, 0.5, 1.5, 0.9694796728_f64, 0.0768108919_f64),
    ] {
        let d = TruncWeibullMin::new(c, a, b);
        let m = d.mean();
        let v = d.var();
        println!("c={c} a={a} b={b}: mean={m:.10} (scipy {em:.10})  var={v:.10} (scipy {ev:.10})  dm={:.2e} dv={:.2e}",
            (m-em).abs(), (v-ev).abs());
    }
}
