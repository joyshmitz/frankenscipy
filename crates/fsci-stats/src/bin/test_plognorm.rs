fn main() {
    use fsci_stats::{ContinuousDistribution, PowerLognorm};
    let cases = [
        (1.0_f64, 0.5_f64, 1.1331484531_f64, 0.3646958540_f64),
        (2.0, 0.7, 0.7929147157, 0.2297708738),
        (0.5, 0.5, 1.7370995111, 1.5754546415),
        (5.0, 0.3, 0.7195389147, 0.0200597815),
    ];
    for (c, s, em, ev) in cases {
        let d = PowerLognorm::new(c, s);
        let m = d.mean();
        let v = d.var();
        println!("c={c} s={s}: mean={m:.10} var={v:.10}  scipy mean={em:.10} var={ev:.10}  d_mean={:.2e} d_var={:.2e}",
                 (m-em).abs(), (v-ev).abs());
    }
}
