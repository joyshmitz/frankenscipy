use fsci_stats::*;
fn p(name: &str, a: f64, b: f64, x: f64, v: f64) {
    println!("{name} {a} {b} {x:.17e} {v:.17e}");
}
fn main() {
    let chi = Chi::new(4.0);
    for &x in &[10.0, 15.0, 25.0] {
        p("chi", 4.0, 0.0, x, chi.sf(x));
    }
    let er = Erlang::new(3, 1.0);
    for &x in &[30.0, 60.0, 100.0] {
        p("erlang", 3.0, 1.0, x, er.sf(x));
    }
    let nak = Nakagami::new(2.0);
    for &x in &[5.0, 8.0, 12.0] {
        p("nakagami", 2.0, 0.0, x, nak.sf(x));
    }
    let mx = Maxwell::new(1.0);
    for &x in &[8.0, 12.0, 18.0] {
        p("maxwell", 0.0, 0.0, x, mx.sf(x));
    }
    let ln = Lognormal::new(1.0, 1.0);
    for &x in &[1.0e4, 1.0e6, 1.0e9] {
        p("lognorm", 1.0, 0.0, x, ln.sf(x));
    }
    let go = Gompertz::new(1.0);
    for &x in &[5.0, 8.0, 12.0] {
        p("gompertz", 1.0, 0.0, x, go.sf(x));
    }
    let ge = GenExtreme::new(0.0);
    for &x in &[20.0, 40.0, 80.0] {
        p("genextreme", 0.0, 0.0, x, ge.sf(x));
    }
    let ri = Rice::new(1.0);
    for &x in &[8.0, 12.0, 18.0] {
        p("rice", 1.0, 0.0, x, ri.sf(x));
    }
    let fn_ = FoldedNormal::new(1.0);
    for &x in &[8.0, 15.0, 25.0] {
        p("foldnorm", 1.0, 0.0, x, fn_.sf(x));
    }
    let en = ExponNorm::new(1.5);
    for &x in &[15.0, 30.0, 60.0] {
        p("exponnorm", 1.5, 0.0, x, en.sf(x));
    }
}
